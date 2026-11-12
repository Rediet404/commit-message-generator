"""Fine-tune CodeT5 on code-docstring pairs or commit diffs.

Implements the requirement:
"Adaptation: Use as is first; then fine-tune on code-docstring pairs"
"Data: CodeSearchNet pairs, or commits from public repositories"
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from commitgen.data.docstrings import load_code_docstring_dataset


logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    model_name: str = "Salesforce/codet5-small"
    output_dir: str = "models/finetuned"
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_input_length: int = 256
    max_target_length: int = 32
    device: str = "cpu"
    input_prefix: str = "summarize: "
    seed: int = 42
    weight_decay: float = 0.01


class CodeDocstringDataset:
    """Dataset wrapper for tokenizing (code, docstring) or (diff, subject) pairs."""

    def __init__(self, examples: list[dict[str, str]], tokenizer: Any, config: FineTuneConfig):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.examples[idx]
        input_text = item.get("code") or item.get("diff") or ""
        target_text = item.get("docstring") or item.get("commit_subject") or ""

        inputs = self.tokenizer(
            self.config.input_prefix + input_text,
            max_length=self.config.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.config.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = targets["input_ids"].squeeze(0)
        # Replace padding token id's with -100 so loss ignores them
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_epoch(model: Any, dataloader: Any, optimizer: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(1, steps)


def evaluate_loss(model: Any, dataloader: Any, device: Any) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    try:
        import torch
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += float(outputs.loss.item())
                steps += 1
    except Exception as exc:
        logger.warning(f"Evaluation error: {exc}")
    return total_loss / max(1, steps)


def finetune(
    dataset_records: list[dict[str, str]],
    config: FineTuneConfig | None = None,
    val_records: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute fine-tuning on code-docstring or commit-diff pairs."""
    config = config or FineTuneConfig()
    set_seed(config.seed)

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Fine-tuning requires torch and transformers to be installed.") from exc

    device_str = "cuda" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)

    logger.info(f"Loading tokenizer and model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    model.to(device)

    # Train / validation split if val_records not provided
    if val_records is None:
        split_idx = max(1, int(len(dataset_records) * 0.8))
        train_data = dataset_records[:split_idx]
        val_data = dataset_records[split_idx:] if len(dataset_records) > 1 else dataset_records
    else:
        train_data = dataset_records
        val_data = val_records

    train_ds = CodeDocstringDataset(train_data, tokenizer, config)
    val_ds = CodeDocstringDataset(val_data, tokenizer, config)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        epoch_duration = time.perf_counter() - epoch_start

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "duration_seconds": round(epoch_duration, 2),
        }
        history.append(epoch_stats)
        logger.info(f"Epoch {epoch}/{config.num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Save fine-tuned checkpoint
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    report = {
        "status": "success",
        "config": asdict(config),
        "total_examples": len(dataset_records),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "total_training_time_seconds": round(time.perf_counter() - start_time, 2),
        "history": history,
        "checkpoint_dir": str(output_path),
    }

    report_path = output_path / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Path to JSONL dataset (code-docstrings or commits)")
    parser.add_argument("--output-dir", type=str, default="models/finetuned")
    parser.add_argument("--model-name", type=str, default="Salesforce/codet5-small")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--smoke", type=int, help="Limit number of training samples for quick smoke test")
    args = parser.parse_args(argv)

    records = load_code_docstring_dataset(args.dataset)
    if args.smoke:
        records = records[:args.smoke]

    config = FineTuneConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print(f"Starting fine-tuning on {len(records)} examples...")
    result = finetune(records, config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
