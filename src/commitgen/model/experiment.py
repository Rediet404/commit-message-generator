"""Run baseline and CodeT5 zero-shot predictions on frozen JSONL splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

from commitgen.baseline.generator import generate_baseline
from commitgen.model.zeroshot import CodeT5Generator, GenerationSettings


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prediction_record(example: dict[str, str], baseline_text: str, model_prediction: object, baseline_time: float) -> dict[str, object]:
    prediction = model_prediction
    return {
        "example_id": f"{example['repository']}::{example['commit_sha']}",
        "repository": example["repository"],
        "commit_sha": example["commit_sha"],
        "reference_subject": example["commit_subject"],
        "baseline_prediction": baseline_text,
        "codet5_prediction": prediction.text,
        "input_tokens": prediction.input_tokens,
        "truncated": prediction.truncated,
        "baseline_inference_time_seconds": baseline_time,
        "codet5_inference_time_seconds": prediction.inference_time_seconds,
        "codet5_status": prediction.status,
        "codet5_error": prediction.error,
    }


def run_predictions(examples: Iterable[dict[str, str]], generator: CodeT5Generator) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for example in examples:
        started = time.perf_counter()
        baseline_text = generate_baseline(example.get("diff", ""))
        baseline_time = time.perf_counter() - started
        results.append(_prediction_record(example, baseline_text, generator.generate(example.get("diff", "")), baseline_time))
    return results


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_rating_sheet(path: Path, records: list[dict[str, object]], sample_size: int = 30) -> None:
    selected = records[:sample_size]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("example_id,reference_subject,system_a,system_b,rater_1_score_a,rater_1_score_b,rater_2_score_a,rater_2_score_b,notes\n")
        for record in selected:
            fields = [
                record["example_id"],
                record["reference_subject"],
                record["baseline_prediction"],
                record["codet5_prediction"],
                "", "", "", "", "",
            ]
            handle.write(",".join('"' + str(field).replace('"', '""') + '"' for field in fields) + "\n")


def run_experiment(
    split_path: Path,
    output_dir: Path,
    smoke: int | None = None,
    model_name: str = "Salesforce/codet5-small",
    output_name: str | None = None,
) -> dict[str, object]:
    examples = load_jsonl(split_path)
    if smoke is not None:
        examples = examples[:smoke]
    generator = CodeT5Generator(GenerationSettings(model_name=model_name))
    records = run_predictions(examples, generator)
    suffix = output_name or ("smoke" if smoke is not None else "test")
    write_jsonl(output_dir / f"predictions_{suffix}.jsonl", records)
    write_rating_sheet(output_dir / "human_rating_sheet.csv", records)
    successful = sum(record["codet5_status"] == "ok" for record in records)
    return {"examples": len(records), "codet5_successful": successful, "codet5_failures": len(records) - successful}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--smoke", type=int)
    parser.add_argument("--model-name", default="Salesforce/codet5-small")
    parser.add_argument("--output-name", help="Prediction filename suffix, without predictions_ or .jsonl")
    args = parser.parse_args(argv)
    print(json.dumps(
        run_experiment(args.split, args.output_dir, args.smoke, args.model_name, args.output_name),
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
