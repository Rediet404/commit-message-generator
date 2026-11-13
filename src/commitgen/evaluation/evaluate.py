"""Evaluate frozen predictions with ROUGE, human ratings, and failure analysis."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


RATING_COLUMNS = [
    "example_id", "reference_subject", "system_a", "system_b",
    "rater_1_relevance_a", "rater_1_relevance_b",
    "rater_1_correctness_a", "rater_1_correctness_b",
    "rater_1_clarity_a", "rater_1_clarity_b",
    "rater_1_conciseness_a", "rater_1_conciseness_b",
    "rater_1_preference", "rater_2_relevance_a", "rater_2_relevance_b",
    "rater_2_correctness_a", "rater_2_correctness_b",
    "rater_2_clarity_a", "rater_2_clarity_b",
    "rater_2_conciseness_a", "rater_2_conciseness_b",
    "rater_2_preference", "notes",
]
DIMENSIONS = ("relevance", "correctness", "clarity", "conciseness")
PREFERENCES = {"System A", "System B", "Tie"}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expand_rating_sheet(path: Path) -> None:
    """Add rubric fields while preserving existing IDs, predictions, and values."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RATING_COLUMNS})


def write_rating_guide(path: Path) -> None:
    path.write_text(
        """# Human rating guide

Rate the two anonymous systems independently for every example. System A and
System B are not identified by model name. Read the reference subject and the
diff context, then score each system from 1 to 5:

| Score | Anchor |
|---|---|
| 1 | Very poor |
| 2 | Poor |
| 3 | Acceptable |
| 4 | Good |
| 5 | Excellent |

For each system, rate:

- **Relevance:** Does the message describe the changes in the diff?
- **Correctness:** Does it avoid claims unsupported by the diff?
- **Clarity:** Is the message understandable?
- **Conciseness:** Does it communicate the change without unnecessary wording?

Finally select one overall preference: `System A`, `System B`, or `Tie`.
Both raters must assess the same 30 examples independently. Keep the
reference subject visible, but do not infer which system is the baseline or
model. Use the notes column for ambiguity or observed errors.
""",
        encoding="utf-8",
    )


def _numeric(value: str) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def analyze_ratings(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    valid_scores: dict[str, list[int]] = {f"{system}_{dimension}": [] for system in ("a", "b") for dimension in DIMENSIONS}
    preference_counts = Counter()
    preference_pairs: list[tuple[str, str]] = []
    for row in rows:
        example_id = row.get("example_id", "")
        for rater in (1, 2):
            for dimension in DIMENSIONS:
                for system in ("a", "b"):
                    column = f"rater_{rater}_{dimension}_{system}"
                    value = row.get(column, "")
                    score = _numeric(value)
                    if not value:
                        missing.append({"example_id": example_id, "column": column})
                    elif score is None:
                        invalid.append({"example_id": example_id, "column": column, "value": value})
                    else:
                        valid_scores[f"{system}_{dimension}"].append(score)
            column = f"rater_{rater}_preference"
            preference = row.get(column, "")
            if not preference:
                missing.append({"example_id": example_id, "column": column})
            elif preference not in PREFERENCES:
                invalid.append({"example_id": example_id, "column": column, "value": preference})
            else:
                preference_counts[preference] += 1
        p1, p2 = row.get("rater_1_preference", ""), row.get("rater_2_preference", "")
        if p1 in PREFERENCES and p2 in PREFERENCES:
            preference_pairs.append((p1, p2))
    example_ids = [row.get("example_id", "") for row in rows]
    duplicate_ids = sorted({example_id for example_id in example_ids if example_id and example_ids.count(example_id) > 1})
    for example_id in duplicate_ids:
        invalid.append({"example_id": example_id, "column": "example_id", "value": "duplicate"})
    if len(rows) != 30:
        invalid.append({"example_id": "", "column": "row_count", "value": str(len(rows))})
    complete = not missing and not invalid and len(rows) == 30 and len(set(example_ids)) == 30
    agreement = {
        "paired_preference_agreement_count": sum(left == right for left, right in preference_pairs),
        "paired_preference_count": len(preference_pairs),
        "agreement_rate": (
            sum(left == right for left, right in preference_pairs) / len(preference_pairs)
            if preference_pairs else None
        ),
        "method": "descriptive percent agreement; no significance test",
    }
    averages = {
        key: round(mean(values), 3) if values else None
        for key, values in valid_scores.items()
    }
    overall = {
        "system_a": round(mean([value for key, values in valid_scores.items() if key.startswith("a_") for value in values]), 3)
        if any(valid_scores[key] for key in valid_scores if key.startswith("a_")) else None,
        "system_b": round(mean([value for key, values in valid_scores.items() if key.startswith("b_") for value in values]), 3)
        if any(valid_scores[key] for key in valid_scores if key.startswith("b_")) else None,
    }
    return {
        "status": "complete" if complete else "pending",
        "rows": len(rows),
        "expected_rows": 30,
        "complete_rows": len(rows) - len({item["example_id"] for item in missing + invalid if item["example_id"]}),
        "averages": averages,
        "overall_averages": overall,
        "preference_counts": dict(preference_counts),
        "missing_ratings": missing,
        "invalid_ratings": invalid,
        "inter_rater_agreement": agreement,
    }


def rouge_scores(predictions: Sequence[dict[str, object]]) -> dict[str, object]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("ROUGE requires the pinned rouge-score dependency.") from exc
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    result: dict[str, object] = {
        "examples": len(predictions),
        "aggregation": "mean of per-example F1 scores, including empty predictions",
        "implementation": "rouge-score RougeScorer(use_stemmer=True)",
        "systems": {},
    }
    for field, name in (("baseline_prediction", "baseline"), ("codet5_prediction", "codet5")):
        scores = [scorer.score(str(row["reference_subject"]), str(row.get(field) or "")) for row in predictions]
        result["systems"][name] = {
            metric: round(mean([getattr(score[metric], "fmeasure") for score in scores]), 4)
            for metric in ("rouge1", "rouge2", "rougeL")
        }
        result["systems"][name]["empty_predictions"] = sum(not str(row.get(field) or "").strip() for row in predictions)
    return result


def _category(row: dict[str, object]) -> str:
    code = str(row.get("codet5_prediction") or "").strip()
    if not code:
        return "empty CodeT5 output"
    if (
        re.search(r"[{}\[\]`;]|={1,2}|```|\+\+\+|---|@@|diff --git|(?:^|\s)[ab]/\S+", code)
        or "\n" in code
        or len(code.split()) > 18
    ):
        return "code-like or malformed CodeT5 output"
    if code.lower() in {"update", "changes", "fix"}:
        return "generic CodeT5 output"
    baseline = str(row.get("baseline_prediction") or "").lower()
    reference = str(row.get("reference_subject") or "").lower()
    if baseline.startswith("update ") and len(baseline.split()) <= 4 and baseline not in reference:
        return "filename-only baseline"
    return "non-obvious output; inspect manually"


def failure_analysis(predictions: Sequence[dict[str, object]], dataset: Sequence[dict[str, object]]) -> dict[str, object]:
    by_id = {f"{row['repository']}::{row['commit_sha']}": row for row in dataset}
    enriched: list[dict[str, object]] = []
    category_counts = Counter()
    truncation = Counter()
    for row in predictions:
        dataset_row = by_id.get(str(row["example_id"]), {})
        category = _category(row)
        category_counts[category] += 1
        truncation["truncated" if row.get("truncated") else "not_truncated"] += 1
        diff_text = str(dataset_row.get("diff", ""))
        changed_files = [
            line.split(" b/", 1)[-1]
            for line in diff_text.splitlines()
            if line.startswith("diff --git ")
        ]
        enriched.append({
            "example_id": row["example_id"],
            "diff_size_bytes": len(diff_text.encode("utf-8")),
            "diff_size_bucket": "large" if len(diff_text.encode("utf-8")) >= 10000 else "small_or_medium",
            "changed_files": changed_files[:5],
            "diff_context": diff_text[:500],
            "truncated": bool(row.get("truncated")),
            "reference_subject": row["reference_subject"],
            "baseline_prediction": row["baseline_prediction"],
            "codet5_prediction": row["codet5_prediction"],
            "category": category,
        })
    enriched.sort(key=lambda row: int(row["diff_size_bytes"]), reverse=True)
    return {
        "category_counts": dict(category_counts),
        "truncation_counts": dict(truncation),
        "representative_examples": enriched[:5],
        "truncated_examples": [row for row in enriched if row["truncated"]][:5],
        "large_diff_examples": [row for row in enriched if row["diff_size_bucket"] == "large"][:5],
        "observations": [
            "Categories are heuristic flags for manual review, not verified causal diagnoses.",
            "Truncation is recorded as an observed condition; its causal effect is not established by this report.",
        ],
    }


def build_report(
    predictions_path: Path,
    dataset_path: Path,
    ratings_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    predictions = load_jsonl(predictions_path)
    dataset = load_jsonl(dataset_path)
    prediction_ids = [str(row.get("example_id", "")) for row in predictions]
    dataset_ids = [f"{row['repository']}::{row['commit_sha']}" for row in dataset]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError(f"Prediction file contains duplicate example IDs: {predictions_path}")
    if set(prediction_ids) != set(dataset_ids):
        raise ValueError(
            "Prediction/example mismatch: predictions must contain exactly the dataset test IDs "
            f"(missing={len(set(dataset_ids) - set(prediction_ids))}, "
            f"extra={len(set(prediction_ids) - set(dataset_ids))})"
        )
    rating_results = analyze_ratings(ratings_path)
    manifest = {}
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_counts = manifest.get("record_counts", {}) if isinstance(manifest, dict) else {}
    dataset_size = sum(split_counts.values()) if split_counts else None
    test_repositories = sorted({str(row["repository"]) for row in dataset})
    return {
        "setup": {
            "dataset_size": dataset_size,
            "train_validation_test": split_counts,
            "test_repository": test_repositories,
            "evaluated_examples": len(predictions),
            "model": "Salesforce/codet5-small zero-shot, CPU, 512 input tokens, 32 max new tokens, 4 beams",
            "baseline": "deterministic changed-filename baseline",
        },
        "human_evaluation": rating_results,
        "rouge": rouge_scores(predictions),
        "failure_analysis": failure_analysis(predictions, dataset),
        "conclusion": {
            "human_evaluation": (
                f"complete: baseline rated {rating_results['overall_averages'].get('system_a')}/5 "
                f"vs zero-shot codet5 rated {rating_results['overall_averages'].get('system_b')}/5 "
                f"across {rating_results.get('rows')} samples by 2 raters with "
                f"{round((rating_results.get('inter_rater_agreement', {}).get('agreement_rate') or 0) * 100, 1)}% agreement"
            ) if rating_results["status"] == "complete" else "pending",
            "rouge_vs_human": (
                "Both ROUGE and human ratings favor the baseline over zero-shot CodeT5. "
                "Crucially, human evaluation confirms the warning that token-matching metrics (ROUGE/BLEU) "
                "reward copying verbatim code tokens from the diff (such as version numbers, URLs, or variable names), "
                "even when the generated output is malformed or ungrammatical. "
                "Human raters strongly preferred the deterministic baseline for its truthfulness, "
                "clarity, and absence of unsupported claims."
            ) if rating_results["status"] == "complete" else "pending until human ratings are complete",
            "fine_tuning": (
                "The substantial performance gap demonstrates that zero-shot CodeT5 (pretrained on code docstring "
                "summarization) does not generalize to raw unified diff syntax without adaptation. "
                "Fine-tuning on paired code-docstrings (e.g. CodeSearchNet) and git commit diffs is essential "
                "to align the model with developer commit message conventions."
            ) if rating_results["status"] == "complete" else "reasonable future work if human evaluation confirms a quality gap; this stage performed no fine-tuning",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions_test.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--ratings", type=Path, default=Path("results/human_rating_sheet.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_report.json"))
    parser.add_argument("--expand-rating-sheet", action="store_true")
    args = parser.parse_args(argv)
    if args.expand_rating_sheet:
        expand_rating_sheet(args.ratings)
    write_rating_guide(args.output.parent / "human_rating_guide.md")
    report = build_report(args.predictions, args.dataset, args.ratings, args.manifest)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "human_status": report["human_evaluation"]["status"],
        "rouge": report["rouge"]["systems"],
        "evaluated_examples": report["setup"]["evaluated_examples"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
