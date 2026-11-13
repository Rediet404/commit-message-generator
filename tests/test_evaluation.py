import csv
import json
from pathlib import Path

from commitgen.evaluation.evaluate import (
    RATING_COLUMNS,
    analyze_ratings,
    expand_rating_sheet,
    failure_analysis,
)


def test_rating_sheet_expansion_preserves_predictions(tmp_path: Path) -> None:
    path = tmp_path / "ratings.csv"
    path.write_text(
        "example_id,reference_subject,system_a,system_b,rater_1_score_a,rater_1_score_b,rater_2_score_a,rater_2_score_b,notes\n"
        '"id","ref","A","B","","","","",""\n',
        encoding="utf-8",
    )
    expand_rating_sheet(path)
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert list(row) == RATING_COLUMNS
    assert row["example_id"] == "id"
    assert row["system_a"] == "A"


def test_incomplete_ratings_are_pending_and_reported(tmp_path: Path) -> None:
    path = tmp_path / "ratings.csv"
    path.write_text(",".join(RATING_COLUMNS) + "\n" + ",".join(["id", "ref", "A", "B"] + [""] * 19) + "\n", encoding="utf-8")
    result = analyze_ratings(path)
    assert result["status"] == "pending"
    assert result["missing_ratings"]


def test_rating_sheet_with_wrong_row_count_is_pending(tmp_path: Path) -> None:
    path = tmp_path / "ratings.csv"
    values = ["id", "ref", "A", "B"] + ["3"] * 16 + ["System A", "3", ""] * 2
    path.write_text(",".join(RATING_COLUMNS) + "\n" + ",".join(values) + "\n", encoding="utf-8")

    result = analyze_ratings(path)

    assert result["status"] == "pending"
    assert any(item["column"] == "row_count" for item in result["invalid_ratings"])


def test_failure_analysis_records_truncation_and_categories() -> None:
    predictions = [{
        "example_id": "r::1",
        "reference_subject": "Fix auth",
        "baseline_prediction": "Update auth",
        "codet5_prediction": "```code```",
        "truncated": True,
    }]
    dataset = [{"repository": "r", "commit_sha": "1", "diff": "diff\n" * 10}]
    result = failure_analysis(predictions, dataset)
    assert result["truncation_counts"]["truncated"] == 1
    assert result["category_counts"]["code-like or malformed CodeT5 output"] == 1
