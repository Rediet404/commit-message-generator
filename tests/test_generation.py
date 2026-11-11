import json
from pathlib import Path

from commitgen.baseline.generator import changed_paths, generate_baseline
from commitgen.model.experiment import run_predictions, write_rating_sheet
from commitgen.model.zeroshot import CodeT5Generator, GenerationSettings, Prediction


ADDED_DIFF = """diff --git a/src/auth.py b/src/auth.py
new file mode 100644
--- /dev/null
+++ b/src/auth.py
@@ -0,0 +1 @@
+def login(): pass
"""
MODIFIED_DIFF = """diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -1 +1 @@
-return False
+return True
"""
DELETED_DIFF = """diff --git a/src/old.py b/src/old.py
deleted file mode 100644
--- a/src/old.py
+++ /dev/null
@@ -1 +0,0 @@
-old()
"""


def test_baseline_handles_added_modified_and_deleted_files() -> None:
    assert changed_paths(ADDED_DIFF) == ["src/auth.py"]
    assert generate_baseline(ADDED_DIFF) == "Add auth"
    assert generate_baseline(MODIFIED_DIFF) == "Update auth"
    assert generate_baseline(DELETED_DIFF) == "Remove old"


def test_baseline_falls_back_for_empty_or_malformed_diff() -> None:
    assert generate_baseline("") == "Update project files"
    assert generate_baseline("not a unified diff") == "Update project files"


def test_prepare_input_reports_original_length_and_truncation() -> None:
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            length = 8 if kwargs.get("truncation") else 20
            return {"input_ids": FakeTensor(length)}

    class FakeTensor:
        def __init__(self, length):
            self.shape = (1, length)

    generator = CodeT5Generator(
        GenerationSettings(max_input_tokens=10),
        model=object(),
        tokenizer=FakeTokenizer(),
    )
    _, input_tokens, truncated = generator.prepare_input("diff")
    assert input_tokens == 20
    assert truncated is True


def test_prediction_serialization_and_shared_example_id(tmp_path: Path) -> None:
    class FakeGenerator:
        def generate(self, diff):
            return Prediction("model subject", 12, False, 0.1, "ok")

    examples = [{
        "repository": "example/repo",
        "commit_sha": "abc",
        "commit_subject": "Reference",
        "diff": MODIFIED_DIFF,
    }]
    records = run_predictions(examples, FakeGenerator())
    assert records[0]["example_id"] == "example/repo::abc"
    output = tmp_path / "ratings.csv"
    write_rating_sheet(output, records)
    assert "System" not in output.read_text(encoding="utf-8")
    assert "model subject" in output.read_text(encoding="utf-8")
    json.dumps(records)


def test_generation_failure_is_recorded() -> None:
    class FailingGenerator:
        def generate(self, diff):
            return Prediction("", 0, False, 0.01, "error", "network unavailable")

    records = run_predictions(
        [{"repository": "r", "commit_sha": "s", "commit_subject": "x", "diff": ""}],
        FailingGenerator(),
    )
    assert records[0]["codet5_status"] == "error"
    assert records[0]["codet5_error"] == "network unavailable"
