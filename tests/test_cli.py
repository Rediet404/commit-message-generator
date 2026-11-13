from pathlib import Path

from commitgen.application.cli import (
    create_parser,
    generate_commit_suggestion,
    generate_docstring_suggestion,
)


DIFF_SAMPLE = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
"""

CODE_SAMPLE = """def calculate_total(prices):
    # Calculate sum of all prices
    return sum(prices)
"""


def test_cli_parser_creation() -> None:
    parser = create_parser()
    assert parser.prog == "commitgen"
    # Test subcommands
    commit_args = parser.parse_args(["commit", "--diff", "dummy.diff", "--model", "baseline", "-y"])
    assert commit_args.command == "commit"
    assert commit_args.model == "baseline"
    assert commit_args.yes is True
    assert parser.parse_args(["commit", "--model-path", "models/finetuned-commits"]).model_path == "models/finetuned-commits"

    doc_args = parser.parse_args(["docstring", "--code", "def f(): pass", "--model", "codet5", "-w"])
    assert doc_args.command == "docstring"
    assert doc_args.write is True


def test_generate_commit_suggestion_with_baseline() -> None:
    suggestion = generate_commit_suggestion(
        DIFF_SAMPLE,
        model="baseline",
        mood="imperative",
        length="short",
    )
    assert suggestion.startswith("Update app")


def test_generate_docstring_suggestion_with_baseline() -> None:
    suggestion = generate_docstring_suggestion(
        CODE_SAMPLE,
        model="baseline",
        mood="imperative",
        length="short",
    )
    assert "Calculate" in suggestion
