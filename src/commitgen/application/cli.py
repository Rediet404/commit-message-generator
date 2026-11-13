"""Interactive Git commit message and docstring writer CLI.

Implements Stage 9:
"Application: add the staged-diff CLI with explicit user confirmation before any commit."
And supports docstring writing for function code.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from commitgen.baseline.generator import generate_baseline, generate_code_baseline, generate_diff_baseline
from commitgen.model.zeroshot import CodeT5Generator, GenerationSettings
from commitgen.preprocessing.style import format_style


def get_staged_diff() -> str:
    """Retrieve staged Git changes using git diff --cached."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to execute git diff: {exc}") from exc


def get_diff_summary() -> str:
    """Get a short summary of staged changes (e.g. git diff --cached --stat)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _read_diff_argument(value: str) -> str:
    """Read a diff path, or treat the argument as a raw diff without path errors."""
    candidate = Path(value)
    try:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    except OSError:
        pass
    return value


def run_git_commit(message: str) -> bool:
    """Execute git commit -m <message>."""
    try:
        subprocess.run(["git", "commit", "-m", message], check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error during commit: {exc}", file=sys.stderr)
        return False


def generate_commit_suggestion(
    diff: str,
    model: str = "codet5",
    mood: str | None = "imperative",
    length: str | None = "short",
    model_path: str | None = None,
) -> str:
    """Generate a commit message using CodeT5 or deterministic baseline."""
    if model.lower() == "baseline":
        raw = generate_diff_baseline(diff)
    else:
        try:
            settings = GenerationSettings(
                model_name=model_path or "Salesforce/codet5-small",
                style_mood=mood,
                style_length=length,
            )
            generator = CodeT5Generator(settings)
            prediction = generator.generate_commit_message(diff)
            if prediction.status == "ok" and prediction.text.strip():
                raw = prediction.text.strip()
            else:
                print(
                    f"CodeT5 returned {prediction.status}; using the baseline fallback.",
                    file=sys.stderr,
                )
                raw = generate_diff_baseline(diff)
        except Exception as exc:
            print(f"CodeT5 unavailable ({exc}); falling back to baseline.", file=sys.stderr)
            raw = generate_diff_baseline(diff)

    return format_style(raw, mood=mood, length=length)


def generate_docstring_suggestion(
    code: str,
    model: str = "codet5",
    mood: str | None = "imperative",
    length: str | None = "short",
    model_path: str | None = None,
) -> str:
    """Generate a docstring description from Python function code."""
    if model.lower() == "baseline":
        raw = generate_code_baseline(code)
    else:
        try:
            settings = GenerationSettings(
                model_name=model_path or "Salesforce/codet5-small",
                style_mood=mood,
                style_length=length,
            )
            generator = CodeT5Generator(settings)
            prediction = generator.generate_docstring(code)
            if prediction.status == "ok" and prediction.text.strip():
                raw = prediction.text.strip()
            else:
                print(
                    f"CodeT5 returned {prediction.status}; using the baseline fallback.",
                    file=sys.stderr,
                )
                raw = generate_code_baseline(code)
        except Exception as exc:
            print(f"CodeT5 unavailable ({exc}); using the baseline fallback.", file=sys.stderr)
            raw = generate_code_baseline(code)

    return format_style(raw, mood=mood, length=length)


def handle_commit(args: argparse.Namespace) -> int:
    """Handle git commit message generation and interactive confirmation."""
    if args.diff:
        diff = _read_diff_argument(args.diff)
    else:
        diff = get_staged_diff()

    if not diff.strip():
        print("No staged changes detected.")
        print("Stage your changes first with 'git add <files>' or pass a diff with --diff.")
        return 1

    summary = get_diff_summary()
    if summary:
        print("Staged changes summary:")
        print(summary)
        print("-" * 40)

    suggestion = generate_commit_suggestion(
        diff,
        model=args.model,
        mood=args.mood,
        length=args.length,
        model_path=args.model_path,
    )

    print(f"\nProposed commit message:\n  {suggestion}\n")

    if args.yes:
        print("Auto-committing (--yes specified)...")
        success = run_git_commit(suggestion)
        return 0 if success else 1

    # Interactive confirmation
    try:
        response = input("Actions: [c]ommit / [e]dit / [a]bort [default: c]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1

    if response in ("", "c", "commit", "y", "yes"):
        success = run_git_commit(suggestion)
        if success:
            print("Successfully committed changes.")
            return 0
        return 1
    elif response in ("e", "edit"):
        try:
            custom_msg = input("Enter commit message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if not custom_msg:
            print("Empty message. Aborted.")
            return 1
        success = run_git_commit(custom_msg)
        if success:
            print("Successfully committed changes.")
            return 0
        return 1
    else:
        print("Commit aborted by user.")
        return 0


def handle_docstring(args: argparse.Namespace) -> int:
    """Handle docstring generation for code snippet or file."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}", file=sys.stderr)
            return 1
        code = file_path.read_text(encoding="utf-8")
    elif args.code:
        code = args.code
    else:
        print("Please provide --file <path> or --code <string>.", file=sys.stderr)
        return 1

    suggestion = generate_docstring_suggestion(
        code,
        model=args.model,
        mood=args.mood,
        length=args.length,
        model_path=args.model_path,
    )

    print(f"\nGenerated Docstring:\n  \"\"\"{suggestion}\"\"\"\n")

    # If file was passed and write was requested
    if args.file and args.write:
        print(f"File update requested for: {args.file}")
        # Insert docstring into first function missing docstring
        try:
            tree = ast.parse(code)
            inserted = False
            lines = code.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not ast.get_docstring(node):
                        first_body_line_idx = node.body[0].lineno - 1
                        indent = " " * (node.col_offset + 4)
                        doc_line = f'{indent}"""{suggestion}"""'
                        lines.insert(first_body_line_idx, doc_line)
                        inserted = True
                        break
            if inserted:
                Path(args.file).write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(f"Updated {args.file} with generated docstring.")
            else:
                print("All functions already have docstrings.")
        except Exception as exc:
            print(f"Could not parse and update file: {exc}", file=sys.stderr)
            return 1

    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commitgen",
        description="Docstring and commit-message writer powered by CodeT5 and transparent baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # commit command
    commit_parser = subparsers.add_parser("commit", help="Generate commit message from staged changes")
    commit_parser.add_argument("--diff", type=str, help="Path to diff file or raw diff string (defaults to git diff --cached)")
    commit_parser.add_argument("--model", choices=["codet5", "baseline"], default="codet5", help="Generation engine")
    commit_parser.add_argument("--model-path", help="Local CodeT5 checkpoint directory (optional)")
    commit_parser.add_argument("--mood", choices=["imperative", "descriptive"], default="imperative", help="Verb style")
    commit_parser.add_argument("--length", choices=["short", "detailed"], default="short", help="Length preference")
    commit_parser.add_argument("-y", "--yes", action="store_true", help="Auto-commit without confirmation")

    # docstring command
    doc_parser = subparsers.add_parser("docstring", help="Generate docstring for Python function code")
    doc_parser.add_argument("--code", type=str, help="Function code string")
    doc_parser.add_argument("--file", type=str, help="Path to Python file")
    doc_parser.add_argument("--model", choices=["codet5", "baseline"], default="codet5", help="Generation engine")
    doc_parser.add_argument("--model-path", help="Local CodeT5 checkpoint directory (optional)")
    doc_parser.add_argument("--mood", choices=["imperative", "descriptive"], default="imperative", help="Verb style")
    doc_parser.add_argument("--length", choices=["short", "detailed"], default="short", help="Length preference")
    doc_parser.add_argument("-w", "--write", action="store_true", help="Write docstring back to file")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "commit":
        return handle_commit(args)
    elif args.command == "docstring":
        return handle_docstring(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
