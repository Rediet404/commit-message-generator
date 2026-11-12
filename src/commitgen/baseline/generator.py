"""Deterministic baseline for Git diffs and function code.

Implements the requirement:
Template: 'Update <file>' (for diffs) or the first comment line (for function code).
"""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath


_PATH_PATTERN = re.compile(r"^(?:diff --git a/(.+?) b/(.+?)|--- (?:a/)?(.+)|\+\+\+ (?:b/)?(.+))$")
_IGNORED_NAMES = {"readme", "changelog", "license", "makefile"}


def changed_paths(diff: str) -> list[str]:
    """Extract unique paths from a unified Git diff."""
    paths: list[str] = []
    for line in diff.splitlines():
        match = _PATH_PATTERN.match(line)
        if not match:
            continue
        candidates = match.groups()
        path = next((candidate for candidate in candidates if candidate and candidate != "/dev/null"), None)
        if path and path not in paths:
            paths.append(path)
    return paths


def _verb(diff: str, paths: list[str]) -> str:
    if "new file mode" in diff:
        return "Add"
    if "deleted file mode" in diff:
        return "Remove"
    if not paths:
        return "Update"
    return "Update"


def _subject_path(paths: list[str]) -> str:
    if not paths:
        return "project files"
    if len(paths) == 1:
        path = PurePosixPath(paths[0])
        name = path.stem or path.name
        if name.lower() in _IGNORED_NAMES and path.parent != PurePosixPath("."):
            name = path.parent.name
        name = re.sub(r"[-_.]+", " ", name).strip()
        return name or "project files"
    suffixes = {PurePosixPath(path).suffix.lower() for path in paths if PurePosixPath(path).suffix}
    if len(suffixes) == 1:
        return f"{len(paths)} {suffixes.pop().lstrip('.')} files"
    return f"{len(paths)} files"


def generate_diff_baseline(diff: str) -> str:
    """Generate a short deterministic subject using the template: Update <file>."""
    paths = changed_paths(diff)
    return f"{_verb(diff, paths)} {_subject_path(paths)}"


def extract_first_comment(code: str) -> str | None:
    """Extract the first meaningful comment line or docstring from function code."""
    # Check for docstrings first using ast if parseable
    try:
        tree = ast.parse(code.strip())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                doc = ast.get_docstring(node)
                if doc:
                    first_line = doc.strip().splitlines()[0].strip()
                    if first_line:
                        return first_line
    except Exception:
        pass

    # Regex search for comments or docstring quotes
    for line in code.splitlines():
        stripped = line.strip()
        # Single line comment
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if comment and not comment.startswith("!") and not comment.startswith("-*-"):
                return comment
        # Docstring starting line
        if stripped.startswith('"""') or stripped.startswith("'''"):
            content = stripped.strip("\"'").strip()
            if content:
                return content
    return None


def extract_function_name(code: str) -> str | None:
    """Extract function name from code string if present."""
    match = re.search(r"^\s*(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(", code, re.MULTILINE)
    if match:
        name = match.group(1).replace("_", " ").strip()
        return name
    return None


def generate_code_baseline(code: str) -> str:
    """Generate baseline for function code: first comment line or function template."""
    first_comment = extract_first_comment(code)
    if first_comment:
        return first_comment
    func_name = extract_function_name(code)
    if func_name:
        return f"Execute {func_name}"
    return "Execute function"


def is_git_diff(text: str) -> bool:
    """Check if the given text represents a Git unified diff."""
    if not text.strip():
        return False
    diff_markers = ("diff --git ", "--- a/", "+++ b/", "@@ -", "new file mode", "deleted file mode")
    return any(marker in text for marker in diff_markers)


def generate_baseline(input_text: str, is_diff: bool | None = None) -> str:
    """Generate baseline one-line description from git diff or function code.
    
    If is_diff is None, automatically detects whether input is a diff or code.
    """
    if is_diff is True:
        return generate_diff_baseline(input_text)
    if is_diff is False:
        return generate_code_baseline(input_text)

    # Auto-detection
    if is_git_diff(input_text):
        return generate_diff_baseline(input_text)

    # If code markers present (def, class, etc.), treat as function code
    if re.search(r"\b(def|class|lambda|return)\b", input_text):
        return generate_code_baseline(input_text)

    # Fallback to diff baseline (handles empty strings or path-less inputs)
    return generate_diff_baseline(input_text)
