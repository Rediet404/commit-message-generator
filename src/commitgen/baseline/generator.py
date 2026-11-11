"""Deterministic filename-based commit subject baseline."""

from __future__ import annotations

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


def generate_baseline(diff: str) -> str:
    """Generate a short deterministic subject using only the supplied diff."""
    paths = changed_paths(diff)
    return f"{_verb(diff, paths)} {_subject_path(paths)}"
