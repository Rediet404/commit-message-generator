"""Style classification, acceptance analysis, and transformation.

Addresses Project 2 requirements:
"GROWS INTO PROJECT 2: Learn which style developers accept: short or detailed,
imperative or descriptive."
"""

from __future__ import annotations

import re
from typing import Sequence


# Common verb mappings between imperative, third-person present, and past tense
_VERB_MAP: dict[str, dict[str, str]] = {
    "add": {"imperative": "Add", "descriptive": "Adds", "past": "Added"},
    "fix": {"imperative": "Fix", "descriptive": "Fixes", "past": "Fixed"},
    "update": {"imperative": "Update", "descriptive": "Updates", "past": "Updated"},
    "remove": {"imperative": "Remove", "descriptive": "Removes", "past": "Removed"},
    "delete": {"imperative": "Delete", "descriptive": "Deletes", "past": "Deleted"},
    "refactor": {"imperative": "Refactor", "descriptive": "Refactors", "past": "Refactored"},
    "support": {"imperative": "Support", "descriptive": "Supports", "past": "Supported"},
    "implement": {"imperative": "Implement", "descriptive": "Implements", "past": "Implemented"},
    "change": {"imperative": "Change", "descriptive": "Changes", "past": "Changed"},
    "make": {"imperative": "Make", "descriptive": "Makes", "past": "Made"},
    "allow": {"imperative": "Allow", "descriptive": "Allows", "past": "Allowed"},
    "avoid": {"imperative": "Avoid", "descriptive": "Avoids", "past": "Avoided"},
    "move": {"imperative": "Move", "descriptive": "Moves", "past": "Moved"},
    "improve": {"imperative": "Improve", "descriptive": "Improves", "past": "Improved"},
    "handle": {"imperative": "Handle", "descriptive": "Handles", "past": "Handled"},
    "clean": {"imperative": "Clean", "descriptive": "Cleans", "past": "Cleaned"},
    "bump": {"imperative": "Bump", "descriptive": "Bumps", "past": "Bumped"},
    "drop": {"imperative": "Drop", "descriptive": "Drops", "past": "Dropped"},
    "expose": {"imperative": "Expose", "descriptive": "Exposes", "past": "Exposed"},
    "adapt": {"imperative": "Adapt", "descriptive": "Adapts", "past": "Adapted"},
    "ensure": {"imperative": "Ensure", "descriptive": "Ensures", "past": "Ensured"},
    "enable": {"imperative": "Enable", "descriptive": "Enables", "past": "Enabled"},
    "disable": {"imperative": "Disable", "descriptive": "Disables", "past": "Disabled"},
    "create": {"imperative": "Create", "descriptive": "Creates", "past": "Created"},
    "set": {"imperative": "Set", "descriptive": "Sets", "past": "Set"},
    "use": {"imperative": "Use", "descriptive": "Uses", "past": "Used"},
    "show": {"imperative": "Show", "descriptive": "Shows", "past": "Showed"},
    "pass": {"imperative": "Pass", "descriptive": "Passes", "past": "Passed"},
    "return": {"imperative": "Return", "descriptive": "Returns", "past": "Returned"},
    "convert": {"imperative": "Convert", "descriptive": "Converts", "past": "Converted"},
    "rename": {"imperative": "Rename", "descriptive": "Renames", "past": "Renamed"},
    "prevent": {"imperative": "Prevent", "descriptive": "Prevents", "past": "Prevented"},
    "correct": {"imperative": "Correct", "descriptive": "Corrects", "past": "Corrected"},
    "simplify": {"imperative": "Simplify", "descriptive": "Simplifies", "past": "Simplified"},
}

# Reverse lookup for fast matching
_LOOKUP: dict[str, tuple[str, str]] = {}
for base, forms in _VERB_MAP.items():
    _LOOKUP[forms["imperative"].lower()] = (base, "imperative")
    _LOOKUP[forms["descriptive"].lower()] = (base, "descriptive")
    _LOOKUP[forms["past"].lower()] = (base, "descriptive")


def get_first_word(text: str) -> str:
    """Extract the first word of text, ignoring prefixes like '[docs]' or 'docs:'."""
    cleaned = re.sub(r"^(?:\[[^\]]+\]|[a-zA-Z0-9_\-\.\/]+:\s*)", "", text.strip())
    match = re.match(r"^([a-zA-Z]+)", cleaned.strip())
    return match.group(1) if match else ""


def detect_mood(text: str) -> str:
    """Detect whether commit/docstring text is 'imperative', 'descriptive', or 'neutral'."""
    first_word = get_first_word(text).lower()
    if not first_word:
        return "neutral"
    if first_word in _LOOKUP:
        return _LOOKUP[first_word][1]
    # Heuristics for -s / -es (third-person) or -ed (past)
    if first_word.endswith(("s", "es", "ed", "ing")):
        return "descriptive"
    return "imperative"


def detect_length_category(text: str, max_short_words: int = 7) -> str:
    """Classify message length as 'short' (<= max_short_words) or 'detailed' (> max_short_words)."""
    words = text.strip().split()
    return "short" if len(words) <= max_short_words else "detailed"


def convert_mood(text: str, target_mood: str) -> str:
    """Convert the leading verb to target_mood ('imperative' or 'descriptive')."""
    if target_mood not in ("imperative", "descriptive"):
        return text

    prefix_match = re.match(r"^(\s*(?:\[[^\]]+\]|[a-zA-Z0-9_\-\.\/]+:\s*))?([a-zA-Z]+)(.*)$", text, re.DOTALL)
    if not prefix_match:
        return text

    prefix = prefix_match.group(1) or ""
    first_word = prefix_match.group(2)
    rest = prefix_match.group(3)

    lower = first_word.lower()
    if lower in _LOOKUP:
        base, _ = _LOOKUP[lower]
        target_form = _VERB_MAP[base][target_mood]
        # Preserve original capitalization style (Title Case)
        return f"{prefix}{target_form}{rest}"

    # Fallback heuristic for common suffixes
    if target_mood == "imperative":
        if lower.endswith("ed") and len(lower) > 4:
            new_verb = first_word[:-2].capitalize()
            return f"{prefix}{new_verb}{rest}"
        if lower.endswith("s") and len(lower) > 3:
            new_verb = first_word[:-1].capitalize()
            return f"{prefix}{new_verb}{rest}"
    elif target_mood == "descriptive":
        if not lower.endswith("s") and len(lower) > 2:
            new_verb = f"{first_word}s".capitalize()
            return f"{prefix}{new_verb}{rest}"

    return text


def truncate_to_short(text: str, max_words: int = 7) -> str:
    """Condense message to short length by truncating additional clauses or details."""
    # If already short, return as is
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()

    # Split on punctuation (comma, semicolon, colon, dash, parenthesis)
    clauses = re.split(r"[,;:\-\(\)]", text)
    first_clause = clauses[0].strip()
    clause_words = first_clause.split()
    if 2 <= len(clause_words) <= max_words:
        return first_clause

    # Otherwise take up to max_words
    return " ".join(words[:max_words]).rstrip(",.:;")


def format_style(
    text: str,
    mood: str | None = None,
    length: str | None = None,
) -> str:
    """Format text to target mood and length."""
    result = text.strip()
    if mood:
        result = convert_mood(result, mood)
    if length == "short":
        result = truncate_to_short(result)
    return result


def analyze_styles(texts: Sequence[str]) -> dict[str, object]:
    """Analyze style distribution across a dataset or predictions."""
    total = len(texts)
    if total == 0:
        return {"total": 0}

    mood_counts = {"imperative": 0, "descriptive": 0, "neutral": 0}
    length_counts = {"short": 0, "detailed": 0}
    word_lengths: list[int] = []

    for text in texts:
        mood = detect_mood(text)
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

        length_cat = detect_length_category(text)
        length_counts[length_cat] = length_counts.get(length_cat, 0) + 1

        word_lengths.append(len(text.split()))

    avg_words = sum(word_lengths) / total if total else 0.0

    return {
        "total_messages": total,
        "mood_distribution": {
            k: {"count": v, "percentage": round(v / total * 100, 1)}
            for k, v in mood_counts.items()
        },
        "length_distribution": {
            k: {"count": v, "percentage": round(v / total * 100, 1)}
            for k, v in length_counts.items()
        },
        "average_word_count": round(avg_words, 2),
    }
