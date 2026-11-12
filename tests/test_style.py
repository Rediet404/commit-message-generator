from commitgen.preprocessing.style import (
    analyze_styles,
    convert_mood,
    detect_length_category,
    detect_mood,
    format_style,
    truncate_to_short,
)


def test_detect_mood_identifies_imperative_and_descriptive() -> None:
    assert detect_mood("Add user authentication") == "imperative"
    assert detect_mood("Fix null pointer exception") == "imperative"
    assert detect_mood("Adds user authentication") == "descriptive"
    assert detect_mood("Fixed null pointer exception") == "descriptive"
    assert detect_mood("Updating configuration settings") == "descriptive"
    assert detect_mood("[docs] Update README") == "imperative"
    assert detect_mood("docs: Adds examples") == "descriptive"


def test_detect_length_category() -> None:
    assert detect_length_category("Add auth") == "short"
    assert detect_length_category("Update 3 files across the repo") == "short"
    assert detect_length_category("This is a much longer detailed commit subject explaining the rationale behind the change") == "detailed"


def test_convert_mood_transforms_verbs() -> None:
    assert convert_mood("Adds support for python 3.13", "imperative") == "Add support for python 3.13"
    assert convert_mood("Fixed broken tests in auth module", "imperative") == "Fix broken tests in auth module"
    assert convert_mood("Add user authentication", "descriptive") == "Adds user authentication"
    assert convert_mood("Fix bug in parser", "descriptive") == "Fixes bug in parser"
    # Preserves prefix
    assert convert_mood("docs: Adds details", "imperative") == "docs: Add details"


def test_truncate_to_short() -> None:
    text = "Add authentication support, which handles OAuth2 and JWT tokens"
    shortened = truncate_to_short(text, max_words=7)
    assert len(shortened.split()) <= 7
    assert shortened.startswith("Add authentication support")


def test_format_style_applies_both() -> None:
    raw = "Fixes bug in authentication, avoiding memory leak during shutdown"
    formatted = format_style(raw, mood="imperative", length="short")
    assert formatted.startswith("Fix")
    assert len(formatted.split()) <= 7


def test_analyze_styles_computes_distributions() -> None:
    messages = [
        "Add auth",
        "Fixes bug in connection pool",
        "Updated documentation for client usage, configuration options, and authentication setup",
    ]
    report = analyze_styles(messages)
    assert report["total_messages"] == 3
    assert report["mood_distribution"]["imperative"]["count"] == 1
    assert report["mood_distribution"]["descriptive"]["count"] == 2
    assert report["length_distribution"]["short"]["count"] == 2
    assert report["length_distribution"]["detailed"]["count"] == 1
