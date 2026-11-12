from pathlib import Path

from commitgen.baseline.generator import (
    extract_first_comment,
    extract_function_name,
    generate_baseline,
    generate_code_baseline,
)
from commitgen.data.docstrings import (
    STARTER_PAIRS,
    extract_pairs_from_code,
    load_code_docstring_dataset,
    save_code_docstring_dataset,
)


SAMPLE_CODE_WITH_DOCSTRING = '''
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle.
    
    Detailed explanation here.
    """
    return 3.14159 * radius * radius
'''

SAMPLE_CODE_WITH_COMMENT = '''
# Validate username format and length
def validate_username(name: str) -> bool:
    return len(name) >= 3
'''

SAMPLE_CODE_NO_COMMENT = '''
def normalize_matrix(matrix):
    return [[val / 2.0 for val in row] for row in matrix]
'''


def test_extract_first_comment() -> None:
    assert extract_first_comment(SAMPLE_CODE_WITH_DOCSTRING) == "Calculate the area of a circle."
    assert extract_first_comment(SAMPLE_CODE_WITH_COMMENT) == "Validate username format and length"
    assert extract_first_comment(SAMPLE_CODE_NO_COMMENT) is None


def test_extract_function_name() -> None:
    assert extract_function_name(SAMPLE_CODE_NO_COMMENT) == "normalize matrix"


def test_generate_code_baseline() -> None:
    assert generate_code_baseline(SAMPLE_CODE_WITH_DOCSTRING) == "Calculate the area of a circle."
    assert generate_code_baseline(SAMPLE_CODE_WITH_COMMENT) == "Validate username format and length"
    assert generate_code_baseline(SAMPLE_CODE_NO_COMMENT) == "Execute normalize matrix"


def test_generate_baseline_auto_detects_code() -> None:
    # Auto-detection routes code to code baseline
    res = generate_baseline(SAMPLE_CODE_WITH_COMMENT)
    assert res == "Validate username format and length"


def test_extract_pairs_from_code_removes_docstring() -> None:
    pairs = extract_pairs_from_code(SAMPLE_CODE_WITH_DOCSTRING)
    assert len(pairs) == 1
    assert pairs[0]["function_name"] == "calculate_area"
    assert pairs[0]["docstring"] == "Calculate the area of a circle."
    # Docstring must NOT leak into the code input
    assert "Calculate the area of a circle" not in pairs[0]["code"]
    assert "return 3.14159 * radius * radius" in pairs[0]["code"]


def test_save_and_load_code_docstrings(tmp_path: Path) -> None:
    out_file = tmp_path / "pairs.jsonl"
    save_code_docstring_dataset(STARTER_PAIRS, out_file)
    loaded = load_code_docstring_dataset(out_file)
    assert len(loaded) == len(STARTER_PAIRS)
    assert loaded[0]["function_name"] == "add_vectors"
