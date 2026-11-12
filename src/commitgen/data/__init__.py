"""Data collection, provenance, and dataset loading components."""

from commitgen.data.collector import collect_repository, split_by_repository
from commitgen.data.docstrings import (
    STARTER_PAIRS,
    extract_pairs_from_code,
    extract_pairs_from_directory,
    load_code_docstring_dataset,
    save_code_docstring_dataset,
)

__all__ = [
    "STARTER_PAIRS",
    "collect_repository",
    "extract_pairs_from_code",
    "extract_pairs_from_directory",
    "load_code_docstring_dataset",
    "save_code_docstring_dataset",
    "split_by_repository",
]
