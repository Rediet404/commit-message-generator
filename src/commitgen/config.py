"""Centralized project configuration."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the project."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    processed_data: Path = PROJECT_ROOT / "data" / "processed"
    splits: Path = PROJECT_ROOT / "data" / "splits"
    results: Path = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class GenerationConfig:
    """Defaults shared by later generation experiments."""

    model_name: str = "Salesforce/codet5-small"
    max_input_tokens: int = 512
    max_output_tokens: int = 32
    seed: int = 42


PATHS = ProjectPaths()
GENERATION = GenerationConfig()
