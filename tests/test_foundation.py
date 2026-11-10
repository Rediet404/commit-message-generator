from pathlib import Path

import commitgen
from commitgen.config import GENERATION, PATHS
from commitgen.logging_config import configure_logging


def test_package_metadata_is_available() -> None:
    assert commitgen.__version__ == "0.1.0"


def test_project_paths_point_to_project_root() -> None:
    assert PATHS.root.is_dir()
    assert PATHS.data == Path(PATHS.root, "data")


def test_generation_defaults_are_defined() -> None:
    assert GENERATION.model_name == "Salesforce/codet5-small"
    assert GENERATION.max_input_tokens > GENERATION.max_output_tokens


def test_logging_configuration_is_callable() -> None:
    configure_logging()
