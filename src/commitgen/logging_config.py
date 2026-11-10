"""Logging setup shared by command-line tools and experiments."""

import logging


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging without changing existing handlers."""

    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
