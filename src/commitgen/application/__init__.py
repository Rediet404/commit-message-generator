"""Git-facing application components."""

from __future__ import annotations


def main(*args, **kwargs):
    from commitgen.application.cli import main as _main
    return _main(*args, **kwargs)


__all__ = ["main"]
