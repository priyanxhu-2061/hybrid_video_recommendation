"""Config loading and path helpers."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """The recsys/ directory, regardless of where the script was launched from.

    This file sits at recsys/src/recsys/utils/io.py, so four parents up is
    recsys/. Config files reference paths relative to that, not to the caller's
    working directory - otherwise every command would only work from one folder.
    """
    return Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p