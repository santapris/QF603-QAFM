"""Project paths, config loading and parquet I/O."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=None)
def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> dict:
    """Return a fresh copy of config.yaml (callers may mutate it safely)."""
    return copy.deepcopy(_load_config(str(path or CONFIG_PATH)))


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(project_path(str(path)) if not Path(path).is_absolute() else path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    out = project_path(str(path)) if not Path(path).is_absolute() else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out
