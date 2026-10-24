"""Load config.yaml and resolve project paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    """Return the parsed config.yaml (cached)."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a directory listed under `data:` in the config, created if missing."""
    p = ROOT / load_config()["data"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
