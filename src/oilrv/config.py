"""Load config.yaml, resolve project paths and hold the data source of the current run."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("futures", "spot")
_SOURCE: list[str] = []


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    """Return the parsed config.yaml (cached)."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def default_source() -> str:
    """Futures (binding) once the Bloomberg export is in place, else spot (proxy)."""
    f = load_config()["futures"].get("export_file")
    return "futures" if f and (ROOT / f).exists() else "spot"


def set_source(name: str | None) -> str:
    """Select the data source for this run (None -> default_source()). Returns it."""
    name = name or default_source()
    if name not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {name!r}")
    _SOURCE[:] = [name]
    return name


def get_source() -> str:
    return _SOURCE[0] if _SOURCE else default_source()


def output_root() -> Path:
    """output/ for the binding futures run; output/spot_proxy/ for the spot proxy."""
    return ROOT / "output" / ("spot_proxy" if get_source() == "spot" else "")


def path(key: str) -> Path:
    """Absolute path for a directory listed under `data:` in the config, created if missing.

    `derived_dir` depends on the source: model outputs of the spot proxy go to its `spot_proxy/` subfolder.
    """
    p = ROOT / load_config()["data"][key]
    if key == "derived_dir" and get_source() == "spot":
        p = p / "spot_proxy"
    p.mkdir(parents=True, exist_ok=True)
    return p
