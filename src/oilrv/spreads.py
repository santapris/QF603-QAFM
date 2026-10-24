"""Spread definitions (from config.yaml `spreads`) and construction from leg prices.

Leg prices come in native units: cl, brent in $/bbl; rb, ho in $/gal. Every spread is built in
$/bbl, in levels (never logs: WTI settled negative on 20 Apr 2020).

spread = sum(legs[k] * price_bbl[k]) / divisor

so one unit (the contract counts in `legs`, 1,000 bbl each) moves `usd_per_1bbl_per_unit` dollars
per $1/bbl move in the spread.
"""
from __future__ import annotations

import pandas as pd

from .config import load_config

GALLON_LEGS = ("rb", "ho")


def traded_spreads(cfg: dict | None = None) -> dict:
    """{name: spec} for the spreads with traded: true."""
    cfg = cfg or load_config()
    return {k: v for k, v in cfg["spreads"].items() if v.get("traded")}


def to_bbl(px: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Convert the gallon-quoted legs to $/bbl; leave the others unchanged."""
    gal = (cfg or load_config())["data"]["gal_per_bbl"]
    out = px.copy()
    for leg in GALLON_LEGS:
        if leg in out:
            out[leg] = out[leg] * gal
    return out


def spread_value(px_bbl: pd.DataFrame, spec: dict, weights: dict | None = None) -> pd.Series:
    """One spread from $/bbl legs. `weights` overrides spec['legs'] (e.g. Johansen weights)."""
    legs = weights or spec["legs"]
    s = sum(w * px_bbl[leg] for leg, w in legs.items())
    return s / spec["divisor"]


def build_spreads(px: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """The traded spreads ($/bbl) plus each leg in $/bbl (columns `<leg>_bbl`)."""
    cfg = cfg or load_config()
    bbl = to_bbl(px, cfg)
    out = pd.DataFrame(index=px.index)
    for name, spec in traded_spreads(cfg).items():
        out[name] = spread_value(bbl, spec)
    for leg in px.columns:
        out[f"{leg}_bbl"] = bbl[leg]
    return out


def usd_per_bbl(name: str, cfg: dict | None = None) -> float:
    """Dollars per $1/bbl move per unit of the spread (M in the sizing formulas)."""
    return float((cfg or load_config())["spreads"][name]["usd_per_1bbl_per_unit"])
