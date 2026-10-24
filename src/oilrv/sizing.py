"""Position sizing (plan: Trading rules, sizing and risk limits).

Two streams, as pre-registered:
- research stream (gated): units = R / (sigma_hat * M), unrounded, fixed at entry
- integer book (reported): N = min(floor(R / (sigma_hat * M)), floor(L / ((z_stop - |z|) * sd_w * M))),
  capped at max_units; halved after a book drawdown of `drawdown_halve`, no trading after `drawdown_stop`

R is the daily risk budget, L the maximum loss per trade, sigma_hat the one-day volatility forecast
of the spread ($/bbl), sd_w the window sd of the (deseasonalised) spread level, and M the dollars per
$1/bbl move per unit.
"""
from __future__ import annotations

import numpy as np

from .config import load_config


def vol_limit(risk_usd: float, sigma: float, usd_per_bbl: float) -> float:
    """Units such that a one-sigma day costs `risk_usd` (before rounding)."""
    return risk_usd / (sigma * usd_per_bbl) if sigma > 0 else 0.0


def stop_limit(loss_usd: float, abs_z: float, stop_z: float, sd_w: float, usd_per_bbl: float) -> float:
    """Units such that a move from |z| to the stop costs `loss_usd` (before rounding)."""
    dist = (stop_z - abs_z) * sd_w
    return loss_usd / (dist * usd_per_bbl) if dist > 0 else 0.0


def research_units(sigma: float, usd_per_bbl: float, cfg: dict | None = None) -> float:
    return vol_limit((cfg or load_config())["sizing"]["risk_per_day_usd"], sigma, usd_per_bbl)


def integer_units(sigma: float, abs_z: float, sd_w: float, usd_per_bbl: float, size_mult: float = 1.0,
                  cfg: dict | None = None) -> dict:
    """The plan's integer size: the smaller of the two limits, floored, capped, then scaled by the
    book's drawdown multiplier (1, 0.5 or 0) and floored again."""
    cfg = cfg or load_config()
    s = cfg["sizing"]
    v = vol_limit(s["risk_per_day_usd"], sigma, usd_per_bbl)
    st = stop_limit(s["max_loss_per_trade_usd"], abs_z, cfg["strategy"]["stop_z"], sd_w, usd_per_bbl)
    n = min(int(np.floor(v)), int(np.floor(st)), int(s["max_units"]))
    n = int(np.floor(max(n, 0) * size_mult))
    return {"vol_limit": v, "stop_limit": st, "units": n}


def drawdown_multiplier(drawdown: float, cfg: dict | None = None) -> float:
    """1 below the halve threshold, 0.5 between halve and stop, 0 at or beyond the stop."""
    s = (cfg or load_config())["sizing"]
    if drawdown >= s["drawdown_stop"]:
        return 0.0
    if drawdown >= s["drawdown_halve"]:
        return 0.5
    return 1.0
