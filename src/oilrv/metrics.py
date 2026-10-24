"""Performance metrics and the pre-registered gate verdicts (PREREGISTRATION.md section 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def sharpe(pnl: pd.Series, ann: int = 252) -> float:
    """Annualised Sharpe of daily P&L, counting flat days as zero P&L."""
    sd = pnl.std()
    return float(pnl.mean() / sd * np.sqrt(ann)) if sd and np.isfinite(sd) and sd > 0 else np.nan


def max_drawdown(pnl: pd.Series) -> float:
    cum = pnl.cumsum()
    return float((cum - cum.cummax()).min()) if len(cum) else np.nan


def summarize(daily: pd.DataFrame, trades: pd.DataFrame, col: str = "pnl_usd", ann: int = 252) -> dict:
    """Headline metrics for one run, on P&L column `col` ('pnl_usd' research stream or 'pnl_bbl' per unit)."""
    pnl = daily[col]
    net_col = {"pnl_usd": "net_usd", "pnl_bbl": "net_bbl"}[col]
    closed = trades[trades["exit_reason"] != "open_at_end"] if len(trades) else trades
    years = len(pnl) / ann
    sr = sharpe(pnl, ann)
    top5 = closed[net_col].nlargest(5).sum() if len(closed) else np.nan
    total_closed = closed[net_col].sum() if len(closed) else np.nan
    return {
        "sharpe": sr,
        "sharpe_t": sr * np.sqrt(years) if np.isfinite(sr) else np.nan,
        "total": float(pnl.sum()),
        "ann_mean": float(pnl.mean() * ann),
        "max_dd": max_drawdown(pnl),
        "n_trades": int(len(closed)),
        "open_at_end": int(len(trades) - len(closed)),
        "win_rate": float((closed[net_col] > 0).mean()) if len(closed) else np.nan,
        "tp_rate": float((closed["exit_reason"] == "take_profit").mean()) if len(closed) else np.nan,
        "avg_hold": float(closed["hold_days"].mean()) if len(closed) else np.nan,
        "avg_hold_over_half_life": float((closed["hold_days"] / closed["half_life"]).mean()) if len(closed) else np.nan,
        "avg_trade": float(closed[net_col].mean()) if len(closed) else np.nan,
        "worst_trade": float(closed[net_col].min()) if len(closed) else np.nan,
        "top5_share_of_pnl": float(top5 / total_closed) if len(closed) and total_closed > 0 else np.nan,
        "time_in_market": float((daily["side"] != 0).mean()),
        "years": years,
    }


def window_pnl(daily: pd.DataFrame, col: str, start: str, end: str | None) -> float:
    return float(daily.loc[start:end, col].sum())


def subperiods(daily: pd.DataFrame, col: str, periods: list, ann: int = 252) -> pd.DataFrame:
    rows = {}
    for start, end in periods:
        d = daily.loc[start:end, col]
        lab = f"{start[:4]}–{(end or str(daily.index[-1].date()))[:4]}"
        rows[lab] = {"net": float(d.sum()), "sharpe": sharpe(d, ann), "days": len(d)}
    return pd.DataFrame(rows).T


def exit_reasons(trades: pd.DataFrame) -> pd.Series:
    return trades["exit_reason"].value_counts() if len(trades) else pd.Series(dtype=int)


def gate_verdict(base_summary: dict, sub: pd.DataFrame, grid_sharpes: dict, stress: dict,
                 cfg: dict | None = None) -> dict:
    """G1-G5 from config['gates'] for one spread's gated configuration.

    `stress` maps each G5 window name to {'filtered': net, 'unfiltered': net} (research stream USD).
    """
    g = (cfg or load_config())["gates"]
    g1 = bool(np.isfinite(base_summary["sharpe"]) and base_summary["sharpe"] > g["min_net_sharpe"])
    g2 = bool(base_summary["n_trades"] >= g["min_round_trips"])
    g3 = bool((sub["net"] > 0).sum() >= g["min_positive_subperiods"])
    thr = g["entry_grid_stability"]["min_sharpe_all_grid_points"]
    g4 = bool(all(np.isfinite(s) and s > thr for s in grid_sharpes.values()))
    sf = g["stress_filter"]
    g5_parts = {w: bool(stress[w]["filtered"] >= stress[w]["unfiltered"] and stress[w]["filtered"] > -sf["max_loss_usd"])
                for w in sf["windows"]}
    g5 = all(g5_parts.values())
    return {
        "G1_sharpe": g1, "G1_value": base_summary["sharpe"],
        "G2_trades": g2, "G2_value": base_summary["n_trades"],
        "G3_subperiods": g3, "G3_value": int((sub["net"] > 0).sum()),
        "G4_grid": g4, "G4_value": grid_sharpes,
        "G5_stress": g5, "G5_value": stress,
        "PASS": g1 and g2 and g3 and g4 and g5,
    }
