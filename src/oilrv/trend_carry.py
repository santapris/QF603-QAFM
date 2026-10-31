"""Exploratory strategies on the Bloomberg futures: time-series trend and curve carry.

Registered 2026-10-28 (PREREGISTRATION_TREND_CARRY.md, config `exploratory_trend_carry`), after the
binding mean-reversion verdict and before any trend or carry result. They never change that verdict.

Instruments. Outrights hold their own front month (`futures.continuous`, k = 1, rolled 5 business days
before expiry); spreads are the registered same-delivery-month spreads from `futures_spreads.csv`.
Every series is the additive back-adjusted level in $/bbl, so its daily change is the held contracts'
change. All series sit on the common futures dates.

Signals at close t:
- trend: sign(x_t - x_{t-L}), L trading days
- carry: sign(M1_t - Mk_t), held month minus the k-th month (unadjusted); + = backwardation = long

Sizing and timing. target_t = signal_t * R / (sigma_t * M), unrounded; sigma_t is the point-in-time
GARCH(1,1)-t forecast for t+1 (`readout.garch_path`). Decided at close t, filled at close t+1, so the
position held over (s-1, s] is target_{s-2}. Costs: cost_usd_per_bbl * M per unit of position change,
and roll ticks per contract leg held through a roll.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import futures
from .config import load_config
from .metrics import max_drawdown, sharpe, subperiods, window_pnl
from .readout import garch_path
from .spreads import to_bbl

SLOPE_K = {"m1_m2": 2, "m1_m3": 3, "m1_m6": 6}


# --------------------------------------------------------------------------- data
def build_instruments(panels: dict[str, pd.DataFrame] | None = None, fut: pd.DataFrame | None = None,
                      cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """{instrument: frame} on the common futures dates with `x` (back-adjusted, $/bbl), `rolled`, `M`,
    `roll_usd` (roll cost per unit) and, for carry outrights, `m1`, `m2`, `m3`, `m6` (unadjusted)."""
    cfg = cfg or load_config()
    tc, st = cfg["exploratory_trend_carry"], cfg["strategy"]
    panels = panels or futures.load_panels(cfg)
    fut = fut if fut is not None else futures.load_futures()
    common = fut.index
    roots = futures.leg_roots(cfg)
    rb, ov = cfg["futures"]["roll_bdays_before_expiry"], futures.load_expiry_overrides()
    tick = {k: st["roll_cost_ticks_per_leg"] * v for k, v in st["tick_usd"].items()}
    out = {}
    for name in dict.fromkeys(tc["trend"]["instruments"] + tc["carry"]["instruments"]):
        if name in roots:                                        # outright
            one = lambda k: futures.continuous({name: panels[name]}, {name: roots[name]}, k, rb, ov)
            c = one(1)
            f = pd.DataFrame(index=common)
            f["x"] = to_bbl(c[[f"{name}_adj"]].rename(columns=lambda s: s[:-4]), cfg)[name].reindex(common)
            f["rolled"] = futures._to_common(c["rolled"], common)
            f["M"] = float(tc["usd_per_1bbl_per_unit_outright"])
            f["roll_usd"] = tick[name]
            if name in tc["carry"]["instruments"]:
                f["m1"] = c[name].reindex(common)
                for s, k in SLOPE_K.items():
                    f[f"m{k}"] = one(k)[name].reindex(common)
        else:                                                    # registered spread
            spec = cfg["spreads"][name]
            f = pd.DataFrame({"x": fut[name], "rolled": fut[f"rolled_{name}"].astype(bool)}, index=common)
            f["M"] = float(spec["usd_per_1bbl_per_unit"])
            f["roll_usd"] = sum(abs(w) * tick[leg] for leg, w in spec["legs"].items())
        out[name] = f
    return out


# --------------------------------------------------------------------------- signals and P&L
def trend_signal(x: pd.Series, lookback: int) -> pd.Series:
    return np.sign(x - x.shift(lookback))


def carry_signal(f: pd.DataFrame, slope: str) -> pd.Series:
    """+1 in backwardation (M1 > Mk), -1 in contango; NaN where either month is missing."""
    return np.sign(f["m1"] - f[f"m{SLOPE_K[slope]}"])


def simulate(f: pd.DataFrame, signal: pd.Series, sigma: pd.Series, start: pd.Timestamp,
             cfg: dict | None = None) -> pd.DataFrame:
    """Daily P&L of one instrument from `start`.

    Columns: pnl, gross, cost, roll, pos, target, and the inputs x, signal, sigma (as of close t)."""
    tc = (cfg or load_config())["exploratory_trend_carry"]
    M, R = float(f["M"].iloc[0]), float(tc["risk_per_day_usd"])
    target = (signal * R / (sigma * M)).where(f.index >= start).fillna(0.0)
    pos = target.shift(2).fillna(0.0)                   # decided at t, filled at t+1, held over (t+1, t+2]
    dx = f["x"].diff().fillna(0.0)
    gross = pos * M * dx
    cost = target.diff().abs().shift(1).fillna(0.0) * tc["cost_usd_per_bbl"] * M
    roll = pos.abs() * f["roll_usd"] * f["rolled"].astype(float)
    d = pd.DataFrame({"pnl": gross - cost - roll, "gross": gross, "cost": cost, "roll": roll,
                      "pos": pos, "target": target, "x": f["x"], "signal": signal, "sigma": sigma},
                     index=f.index)
    return d.loc[start:]


def position_changes(target: pd.Series) -> int:
    """Entries and sign flips of the target position."""
    s = np.sign(target)
    return int(((s != s.shift()) & (s != 0)).sum())


def sigmas(inst: dict[str, pd.DataFrame], cfg: dict | None = None) -> dict[str, pd.Series]:
    """Point-in-time GARCH sigma forecast per instrument (shared by every grid point)."""
    window = (cfg or load_config())["model"]["window"]
    return {n: garch_path(f["x"], window) for n, f in inst.items()}


def run_sleeves(inst: dict[str, pd.DataFrame], cfg: dict | None = None, lookback: int | None = None,
                slope: str | None = None, sig: dict[str, pd.Series] | None = None) -> dict:
    """Trend and carry sleeves and the combined book for one (lookback, slope) setting.

    Returns {'instruments': {sleeve|name: daily}, 'trend', 'carry', 'combined': daily P&L series,
    'changes': {sleeve: count}}."""
    cfg = cfg or load_config()
    tc = cfg["exploratory_trend_carry"]
    lookback = lookback or tc["trend"]["lookback_days"]
    slope = slope or tc["carry"]["slope"]
    sig = sig or sigmas(inst, cfg)
    start = max(s.first_valid_index() for s in sig.values())
    res, books, changes = {}, {}, {}
    for sleeve, names, make in [("trend", tc["trend"]["instruments"], lambda f: trend_signal(f["x"], lookback)),
                                ("carry", tc["carry"]["instruments"], lambda f: carry_signal(f, slope))]:
        ds = {n: simulate(inst[n], make(inst[n]), sig[n], start, cfg) for n in names}
        res.update({f"{sleeve}|{n}": d for n, d in ds.items()})
        books[sleeve] = pd.DataFrame({n: d["pnl"] for n, d in ds.items()}).mean(axis=1)
        changes[sleeve] = sum(position_changes(d["target"]) for d in ds.values())
    w = tc["sleeve_weights"]
    books["combined"] = w["trend"] * books["trend"] + w["carry"] * books["carry"]
    changes["combined"] = changes["trend"] + changes["carry"]
    return {"instruments": res, **books, "changes": changes, "start": start}


# --------------------------------------------------------------------------- evaluation
def summary(pnl: pd.Series) -> dict:
    years = len(pnl) / 252
    sr = sharpe(pnl)
    return {"sharpe": sr, "sharpe_t": sr * np.sqrt(years), "total": float(pnl.sum()),
            "ann_mean": float(pnl.mean() * 252), "ann_sd": float(pnl.std() * np.sqrt(252)),
            "max_dd": max_drawdown(pnl), "years": years}


def verdict(pnl: pd.Series, changes: int, grid: dict[str, float], cfg: dict | None = None) -> dict:
    """G1-G5 of PREREGISTRATION_TREND_CARRY.md section 3 for one book."""
    cfg = cfg or load_config()
    g = cfg["exploratory_trend_carry"]["gates"]
    sub = subperiods(pnl.to_frame("pnl"), "pnl", cfg["evaluation"]["sub_periods"])
    stress = {w: window_pnl(pnl.to_frame("pnl"), "pnl", *cfg["evaluation"]["stress_windows"][w])
              for w in g["stress_windows"]}
    sr = sharpe(pnl)
    out = {
        "G1_sharpe": bool(np.isfinite(sr) and sr > g["min_net_sharpe"]), "G1_value": sr,
        "G2_changes": bool(changes >= g["min_position_changes"]), "G2_value": changes,
        "G3_subperiods": bool((sub["net"] > 0).sum() >= g["min_positive_subperiods"]),
        "G3_value": int((sub["net"] > 0).sum()),
        "G4_grid": bool(all(np.isfinite(v) and v > g["grid_min_sharpe"] for v in grid.values())), "G4_value": grid,
        "G5_stress": bool(all(v > -g["stress_max_loss_usd"] for v in stress.values())), "G5_value": stress,
    }
    out["PASS"] = all(out[k] for k in ["G1_sharpe", "G2_changes", "G3_subperiods", "G4_grid", "G5_stress"])
    return out


# --------------------------------------------------------------------------- descriptive (not gated)
def daily_paths(res: dict) -> pd.DataFrame:
    """Long frame of every instrument's daily path: date, sleeve, instrument and the `simulate` columns."""
    parts = []
    for key, d in res["instruments"].items():
        sleeve, name = key.split("|")
        parts.append(d.assign(sleeve=sleeve, instrument=name).rename_axis("date").reset_index())
    cols = ["date", "sleeve", "instrument", "x", "signal", "sigma", "target", "pos",
            "gross", "cost", "roll", "pnl"]
    return pd.concat(parts, ignore_index=True)[cols]


def annual(books: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year net P&L and Sharpe per book (the final year is to date)."""
    g = books.groupby(books.index.year)
    out = pd.concat({"net": g.sum(), "sharpe": g.agg(sharpe)}, axis=1)
    out["days"] = g.size()
    out.index.name = "year"
    return out


def drawdowns(pnl: pd.Series, n: int = 5) -> pd.DataFrame:
    """The n deepest peak-to-trough drawdowns of cumulative P&L, with recovery dates (NaT if not yet)."""
    cum = pnl.cumsum()
    peak = cum.cummax()
    under = cum < peak
    rows, i, idx = [], 0, cum.index
    while i < len(cum):
        if not under.iloc[i]:
            i += 1
            continue
        j = i
        while j < len(cum) and under.iloc[j]:
            j += 1
        seg = cum.iloc[i:j]
        trough = seg.idxmin()
        rows.append({"peak": idx[i - 1] if i > 0 else idx[0], "trough": trough,
                     "recovered": idx[j] if j < len(cum) else pd.NaT,
                     "depth": float(seg.min() - peak.iloc[i]),
                     "days_to_trough": int(idx.get_loc(trough) - (i - 1)),
                     "days_underwater": int(j - (i - 1)) if j < len(cum) else np.nan})
        i = j
    out = pd.DataFrame(rows)
    return out.nsmallest(n, "depth").reset_index(drop=True) if len(out) else out


def rolling_sharpe(pnl: pd.Series, window: int = 252) -> pd.Series:
    return pnl.rolling(window).mean() / pnl.rolling(window).std() * np.sqrt(252)


def current_book(res: dict, cfg: dict | None = None) -> pd.DataFrame:
    """Latest close per instrument: signal, sigma, target units (decided today, filled next close),
    the units held, and the contracts per leg those units imply."""
    cfg = cfg or load_config()
    rows = {}
    for key, d in res["instruments"].items():
        sleeve, name = key.split("|")
        last = d.iloc[-1]
        legs = cfg["spreads"][name]["legs"] if name in cfg["spreads"] else {name: 1}
        rows[key] = {"date": d.index[-1].date(), "signal": last["signal"],
                     "direction": {1.0: "long", -1.0: "short"}.get(float(last["signal"]), "flat"),
                     "sigma_usd_bbl": last["sigma"], "target_units": last["target"], "held_units": last["pos"],
                     "contracts": ", ".join(f"{leg.upper()} {w * last['target']:+.2f}" for leg, w in legs.items())}
    return pd.DataFrame(rows).T
