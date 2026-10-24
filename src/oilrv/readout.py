"""Part 7: current-regime readout and the plan's switch-on triggers (descriptive only).

The plan ("Strategy for the sector now") says to build and paper-trade the engine while the regime is
extreme, and to switch on normalisation trades only when the filter turns. Its triggers:

- diesel vs gasoline (HO - RB) and the 3-2-1 crack:
    P(revert) >= 0.6, the relevant crack's GARCH volatility back under its 80th percentile, and no
    Hormuz re-closure for 4 weeks. "Relevant crack" is the diesel crack (HO - CL) for HO - RB and the
    3-2-1 crack itself for the crack.
- Brent - WTI: futures spread at |z| >= 2 with the filter on; ignore spot while Dated Brent is dislocated.

Implementation choice: "under its 80th percentile" is measured against the full point-in-time history
of the volatility forecast since the first fit (an expanding percentile), i.e. "back to normal",
not against the trailing window.

Nothing here is a trade recommendation; it reports the state of pre-registered rules on spot data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .config import load_config
from .diagnostics import month_starts, window_before
from .model import fit_garch, garch_filter

HORMUZ_CLEAR_DAYS = 28
VOL_PCT_MAX = 0.80


def garch_path(x: pd.Series, window: int) -> pd.Series:
    """Daily point-in-time GARCH-t sigma forecast for the day after t, refit at each month start on
    the `window` daily changes strictly before it (same scheme as model.model_path)."""
    dx = x.diff()
    out = pd.Series(np.nan, index=x.index)
    starts = [s for s in month_starts(x.index) if s > window]
    ends = starts[1:] + [len(x)]
    for s, e in zip(starts, ends):
        g = fit_garch(window_before(dx, s, window - 1).to_numpy())
        out.iloc[s:e] = garch_filter(g, dx.iloc[s:e].to_numpy())
    return out.rename(f"{x.name}_sigma_next")


def expanding_pct(x: pd.Series) -> pd.Series:
    """Point-in-time percentile of each value within all values up to and including it."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    seen = []
    for i, a in enumerate(v):
        if np.isfinite(a):
            seen.append(a)
            out[i] = np.mean(np.asarray(seen) <= a)
    return pd.Series(out, index=x.index)


def triggers(name: str, p_hat: float, vol_pct: float, days_since_closure: float, z: float,
             source: str = "spot", cfg: dict | None = None) -> dict:
    """The plan's switch-on conditions for one spread. Returns each condition and whether all hold."""
    cfg = cfg or load_config()
    enter_p = cfg["strategy"]["regime_enter_p"]
    if name == "brent_wti":
        conds = {"futures data (not spot)": source == "futures",
                 f"|z| >= {cfg['strategy']['entry_z']}": bool(np.isfinite(z) and abs(z) >= cfg["strategy"]["entry_z"]),
                 f"P(revert) >= {enter_p}": bool(np.isfinite(p_hat) and p_hat >= enter_p)}
    else:
        conds = {f"P(revert) >= {enter_p}": bool(np.isfinite(p_hat) and p_hat >= enter_p),
                 f"vol percentile < {VOL_PCT_MAX:.0%}": bool(np.isfinite(vol_pct) and vol_pct < VOL_PCT_MAX),
                 f"no Hormuz closure for {HORMUZ_CLEAR_DAYS} days": bool(np.isfinite(days_since_closure)
                                                                      and days_since_closure >= HORMUZ_CLEAR_DAYS)}
    return {"conditions": conds, "all_met": all(conds.values())}


def wpsr_wednesdays(start: pd.Timestamp, n: int = 4) -> list[pd.Timestamp]:
    """The next `n` EIA Weekly Petroleum Status Report days (Wednesday; Thursday after a Mon-Wed US holiday)."""
    hol = USFederalHolidayCalendar().holidays(start, start + pd.Timedelta(days=7 * n + 14))
    out, d = [], start
    while len(out) < n:
        if d.weekday() == 2:
            week = pd.date_range(d - pd.Timedelta(days=2), d)
            out.append(d + pd.Timedelta(days=1) if week.isin(hol).any() else d)
        d += pd.Timedelta(days=1)
    return out


def watch_list(events: pd.DataFrame, today: pd.Timestamp, extra: list[dict], horizon_days: int = 130) -> pd.DataFrame:
    """Upcoming dated items: future rows of the events file, the next WPSR releases, and `extra`
    (items from the plan with their source). Sorted by date."""
    end = today + pd.Timedelta(days=horizon_days)
    ev = events[(events["date"] >= today) & (events["date"] <= end)]
    rows = [{"date": r.date, "item": f"{r.type}: {r.name}", "source": r.source} for r in ev.itertuples()]
    rows += [{"date": d, "item": "EIA Weekly Petroleum Status Report (refresh inventory features)",
              "source": "EIA schedule (Wednesday 10:30 ET; Thursday after a holiday)"} for d in wpsr_wednesdays(today)]
    rows += [r for r in extra if today <= pd.Timestamp(r["date"]) <= end]
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)
