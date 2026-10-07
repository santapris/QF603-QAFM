"""Phase 6.4 backtest engine: monthly short delta-hedged SPX straddle + long put wing, sized by stress budget.

Per cycle (one per month, contracts from ``strategy_contracts.parquet``):
* scale = ``risk_budget_stress_loss`` × m_t / stress_loss_per_unit, so a unit's P&L in index points times
  ``scale`` is a return on capital (stress: ``stress_scenario``, hedged book repriced with Black-76);
* entry at the entry-day close: option legs filled at mid ∓ θ·half-spread plus fees; hedge set to offset the
  book's delta (OptionMetrics deltas);
* daily: option legs marked at mid (a missing quote carries the previous mid, counted), hedge P&L
  = H_{d−1}·ΔS, re-hedge when |target − H| > ``hedge_band_delta`` (0 = every day);
* settlement: intrinsic value at the SPX open (AM) or close (PM); hedge closed at the same print;
* capital earns the T-bill rate every day (``rf`` month t → t+1 spread evenly over the month's trading days);
  between settlement and the next entry the strategy holds only T-bills.
m_t = 0 means no position that month. Returns are simple daily returns on start-of-cycle capital, compounded
within calendar months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.costs import CostModel
from src.strategy.instruments import settlement_value, stress_loss
from src.utils.dates import trading_days
from src.utils.io import load_config, project_path

LEGS = ("call", "put", "wing")
SIGN = {"call": -1.0, "put": -1.0, "wing": 1.0}     # short straddle, long wing


def load_inputs(cfg):
    proc = lambda n: pd.read_parquet(project_path("data", "processed", n))  # noqa: E731
    spx = pd.read_parquet(project_path("data", "raw", "optionmetrics", "spx_index_daily.parquet")).set_index("date")
    spx = spx[["open", "close"]].astype(float)
    fred = proc("fred_monthly.parquet")
    daily = proc("strategy_options_daily.parquet")
    num = ["best_bid", "best_offer", "impl_volatility", "delta", "gamma", "vega"]
    daily[num] = daily[num].astype(float)
    contracts = proc("strategy_contracts.parquet")
    for col in contracts.columns:
        if contracts[col].dtype.kind in "iufc" or str(contracts[col].dtype) in ("Float64", "Int64", "Int8"):
            contracts[col] = contracts[col].astype(float)
    return contracts, daily, spx, fred["rf"]


def daily_rf(rf_monthly: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Spread each month's T-bill return (set at month-end t for t → t+1) over the next month's trading days."""
    per = days.to_period("M")
    rf = pd.Series(np.nan, index=days)
    for m, idx in pd.Series(days, index=days).groupby(per):
        prev = rf_monthly.loc[:(m.start_time - pd.Timedelta(days=1))].dropna()
        if len(prev):
            rf.loc[idx.index] = (1 + prev.iloc[-1]) ** (1 / len(idx)) - 1
    return rf.fillna(0.0)


def run_cycle(c: pd.Series, quotes: dict, spx: pd.DataFrame, days: pd.DatetimeIndex, costs: CostModel,
              scale: float, band: float) -> pd.DataFrame:
    """Daily P&L (index points per unit, and cost components) of one cycle from entry to settlement."""
    life = days[(days >= c["entry"]) & (days <= c["settle"])]
    mids = {leg: c[f"{leg}_mid"] for leg in LEGS}
    deltas = {leg: c[f"{leg}_delta"] for leg in LEGS}
    book = lambda m: sum(SIGN[l] * m[l] for l in LEGS)  # noqa: E731,E741
    v_prev = book(mids)
    half = [(c[f"{l}_best_offer"] - c[f"{l}_best_bid"]) / 2 for l in LEGS]
    S_prev = spx.loc[c["entry"], "close"]
    H = -book(deltas)
    rows = [dict(date=c["entry"], opt=0.0, hedge=0.0, spread_fee=-costs.option_entry_cost(half),
                 hedge_cost=-costs.hedge_cost(H, S_prev), stale=0)]
    for d in life[1:]:
        last = d == life[-1] and d == c["settle"]
        stale = 0
        if last:
            S = spx.loc[d, "open" if c["am_settlement"] == 1 else "close"]
            v = settlement_value(c, S)
        else:
            S = spx.loc[d, "close"]
            for l in LEGS:
                q = quotes.get((int(c[f"{l}_optionid"]), d))
                if q is None or not np.isfinite(q[0]):
                    stale += 1
                    continue
                mids[l] = q[0]
                if np.isfinite(q[1]):
                    deltas[l] = q[1]
            v = book(mids)
        row = dict(date=d, opt=v - v_prev, hedge=H * (S - S_prev), spread_fee=0.0, stale=stale)
        if last:
            row["hedge_cost"] = -costs.hedge_cost(H, S)
        else:
            target = -book(deltas)
            trade = target - H if abs(target - H) > band else 0.0
            H += trade
            row["hedge_cost"] = -costs.hedge_cost(trade, S)
        rows.append(row)
        v_prev, S_prev = v, S
    out = pd.DataFrame(rows).set_index("date")
    out["pnl_pts"] = out[["opt", "hedge", "spread_fee", "hedge_cost"]].sum(axis=1)
    out["ret"] = scale * out["pnl_pts"]
    return out


def backtest(multipliers: pd.Series, contracts: pd.DataFrame, daily: pd.DataFrame, spx: pd.DataFrame,
             rf_monthly: pd.Series, cfg: dict, costs: CostModel, start=None, end=None,
             band: float | None = None) -> dict:
    """Run a strategy given m_t per signal date. Returns daily and monthly returns plus a cycle table."""
    s = cfg["strategy"]
    band = s["hedge_band_delta"] or 0.0 if band is None else band
    quotes = {(r.optionid, r.date): ((r.best_bid + r.best_offer) / 2 if r.best_offer > 0 else np.nan, r.delta)
              for r in daily.itertuples()}
    c_all = contracts.set_index("signal_date")
    lo = pd.Timestamp(start) if start is not None else c_all["entry"].min()
    hi = pd.Timestamp(end) if end is not None else c_all["settle"].max()
    days = trading_days(lo, hi)
    days = days[days.isin(spx.index)]
    ret = pd.Series(0.0, index=days)
    parts, cycles = [], []
    for sig, c in c_all.iterrows():
        m = float(multipliers.get(sig, 0.0))
        if not (lo <= c["entry"] and c["settle"] <= hi):
            continue
        S0 = spx.loc[c["entry"], "close"]
        L = stress_loss(c.to_dict(), S0, s["stress_scenario"]["spx_move"], s["stress_scenario"]["iv_mult"])
        scale = s["risk_budget_stress_loss"] * m / L if m > 0 and L > 0 else 0.0
        info = dict(signal_date=sig, entry=c["entry"], settle=c["settle"], m=m, stress_loss_pts=L, scale=scale)
        if scale > 0:
            cyc = run_cycle(c, quotes, spx, days, costs, scale, band)
            ret.loc[cyc.index] += cyc["ret"]
            parts.append(cyc.assign(signal_date=sig))
            info.update(pnl_pts=cyc["pnl_pts"].sum(), cost_pts=cyc[["spread_fee", "hedge_cost"]].sum().sum(),
                        hedge_cost_pts=cyc["hedge_cost"].sum(), cycle_ret=cyc["ret"].sum(),
                        stale_quotes=int(cyc["stale"].sum()))
        cycles.append(info)
    rf = daily_rf(rf_monthly, days)
    total = ret + rf
    monthly = (1 + total).groupby(days.to_period("M")).prod() - 1
    rf_m = (1 + rf).groupby(days.to_period("M")).prod() - 1
    detail = pd.concat(parts) if parts else pd.DataFrame()
    return dict(daily=total, daily_excess=ret, monthly=monthly, monthly_excess=monthly - rf_m, rf_monthly=rf_m,
                cycles=pd.DataFrame(cycles), detail=detail, costs=costs)


def variance_swap_pnl(vrp_expost: pd.Series, notional: float = 1.0) -> pd.Series:
    """Paper short variance-swap payoff (frictionless benchmark): notional × (IV − realised variance)."""
    return notional * vrp_expost


def main(cfg=None, force=False):
    from src.strategy.run import main as run_main
    return run_main(cfg, force)
