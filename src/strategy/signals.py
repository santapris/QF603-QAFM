"""Phase 6.4 signals: expected payoff, its risk, the cost hurdle and the size multiplier m_t.

* μ̂_t: expected short-variance payoff (annualised variance units). S1 uses μ̂_t = vrp_exante_t; S2 uses the
  Stage 4 k = 0 bridge forecast (selected model, validation only until Phase 7).
* σ̂_t: expanding-window regression of the squared payoff surprise (vrp_expost − μ̂)² on [1, VIX², VVIX],
  fitted on rows whose payoff is observed by t (exact dates); σ̂ = √max(fit, floor).
* Expected P&L of the traded unit ≈ straddle vega / (2·IV_atm) × μ̂_t (index points; vega per 1.00 vol),
  compared with the expected round-trip cost = entry spread·θ + fees + expected hedge cost (mean realised
  hedge cost of earlier cycles, known at t). Trade only if expected P&L > ``cost_hurdle`` × expected cost.
* Tier: z_t = μ̂_t / σ̂_t against the terciles of past z (≥ ``MIN_HISTORY`` values, expanding) →
  m ∈ ``multiplier_tiers`` = [0, 0.5, 1, 1.5] (0 if the hurdle fails or μ̂ ≤ 0; 1 before enough history).
* Optional cap: m ≤ 1 when VIX_t is above its training-window 80th percentile (set if Stage 3 finds unstable
  slopes in high-VIX months).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.costs import CostModel
from src.utils.holdout import training_rows

MIN_HISTORY = 24
SIGMA_FLOOR = 1e-6


def sigma_hat(panel: pd.DataFrame, mu: pd.Series, origins, cfg: dict, min_obs: int = 36) -> pd.Series:
    """Conditional s.d. of the payoff surprise at each origin (expanding, observed payoffs only)."""
    d = pd.DataFrame({"y": (panel["vrp_expost"] - mu) ** 2, "vix2": (panel["vix"] / 100) ** 2,
                      "vvix": panel["vvix"]}, index=panel.index)
    obs = panel["obs_date_vrp_expost"]
    out = {}
    for t in pd.DatetimeIndex(origins):
        tr = training_rows(d, t, "vrp_expost", 0, cfg=cfg, obs_dates=obs).dropna()
        x0 = d.loc[t, ["vix2", "vvix"]]
        if len(tr) < min_obs or x0.isna().any():
            out[t] = np.nan
            continue
        A = np.column_stack([np.ones(len(tr)), tr[["vix2", "vvix"]].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(float), rcond=None)
        out[t] = float(np.sqrt(max(np.r_[1.0, x0.to_numpy(float)] @ beta, SIGMA_FLOOR ** 2)))
    return pd.Series(out, name="sigma_hat")


def expected_pnl_pts(contracts: pd.DataFrame, mu: pd.Series) -> pd.Series:
    c = contracts.set_index("signal_date")
    vega = c["call_vega"] + c["put_vega"]
    iv_atm = (c["call_impl_volatility"] + c["put_impl_volatility"]) / 2
    return (vega / (2 * iv_atm) * mu.reindex(c.index)).rename("exp_pnl_pts")


def expected_cost_pts(contracts: pd.DataFrame, costs: CostModel, past_hedge_cost: pd.Series | None) -> pd.Series:
    """Entry spread + fees (known at entry) + mean realised hedge cost of cycles settled before each signal."""
    c = contracts.set_index("signal_date")
    half = sum((c[f"{l}_best_offer"] - c[f"{l}_best_bid"]) / 2 for l in ("call", "put", "wing"))
    entry = costs.theta * half + 3 * costs.fee_per_contract / 100
    hedge = pd.Series(0.0, index=c.index)
    if past_hedge_cost is not None and len(past_hedge_cost):
        hc = past_hedge_cost.dropna()                      # indexed by settlement date, positive = cost
        hedge = pd.Series([hc.loc[:t - pd.Timedelta(days=1)].mean() if (hc.index < t).any() else 0.0
                           for t in c.index], index=c.index)
    return (entry + hedge).rename("exp_cost_pts")


def tiers(z: pd.Series, cfg: dict) -> pd.Series:
    """Tier 1..3 from expanding terciles of past z (tier 2 = middle before MIN_HISTORY observations)."""
    out = {}
    hist = []
    for t, v in z.items():
        if not np.isfinite(v):
            out[t] = np.nan
            continue
        if len(hist) >= MIN_HISTORY:
            lo, hi = np.quantile(hist, [1 / 3, 2 / 3])
            out[t] = 1 if v <= lo else (3 if v > hi else 2)
        else:
            out[t] = 2
        hist.append(v)
    return pd.Series(out)


def multipliers(mu: pd.Series, sigma: pd.Series, exp_pnl: pd.Series, exp_cost: pd.Series, cfg: dict,
                vix: pd.Series | None = None, cap_high_vol: bool = False) -> pd.DataFrame:
    s = cfg["strategy"]
    grid = s["multiplier_tiers"]                           # [0, low, mid, high]
    df = pd.DataFrame({"mu": mu, "sigma": sigma, "exp_pnl": exp_pnl, "exp_cost": exp_cost}).dropna(subset=["mu"])
    df["z"] = df["mu"] / df["sigma"]
    df["tier"] = tiers(df["z"], cfg)
    df["hurdle_ok"] = (df["mu"] > 0) & (df["exp_pnl"] > s["cost_hurdle"] * df["exp_cost"])
    df["m"] = [grid[int(t)] if ok and np.isfinite(t) else 0.0 for t, ok in zip(df["tier"], df["hurdle_ok"])]
    if cap_high_vol and vix is not None:
        thr = np.nanpercentile(vix.loc[:cfg["oos"]["train_end"]], cfg["regimes"]["high_vol_vix_percentile"])
        high = vix.reindex(df.index) > thr
        df.loc[high, "m"] = df.loc[high, "m"].clip(upper=1.0)
        df["high_vol_cap"] = high
    return df
