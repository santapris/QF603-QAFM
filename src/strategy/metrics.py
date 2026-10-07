"""Phase 6.6 performance metrics (monthly returns; annualised where stated).

* ``summary``: mean excess return, vol, Sharpe, Sortino, skew, excess kurtosis, max drawdown, worst month,
  CVaR 95/99 (historical, of total monthly returns), hit rate.
* ``mppm``: Goetzmann, Ingersoll, Spiegel & Welch (2007) manipulation-proof performance measure, ρ = 3.
* ``ce_gain``: CRRA certainty-equivalent (annualised) of a strategy minus that of the benchmark.
* ``sharpe_diff_test``: Ledoit & Wolf (2008) HAC test of equal Sharpe ratios (delta method, Bartlett kernel).
* ``deflated_sharpe``: Bailey & López de Prado (2014) probability that the Sharpe ratio exceeds the maximum
  expected under N independent trials.
* ``attribution``: OLS of strategy excess returns on SPX excess returns and PUT-index excess returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.newey_west import ols_nw

EULER = 0.5772156649015329


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def cvar(r: pd.Series, level: float) -> float:
    q = r.quantile(1 - level)
    return float(r[r <= q].mean())


def summary(total: pd.Series, excess: pd.Series) -> dict:
    ex = excess.dropna()
    down = ex[ex < 0]
    return dict(months=len(ex), mean_excess_ann=12 * ex.mean(), vol_ann=np.sqrt(12) * ex.std(),
                sharpe=np.sqrt(12) * ex.mean() / ex.std() if ex.std() > 0 else np.nan,
                sortino=np.sqrt(12) * ex.mean() / np.sqrt((down ** 2).sum() / len(ex)) if len(down) else np.nan,
                skew=float(stats.skew(ex)), excess_kurt=float(stats.kurtosis(ex)),
                max_drawdown=max_drawdown(total.dropna()), worst_month=float(total.min()),
                cvar95=cvar(total.dropna(), 0.95), cvar99=cvar(total.dropna(), 0.99), hit_rate=float((ex > 0).mean()))


def mppm(total: pd.Series, rf: pd.Series, rho: float = 3.0, dt: float = 1 / 12) -> float:
    x = ((1 + total) / (1 + rf)).dropna()
    return float(np.log(np.mean(x ** (1 - rho))) / ((1 - rho) * dt))


def certainty_equivalent(total: pd.Series, gamma: float) -> float:
    """Annualised CRRA certainty-equivalent return."""
    r = total.dropna()
    ce_m = np.mean((1 + r) ** (1 - gamma)) ** (1 / (1 - gamma)) - 1
    return float((1 + ce_m) ** 12 - 1)


def ce_gain(total: pd.Series, bench: pd.Series, gamma: float) -> float:
    idx = total.dropna().index.intersection(bench.dropna().index)
    return certainty_equivalent(total.loc[idx], gamma) - certainty_equivalent(bench.loc[idx], gamma)


def sharpe_diff_test(r1: pd.Series, r2: pd.Series, lags: int = 3) -> dict:
    """Ledoit–Wolf (2008) HAC inference on SR1 − SR2 (monthly excess returns)."""
    d = pd.concat([r1, r2], axis=1).dropna().to_numpy(float)
    T = len(d)
    mu = d.mean(axis=0)
    g = (d ** 2).mean(axis=0)
    sr = mu / np.sqrt(g - mu ** 2)
    grad = np.array([g[0] / (g[0] - mu[0] ** 2) ** 1.5, -g[1] / (g[1] - mu[1] ** 2) ** 1.5,
                     -0.5 * mu[0] / (g[0] - mu[0] ** 2) ** 1.5, 0.5 * mu[1] / (g[1] - mu[1] ** 2) ** 1.5])
    y = np.column_stack([d[:, 0] - mu[0], d[:, 1] - mu[1], d[:, 0] ** 2 - g[0], d[:, 1] ** 2 - g[1]])
    psi = y.T @ y / T
    for j in range(1, lags + 1):
        G = y[j:].T @ y[:-j] / T
        psi += (1 - j / (lags + 1)) * (G + G.T)
    se = float(np.sqrt(grad @ psi @ grad / T))
    diff = float(sr[0] - sr[1])
    return dict(sr1_ann=sr[0] * np.sqrt(12), sr2_ann=sr[1] * np.sqrt(12), diff_ann=diff * np.sqrt(12),
                t=diff / se, p=float(2 * stats.norm.sf(abs(diff / se))), months=T)


def deflated_sharpe(excess: pd.Series, n_trials: int, sr_trials_var: float) -> dict:
    """DSR with monthly (non-annualised) Sharpe ratios; ``sr_trials_var`` = variance of trial Sharpe ratios."""
    r = excess.dropna()
    T = len(r)
    sr = r.mean() / r.std()
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    sr0 = np.sqrt(sr_trials_var) * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials)
                                    + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e))) if n_trials > 1 else 0.0
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return dict(sr_monthly=float(sr), sr0_monthly=float(sr0), dsr=float(stats.norm.cdf(z)), n_trials=n_trials)


def attribution(excess: pd.Series, factors: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    res = ols_nw(excess, factors, lags)
    out = res.summary()
    out.loc["const", "coef_ann"] = 12 * res.params["const"]
    out["r2"] = res.rsquared
    out["nobs"] = res.nobs
    return out
