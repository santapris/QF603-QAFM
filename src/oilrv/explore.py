"""Step 03 (and the seasonality half of Step 04): summary moments, volatility clustering,
STL decomposition and month-of-year effects."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.seasonal import STL

SPREADS = ["crack321", "brent_wti", "ho_rb"]


def daily_changes(sp: pd.DataFrame, cols: list[str] = SPREADS) -> pd.DataFrame:
    """Daily changes in $/bbl of each spread."""
    return sp[cols].diff().dropna()


def _arch_lm_p(x: pd.Series, nlags: int = 5) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)      # statsmodels return-type notice
        return float(het_arch(x, nlags=nlags)[1])


def moments_table(d: pd.DataFrame) -> pd.DataFrame:
    """Four moments plus tail diagnostics for each column of daily changes ($/bbl).

    `excess_kurt` is pandas' kurt() (already excess). `jb_p` is the Jarque-Bera p-value
    (H0: normal). `arch_lm_p` is Engle's ARCH-LM test on 5 lags (H0: no volatility clustering).
    `p_gt_4sd_*` compares the share of |moves| beyond 4 sd with what a normal predicts.
    `t_df` is the degrees of freedom of a Student-t fitted by maximum likelihood.
    """
    rows = {}
    for c in d:
        x = d[c].dropna()
        z = (x - x.mean()) / x.std()
        rows[c] = {
            "n": len(x), "mean": x.mean(), "std": x.std(), "skew": x.skew(),
            "excess_kurt": x.kurt(), "min": x.min(), "max": x.max(),
            "jb_p": stats.jarque_bera(x).pvalue,
            "arch_lm_p": _arch_lm_p(x - x.mean()),
            "p_gt_4sd_obs": float((z.abs() > 4).mean()),
            "p_gt_4sd_normal": float(2 * stats.norm.sf(4)),
            "t_df": stats.t.fit(x.to_numpy())[0],
        }
    return pd.DataFrame(rows).T


def period_label(start: str, end: str | None, last: pd.Timestamp) -> str:
    a, b = start[:4], (end or str(last.date()))[:4]
    return a if a == b else f"{a}–{b}"


def moments_by_period(d: pd.DataFrame, periods: list) -> pd.DataFrame:
    """Mean, sd, skew and excess kurtosis of daily changes per period (rows: series x period)."""
    out = []
    for start, end in periods:
        s = d.loc[start:end]
        out.append(pd.DataFrame({"n": s.count(), "mean": s.mean(), "std": s.std(), "skew": s.skew(),
                                 "excess_kurt": s.kurt()})
                   .assign(period=period_label(start, end, d.index[-1])))
    return pd.concat(out).rename_axis("series").reset_index().set_index(["series", "period"])


def rolling_vol(x: pd.Series, window: int = 60) -> pd.Series:
    return x.rolling(window).std()


def monthly_mean(x: pd.Series) -> pd.Series:
    """Calendar-month average, indexed by month-end."""
    return x.resample("ME").mean().dropna()


def stl_decompose(x: pd.Series, period: int = 12):
    """Robust STL on a regularly spaced series (monthly: period 12)."""
    return STL(x.reset_index(drop=True), period=period, robust=True).fit()


def stl_strength(res, period: int = 12) -> dict:
    """Hyndman & Athanasopoulos strength measures, both in [0, 1].

    F_trend = max(0, 1 - Var(R) / Var(T + R)),  F_seasonal = max(0, 1 - Var(R) / Var(S + R)).
    Also reports the seasonal amplitude (peak-to-trough of the average seasonal profile, $/bbl).
    """
    T, S, R = np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid)
    prof = pd.Series(S).groupby(np.arange(len(S)) % period).mean()
    return {
        "trend_strength": max(0.0, 1 - np.var(R) / np.var(T + R)),
        "seasonal_strength": max(0.0, 1 - np.var(R) / np.var(S + R)),
        "seasonal_sd": float(S.std()),
        "resid_sd": float(R.std()),
        "seasonal_peak_to_trough": float(prof.max() - prof.min()),
    }


def detrended_by_month(monthly: pd.Series, span: int = 13) -> pd.DataFrame:
    """Month-of-year effect on a monthly series: value minus a centred `span`-month moving average.

    Returns mean, sd and count by calendar month; attrs['kruskal_p'] tests equal distributions
    across months (H0: no seasonality). The centred average uses future months, which is fine for
    a full-sample diagnostic; the model (Part 4) estimates seasonal effects inside its window only.
    """
    dev = (monthly - monthly.rolling(span, center=True).mean()).dropna()
    g = dev.groupby(dev.index.month)
    out = g.agg(["mean", "std", "count"])
    out.index.name = "month"
    out.attrs["kruskal_p"] = float(stats.kruskal(*[v.values for _, v in g]).pvalue)
    return out
