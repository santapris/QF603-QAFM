"""Horizon-matched tests: the questions the daily AR pipeline cannot answer.

Steps 4-6 of the test plan ask whether YESTERDAY'S CHANGE predicts today's.
The mean-reversion strategy asks whether TODAY'S DEVIATION predicts the next
few weeks, and the trend sleeve asks whether a 252-day move persists. Those
are different questions at different horizons, and an AR order chosen on
daily differences has almost no power over either (see `ar_power_sim`).

Everything here is DESCRIPTIVE. It changes no registered rule, no gate and
no verdict. Added after every registered and exploratory result was seen.

Deliberately numpy/scipy only, no statsmodels or arch, so the module runs
anywhere the panel loads. Each estimator is checked against simulated series
with known answers in tests/test_horizon.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize_scalar

__all__ = ["ols", "adf", "kpss", "ar_select", "ma1_css", "variance_ratio",
           "reversion_regression", "reversion_rw_null", "vr_power_curve",
           "ar_power_sim", "tsmom_regression", "signal_sharpe"]


# --------------------------------------------------------------------- OLS
def ols(y, X, hac_lags: int | None = None) -> dict:
    """OLS with optional Newey-West (Bartlett) HAC standard errors."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    dof = n - k
    if hac_lags is None:
        V = XtXi * (e @ e) / dof
    else:
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for L in range(1, hac_lags + 1):
            w = 1.0 - L / (hac_lags + 1.0)
            G = Xe[L:].T @ Xe[:-L]
            S += w * (G + G.T)
        V = XtXi @ S @ XtXi * n / dof
    se = np.sqrt(np.diag(V))
    tss = ((y - y.mean()) ** 2).sum()
    return {"b": b, "se": se, "t": b / se, "e": e, "n": n,
            "r2": 1 - (e @ e) / tss if tss > 0 else np.nan,
            "ssr": float(e @ e)}


def _lagmat(x, p):
    return np.column_stack([x[p - i - 1:len(x) - i - 1] for i in range(p)])


# -------------------------------------------------------------- unit root
_ADF_CV = {"nc": (-2.567, -1.941, -1.617),
           "c": (-3.430, -2.861, -2.567),
           "ct": (-3.960, -3.410, -3.120)}


def _adf_p(t: float, regression: str = "c") -> float:
    """Interpolate a p-value on the normal scale through the three tabulated
    asymptotic critical values. Adequate for reporting around 0.01/0.05/0.10."""
    cv = _ADF_CV[regression]
    A = np.column_stack([np.ones(3), np.array(cv)])
    coef = np.linalg.lstsq(A, sps.norm.ppf([0.01, 0.05, 0.10]), rcond=None)[0]
    return float(np.clip(sps.norm.cdf(coef[0] + coef[1] * t), 1e-6, 1 - 1e-6))


def adf(x, maxlag: int | None = None, regression: str = "c"):
    """Augmented Dickey-Fuller, lag chosen by AIC over 0..maxlag.

    Returns (t-statistic, approximate p-value, chosen lag)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100.0) ** 0.25))
    dx = np.diff(x)
    best = None
    for p in range(maxlag + 1):
        m = len(dx) - p
        y = dx[p:]
        cols = [x[p:p + m][:, None]]
        if p:
            cols.append(_lagmat(dx, p)[-m:])
        Z = np.column_stack(cols)
        if regression == "c":
            Z = np.column_stack([Z, np.ones(m)])
        elif regression == "ct":
            Z = np.column_stack([Z, np.ones(m), np.arange(m, dtype=float)])
        r = ols(y, Z)
        aic = m * np.log(r["ssr"] / m) + 2 * Z.shape[1]
        if best is None or aic < best[0]:
            best = (aic, float(r["t"][0]), p)
    return best[1], _adf_p(best[1], regression), best[2]


def kpss(x, regression: str = "c", lags: int | None = None):
    """KPSS (H0: stationary). Returns (statistic, p-value capped at 0.01/0.10)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if regression == "c":
        e = x - x.mean()
    else:
        e = ols(x, np.column_stack([np.ones(n), np.arange(n, dtype=float)]))["e"]
    if lags is None:
        lags = int(np.ceil(12 * (n / 100.0) ** 0.25))
    s2 = (e @ e) / n
    for L in range(1, lags + 1):
        s2 += 2 * (1.0 - L / (lags + 1.0)) * (e[L:] @ e[:-L]) / n
    S = np.cumsum(e)
    stat = float((S @ S) / (n ** 2 * s2))
    cv = {"c": [(0.347, 0.10), (0.463, 0.05), (0.574, 0.025), (0.739, 0.01)],
          "ct": [(0.119, 0.10), (0.146, 0.05), (0.176, 0.025), (0.216, 0.01)]}[regression]
    p = 0.10
    for c, pv in cv:
        if stat > c:
            p = pv
    return stat, p


# ------------------------------------------------------------- AR order
def ar_select(dx, pmax: int = 20) -> dict:
    """AR(0..pmax) with a constant by OLS on a common sample; AIC and BIC."""
    dx = np.asarray(dx, float)
    dx = dx[~np.isnan(dx)]
    m = len(dx) - pmax
    aic, bic, r2 = {}, {}, {}
    for p in range(pmax + 1):
        y = dx[pmax:]
        Z = np.ones((m, 1)) if not p else np.column_stack([_lagmat(dx, p)[-m:], np.ones(m)])
        r = ols(y, Z)
        k = Z.shape[1]
        base = m * np.log(r["ssr"] / m)
        aic[p], bic[p], r2[p] = base + 2 * k, base + np.log(m) * k, r["r2"]
    return {"aic": aic, "bic": bic, "r2": r2,
            "p_aic": min(aic, key=aic.get), "p_bic": min(bic, key=bic.get)}


def ma1_css(dx) -> float:
    """Conditional-sum-of-squares MA(1) on a differenced series.

    theta near -1 is the over-differencing signature: it means the level was
    already stationary and d = 1 removed a root that was not there."""
    dx = np.asarray(dx, float)
    dx = dx[~np.isnan(dx)]
    z = dx - dx.mean()

    def ssr(th):
        e = np.zeros(len(z))
        for i in range(len(z)):
            e[i] = z[i] - th * (e[i - 1] if i else 0.0)
        return float(e @ e)

    return float(minimize_scalar(ssr, bounds=(-0.999, 0.999), method="bounded").x)


# -------------------------------------------------------- variance ratio
def variance_ratio(dx, q: int) -> dict:
    """Lo-MacKinlay (1988) variance ratio on a series of increments.

    VR > 1 means moves persist (trend); VR < 1 means they reverse. z2 is the
    heteroskedasticity-robust statistic (their equation 18), which is the one
    to read for oil: z1 assumes constant variance, which the ARCH tests reject.
    """
    dx = np.asarray(dx, float)
    dx = dx[~np.isnan(dx)]
    n = len(dx)
    mu = dx.mean()
    cs = np.concatenate([[0.0], np.cumsum(dx)])
    xq = cs[q:] - cs[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    va = ((dx - mu) ** 2).sum() / (n - 1)
    vb = ((xq - q * mu) ** 2).sum() / m
    vr = vb / va
    z1 = (vr - 1) / np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n))
    d = (dx - mu) ** 2
    den = d.sum() ** 2
    theta = sum((2.0 * (q - j) / q) ** 2 * (d[j:] * d[:-j]).sum() / den
                for j in range(1, q))
    z2 = (vr - 1) / np.sqrt(theta) if theta > 0 else np.nan
    return {"q": q, "n": n, "vr": float(vr),
            "z1": float(z1), "p1": float(2 * sps.norm.sf(abs(z1))),
            "z2": float(z2), "p2": float(2 * sps.norm.sf(abs(z2)))}


# ------------------------------------------------- horizon-matched tests
def reversion_regression(level: pd.Series, k: int, window: int = 252) -> dict:
    """dS(t -> t+k) = a + b * z(t), where z is the deviation from a trailing
    `window`-day mean in trailing standard deviations.

    This is the bet the mean-reversion strategy makes. b < 0 is reversion.
    Newey-West at k + 5 lags, because the forward windows overlap.

    The t-statistic is NOT standard normal: regressing a forward change on a
    deviation from a trailing mean of the same series is biased negative in
    finite samples. Use `reversion_rw_null` for the critical value.
    """
    lv = pd.Series(level).dropna()
    z = (lv - lv.rolling(window).mean()) / lv.rolling(window).std()
    fwd = lv.shift(-k) - lv
    d = pd.concat([z.rename("z"), fwd.rename("f")], axis=1).dropna()
    if len(d) < 200:
        return {"k": k, "beta": np.nan, "t": np.nan, "r2": np.nan, "n": len(d)}
    X = np.column_stack([d["z"].to_numpy(), np.ones(len(d))])
    r = ols(d["f"].to_numpy(), X, hac_lags=k + 5)
    return {"k": k, "beta": float(r["b"][0]), "t": float(r["t"][0]),
            "r2": float(r["r2"]), "n": len(d),
            "share_of_1sd": float(r["b"][0] / lv.rolling(window).std().mean())}


def reversion_rw_null(n: int, sd: float, k: int, window: int = 252,
                      reps: int = 600, seed: int = 0) -> np.ndarray:
    """Distribution of `reversion_regression`'s t under a pure random walk
    with the same length and daily volatility. Returns the simulated t's."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        w = pd.Series(np.cumsum(rng.standard_normal(n) * sd))
        t = reversion_regression(w, k, window)["t"]
        if np.isfinite(t):
            out.append(t)
    return np.array(out)


def tsmom_regression(badj: pd.Series, lookback: int = 252, horizon: int = 21) -> dict:
    """Next-`horizon`-day P&L on the sign of the past `lookback`-day move
    (Moskowitz, Ooi & Pedersen 2012), one unit, Newey-West at horizon + 5."""
    b = pd.Series(badj).dropna()
    d = pd.concat([np.sign(b - b.shift(lookback)).rename("s"),
                   (b.shift(-horizon) - b).rename("f")], axis=1).dropna()
    d = d[d["s"] != 0]
    X = np.column_stack([d["s"].to_numpy(), np.ones(len(d))])
    r = ols(d["f"].to_numpy(), X, hac_lags=horizon + 5)
    pnl = d["s"] * d["f"]
    return {"beta": float(r["b"][0]), "t": float(r["t"][0]), "n": len(d),
            "sharpe_ann": float(pnl.mean() / pnl.std() * np.sqrt(252 / horizon)),
            "share_long": float((d["s"] > 0).mean())}


def signal_sharpe(badj: pd.Series, signal: pd.Series, start=None) -> dict:
    """Annualised Sharpe of holding `signal` (lagged one day) in a series whose
    daily P&L is the back-adjusted change. One unit, gross of costs."""
    b = pd.Series(badj).dropna()
    pnl = (pd.Series(signal).reindex(b.index).ffill().shift(1) * b.diff()).dropna()
    if start is not None:
        pnl = pnl[pnl.index >= pd.Timestamp(start)]
    if len(pnl) < 250:
        return {"sharpe": np.nan, "t": np.nan, "total": np.nan, "n": len(pnl)}
    return {"sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)),
            "t": float(pnl.mean() / pnl.std() * np.sqrt(len(pnl))),
            "total": float(pnl.sum()), "n": len(pnl)}


# ----------------------------------------------------------- power audits
def ar_power_sim(half_lives=(30, 90, 150), n: int = 4800, reps: int = 40,
                 pmax: int = 20, seed: int = 0) -> pd.DataFrame:
    """Why an AR order on daily differences says nothing about mean reversion.

    Simulate a level that IS a stationary AR(1) with the given half-life, then
    select an AR order on its differences exactly as step 5 does."""
    rng = np.random.default_rng(seed)
    rows = []
    for hl in half_lives:
        phi = 0.5 ** (1.0 / hl)
        picks = []
        for _ in range(reps):
            z = np.zeros(n)
            for i in range(1, n):
                z[i] = phi * z[i - 1] + rng.standard_normal()
            picks.append(ar_select(np.diff(z), pmax)["p_bic"])
        picks = np.array(picks)
        rows.append({"half_life": hl, "phi": phi, "reps": reps,
                     "share_bic_ar0": float((picks == 0).mean()),
                     "median_bic_order": int(np.median(picks))})
    return pd.DataFrame(rows)


def vr_power_curve(sd_etas=(0.0, 0.002, 0.004, 0.006, 0.010, 0.020), q: int = 252,
                   n: int = 4700, reps: int = 400, rho: float = 0.995,
                   seed: int = 3) -> pd.DataFrame:
    """Power of VR(q) against a series that genuinely trends.

    Increments carry a slow hidden drift mu_t = rho*mu_{t-1} + eta, which
    produces real multi-month trends. For each setting we report the momentum
    Sharpe the series actually delivers and how often VR(q) rejects at 5%, so
    a non-significant variance ratio can be read for what it is worth.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sd_eta in sd_etas:
        srs, hits = [], 0
        for _ in range(reps):
            mu = np.zeros(n)
            for i in range(1, n):
                mu[i] = rho * mu[i - 1] + sd_eta * rng.standard_normal()
            d = mu + rng.standard_normal(n)
            s = pd.Series(np.cumsum(d))
            pnl = (np.sign(s - s.shift(252)).shift(1) * s.diff()).dropna()
            srs.append(pnl.mean() / pnl.std() * np.sqrt(252))
            if abs(variance_ratio(d, q)["z2"]) > 1.96:
                hits += 1
        srs = np.array(srs)
        rows.append({"sd_eta": sd_eta, "median_sharpe": float(np.median(srs)),
                     "sharpe_q25": float(np.percentile(srs, 25)),
                     "sharpe_q75": float(np.percentile(srs, 75)),
                     "reject_rate": hits / reps, "reps": reps})
    return pd.DataFrame(rows)
