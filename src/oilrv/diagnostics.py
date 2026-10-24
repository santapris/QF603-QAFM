"""Steps 04-05: unit roots, rolling stationarity, SARIMA against naive forecasts, and the long-run
links (Engle-Granger for Brent-WTI and HO-RB, Johansen for the three-leg crack).

Every rolling estimate dated t uses only the `window` observations strictly before t, which is the
same timing the backtest will use.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.exceptions import ComplexWarning
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# --------------------------------------------------------------------------- windows
def month_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions of the first trading day of each month."""
    m = index.to_period("M")
    return np.flatnonzero(np.r_[True, m[1:] != m[:-1]])


def window_before(df: pd.DataFrame | pd.Series, pos: int, window: int):
    """The `window` rows strictly before row position `pos` (no look-ahead)."""
    if pos < window:
        raise ValueError(f"need {window} rows before position {pos}")
    return df.iloc[pos - window:pos]


# --------------------------------------------------------------------------- 4a. unit roots
def adf_p(x: np.ndarray | pd.Series) -> float:
    return float(adfuller(np.asarray(x, dtype=float), regression="c", autolag="AIC", result_object=False)[1])


def unit_root_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) on each series' level and first difference."""
    rows = []
    for name, x in series.items():
        for form, s in [("level", x), ("diff", x.diff())]:
            s = s.dropna()
            adf = adfuller(s, regression="c", autolag="AIC", result_object=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)   # p clipped to [0.01, 0.1]
                kp = kpss(s, regression="c", nlags="auto", result_object=False)
            rows.append({
                "series": name, "form": form, "adf_stat": adf[0], "adf_p": adf[1],
                "adf_lags": adf[2], "adf_crit_5pct": adf[4]["5%"],
                "kpss_stat": kp[0], "kpss_p": kp[1],
                "adf_says": "stationary" if adf[1] < 0.05 else "unit root",
                "kpss_says": "not stationary" if kp[1] < 0.05 else "stationary",
            })
    return pd.DataFrame(rows).set_index(["series", "form"])


def ar1(x: np.ndarray | pd.Series) -> tuple[float, float]:
    """AR(1) with a constant: x_t = a + b x_{t-1} + e. Returns (b, half-life in observations)."""
    v = np.asarray(x, dtype=float)
    b = float(sm.OLS(v[1:], sm.add_constant(v[:-1])).fit().params[1])
    hl = float(np.log(2) / -np.log(b)) if 0 < b < 1 else np.inf
    return b, hl


def rolling_stationarity(x: pd.Series, window: int = 756) -> pd.DataFrame:
    """At each month start: ADF p-value, AR(1) b, half-life, mean and sd of the `window` days before it."""
    rows = []
    for s in month_starts(x.index):
        if s < window:
            continue
        w = window_before(x, s, window).to_numpy()
        b, hl = ar1(w)
        rows.append({"date": x.index[s], "adf_p": adf_p(w), "ar1_b": b, "half_life": hl,
                     "mean": w.mean(), "sd": w.std(ddof=1), "last": w[-1]})
    out = pd.DataFrame(rows).set_index("date")
    out["z_last"] = (out["last"] - out["mean"]) / out["sd"]
    return out


def stationarity_summary(rs: pd.DataFrame, periods: list, hl_band: tuple[float, float]) -> pd.DataFrame:
    """Share of months with ADF p < 0.05 and with a half-life in the band, overall and by period."""
    def summ(d):
        return {"months": len(d), "adf_pass_rate": (d["adf_p"] < 0.05).mean(),
                "adf_p_lt_0.10": (d["adf_p"] < 0.10).mean(), "median_adf_p": d["adf_p"].median(),
                "hl_in_band_rate": d["half_life"].between(*hl_band).mean(),
                "median_half_life": d.loc[np.isfinite(d["half_life"]), "half_life"].median()}
    rows = {"all": summ(rs)}
    for start, end in periods:
        d = rs.loc[start:end]
        if len(d):
            rows[f"{start[:4]}–{(end or str(rs.index[-1].date()))[:4]}"] = summ(d)
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- 4b. SARIMA
def fit_sarima(x: np.ndarray, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        return SARIMAX(np.asarray(x, dtype=float), order=order, seasonal_order=seasonal_order,
                       trend="c").fit(disp=False)


def sarima_vs_naive(monthly: pd.Series, train: int = 60, order=(1, 0, 0),
                    seasonal_order=(1, 0, 0, 12)) -> pd.DataFrame:
    """Rolling one-month-ahead forecasts: SARIMA (refit on the `train` months before each forecast)
    against a random walk (last month) and a seasonal naive forecast (same month last year)."""
    v = monthly.to_numpy(dtype=float)
    rows = []
    for t in range(train, len(v)):
        res = fit_sarima(v[t - train:t], order, seasonal_order)
        rows.append({"date": monthly.index[t], "actual": v[t], "sarima": float(res.forecast(1)[0]),
                     "rw": v[t - 1], "snaive": v[t - 12]})
    out = pd.DataFrame(rows).set_index("date")
    for m in ["sarima", "rw", "snaive"]:
        out[f"e_{m}"] = out["actual"] - out[m]
    return out


def diebold_mariano(e1: pd.Series, e2: pd.Series, lags: int | None = None) -> dict:
    """DM test on squared-error loss. d = e1^2 - e2^2; H0: equal accuracy.

    Negative mean d means model 1 is more accurate. Uses Newey-West HAC standard errors.
    """
    d = (e1 ** 2 - e2 ** 2).to_numpy()
    n = len(d)
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    r = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean_d": float(r.params[0]), "dm_stat": float(r.tvalues[0]), "p": float(r.pvalues[0]),
            "n": n, "hac_lags": lags}


def forecast_summary(fc: pd.DataFrame) -> dict:
    rmse = {m: float(np.sqrt((fc[f"e_{m}"] ** 2).mean())) for m in ["sarima", "rw", "snaive"]}
    dm_rw = diebold_mariano(fc["e_sarima"], fc["e_rw"])
    dm_sn = diebold_mariano(fc["e_sarima"], fc["e_snaive"])
    return {"rmse_sarima": rmse["sarima"], "rmse_rw": rmse["rw"], "rmse_snaive": rmse["snaive"],
            "ratio_vs_rw": rmse["sarima"] / rmse["rw"], "ratio_vs_snaive": rmse["sarima"] / rmse["snaive"],
            "dm_vs_rw_p": dm_rw["p"], "dm_vs_snaive_p": dm_sn["p"], "n_forecasts": len(fc),
            "first": str(fc.index[0].date()), "last": str(fc.index[-1].date())}


def sarima_residual_check(monthly: pd.Series, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)) -> dict:
    """Full-sample SARIMA fit: coefficients and Ljung-Box on residuals at lags 12 and 24."""
    res = fit_sarima(monthly.to_numpy(), order, seasonal_order)
    lb = acorr_ljungbox(res.resid[13:], lags=[12, 24])
    return {"params": dict(zip(res.model.param_names, np.round(res.params, 4))),
            "aic": float(res.aic), "lb_p_12": float(lb["lb_pvalue"].iloc[0]),
            "lb_p_24": float(lb["lb_pvalue"].iloc[1])}


# --------------------------------------------------------------------------- 5a. Engle-Granger
def eg_pair(y: pd.Series, x: pd.Series) -> dict:
    """y on x in levels, the Engle-Granger test, and the fixed 1:1 spread y - x.

    The slope test (H0: slope = 1) uses Newey-West standard errors; with I(1) regressors it is only
    indicative (the OLS slope is super-consistent but its t-statistic is not standard).
    """
    df = pd.concat([y, x], axis=1).dropna()
    r = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(np.floor(4 * (len(df) / 100) ** (2 / 9)))})
    a, b = float(r.params.iloc[0]), float(r.params.iloc[1])
    eg_t, eg_p, _ = coint(df.iloc[:, 0], df.iloc[:, 1], trend="c", autolag="aic")
    resid = df.iloc[:, 0] - a - b * df.iloc[:, 1]
    fixed = df.iloc[:, 0] - df.iloc[:, 1]
    return {"alpha": a, "beta": b, "beta_se_hac": float(r.bse.iloc[1]),
            "t_beta_eq_1": (b - 1) / float(r.bse.iloc[1]), "eg_t": float(eg_t), "eg_p": float(eg_p),
            "half_life_resid": ar1(resid)[1], "adf_p_fixed_1to1": adf_p(fixed),
            "half_life_fixed_1to1": ar1(fixed)[1], "n": len(df)}


def rolling_eg(y: pd.Series, x: pd.Series, window: int = 756) -> pd.DataFrame:
    df = pd.concat([y, x], axis=1).dropna()
    rows = []
    for s in month_starts(df.index):
        if s < window:
            continue
        w = window_before(df, s, window)
        r = eg_pair(w.iloc[:, 0], w.iloc[:, 1])
        rows.append({"date": df.index[s], **{k: r[k] for k in ["beta", "eg_p", "adf_p_fixed_1to1",
                                                               "half_life_fixed_1to1"]}})
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- 5b. Johansen
def distance_to_space(target: np.ndarray, beta: np.ndarray) -> float:
    """Share of `target`'s length outside the column space of `beta` (0 = inside, 1 = orthogonal)."""
    if beta.shape[1] == 0:
        return 1.0
    coef, *_ = np.linalg.lstsq(beta, target, rcond=None)
    return float(np.linalg.norm(target - beta @ coef) / np.linalg.norm(target))


def johansen(levels: pd.DataFrame, k_ar_diff: int = 1, det_order: int = 0,
             targets: dict[str, list[float]] | None = None) -> dict:
    """Johansen trace test on the columns of `levels`.

    Returns the trace statistics with 95% critical values, the rank (sequential trace test at 95%),
    the first eigenvector normalised to -3 on the first column (comparable with 3:2:1), and for each
    target vector its distance to the estimated cointegration space.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        j = coint_johansen(levels.to_numpy(dtype=float), det_order=det_order, k_ar_diff=k_ar_diff)
    evec = j.evec
    if np.iscomplexobj(evec):
        # statsmodels can return a complex dtype; on this data every imaginary part is exactly 0
        if np.abs(np.imag(evec)).max() > 1e-10:
            raise ValueError("Johansen eigenvectors have non-zero imaginary parts")
        evec = np.real(evec)
    rank = 0
    for r in range(levels.shape[1]):
        if j.lr1[r] > j.cvt[r, 1]:
            rank = r + 1
        else:
            break
    v1 = evec[:, 0] * (-3.0 / evec[0, 0])
    out = {"rank": rank, **{f"trace_r{r}": float(j.lr1[r]) for r in range(levels.shape[1])},
           **{f"crit95_r{r}": float(j.cvt[r, 1]) for r in range(levels.shape[1])},
           **{f"v1_{c}": float(v) for c, v in zip(levels.columns, v1)}}
    beta = evec[:, :rank]
    for name, vec in (targets or {}).items():
        out[f"dist_{name}"] = distance_to_space(np.asarray(vec, dtype=float), beta)
    return out


def rolling_johansen(levels: pd.DataFrame, window: int = 756, k_ar_diff: int = 1,
                     targets: dict[str, list[float]] | None = None) -> pd.DataFrame:
    rows = []
    for s in month_starts(levels.index):
        if s < window:
            continue
        r = johansen(window_before(levels, s, window), k_ar_diff=k_ar_diff, targets=targets)
        rows.append({"date": levels.index[s], **r})
    return pd.DataFrame(rows).set_index("date")
