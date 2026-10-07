"""OLS with Newey-West (Bartlett) HAC standard errors, and Hodrick (1992) 1B standard errors.

* ``nw_lags`` evaluates the lag rule from config ([F6]: ex-post targets use ``expost_rule``).
* ``ols_nw`` is a plain-numpy implementation (no small-sample correction, normal p-values); it is
  tested against statsmodels' HAC estimator and is fast enough for bootstrap loops.
* ``hodrick_1b`` is for regressions of k-period overlapping returns on predictors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import load_config


@dataclass
class RegResult:
    params: pd.Series
    se: pd.Series
    tstat: pd.Series
    pvalue: pd.Series
    rsquared: float
    nobs: int
    lags: int
    resid: pd.Series
    method: str

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.params, "se": self.se, "t": self.tstat, "p": self.pvalue})


def nw_lags(k: int, T: int, expost: bool = False, cfg: dict | None = None) -> int:
    """Newey-West lag length from the config rule (e.g. max(k, floor(4*(T/100)**(2/9))))."""
    cfg = cfg or load_config()
    rule = cfg["newey_west"]["expost_rule" if expost else "rule"]
    value = eval(rule, {"__builtins__": {}}, {"max": max, "min": min, "floor": math.floor,
                                               "ceil": math.ceil, "k": k, "T": T})
    return int(value)


def _design(y, X, add_const: bool) -> tuple[pd.Series, pd.DataFrame]:
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    X = X.to_frame() if isinstance(X, pd.Series) else pd.DataFrame(X)
    if add_const:
        X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    data = pd.concat([y.rename("__y__"), X], axis=1, join="inner").dropna()
    return data["__y__"], data.drop(columns="__y__")


def nw_cov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance of OLS coefficients with Bartlett weights 1 - j/(lags+1)."""
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for j in range(1, lags + 1):
        G = Xu[j:].T @ Xu[:-j]
        S += (1 - j / (lags + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def _fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _result(beta, cov, names, y, u, lags, index, method) -> RegResult:
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.norm.sf(np.abs(t))
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(u ** 2) / tss if tss > 0 else np.nan
    mk = lambda a: pd.Series(a, index=names)  # noqa: E731
    return RegResult(mk(beta), mk(se), mk(t), mk(p), float(r2), len(y), int(lags),
                     pd.Series(u, index=index, name="resid"), method)


def ols_nw(y, X, lags: int, add_const: bool = True) -> RegResult:
    """OLS of y on X (aligned on index, rows with NaN dropped) with Newey-West SEs."""
    ys, Xs = _design(y, X, add_const)
    yv, Xv = ys.to_numpy(float), Xs.to_numpy(float)
    beta, u = _fit(yv, Xv)
    cov = nw_cov(Xv, u, lags)
    return _result(beta, cov, Xs.columns, yv, u, lags, ys.index, "newey_west")


def hodrick_1b(r1: pd.Series, X, k: int, add_const: bool = True) -> RegResult:
    """Hodrick (1992) 1B standard errors for y_t = sum_{j=0}^{k-1} r1_{t+j} on X_t.

    ``r1`` is the one-period return indexed by origin: r1[t] is the return from t to t+1.
    ``X`` is indexed by origin t. Rows must be consecutive periods (no gaps).
    Under the null of no predictability, S = (1/T) sum_t w_t w_t' with
    w_t = e_{t+1} * sum_{i=0}^{k-1} X_{t-i}, where e_{t+1} = r1[t] - mean(r1).
    """
    X = X.to_frame() if isinstance(X, pd.Series) else pd.DataFrame(X)
    if add_const:
        X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    data = pd.concat([r1.rename("__r1__"), X], axis=1, join="inner").sort_index()
    if data.isna().any().any():
        raise ValueError("hodrick_1b needs a gap-free sample; trim missing values first")

    r = data["__r1__"].to_numpy(float)
    Xv = data.drop(columns="__r1__").to_numpy(float)
    names = data.columns.drop("__r1__")
    n = len(r)

    # Point estimates: k-period forward sums y_t for t = 0..n-k.
    y_k = np.convolve(r, np.ones(k), mode="valid")          # y_k[t] = r[t] + ... + r[t+k-1]
    X_reg = Xv[: n - k + 1]
    beta, u = _fit(y_k, X_reg)

    # Hodrick 1B covariance.
    e = r - r.mean()
    X_sum = np.vstack([np.convolve(Xv[:, j], np.ones(k), mode="valid") for j in range(Xv.shape[1])]).T
    w = e[k - 1:, None] * X_sum                               # pairs e_{t+1} with X_t + ... + X_{t-k+1}
    S = w.T @ w / len(w)
    Z_inv = np.linalg.inv(X_reg.T @ X_reg / len(X_reg))
    cov = Z_inv @ S @ Z_inv / len(X_reg)

    index = data.index[: n - k + 1]
    return _result(beta, cov, names, y_k, u, k, index, "hodrick_1b")
