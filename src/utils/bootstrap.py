"""Moving-block bootstrap p-values for predictive-regression slopes [F11].

Studentised, centred test: resample (y_t, X_t) rows jointly in circular blocks, re-estimate
beta* and its Newey-West SE, form t* = (beta* - beta_hat) / se*, and report
p = mean(|t*| >= |t_hat|). This keeps the serial dependence within blocks and does not rely on
the asymptotic normal approximation that is unreliable with ~150 overlapping monthly observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import load_config
from src.utils.newey_west import _design, _fit, nw_cov, ols_nw


def block_indices(T: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices for one circular moving-block resample of length T."""
    n_blocks = -(-T // block_length)
    starts = rng.integers(0, T, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_length)[None, :]) % T
    return idx.ravel()[:T]


def mbb_pvalues(y, X, lags: int, add_const: bool = True, block_length: int | None = None,
                reps: int | None = None, seed: int | None = None,
                cfg: dict | None = None) -> pd.Series:
    """Two-sided moving-block bootstrap p-values for each coefficient (defaults from config)."""
    cfg = cfg or load_config()
    block_length = block_length or cfg["bootstrap"]["block_length"]
    reps = reps or cfg["bootstrap"]["reps"]
    seed = cfg["seed"] if seed is None else seed

    base = ols_nw(y, X, lags, add_const)
    ys, Xs = _design(y, X, add_const)
    yv, Xv = ys.to_numpy(float), Xs.to_numpy(float)
    beta_hat, t_hat = base.params.to_numpy(), base.tstat.to_numpy()

    rng = np.random.default_rng(seed)
    T = len(yv)
    exceed = np.zeros_like(beta_hat)
    valid = 0
    for _ in range(reps):
        idx = block_indices(T, block_length, rng)
        yb, Xb = yv[idx], Xv[idx]
        try:
            b, u = _fit(yb, Xb)
            se = np.sqrt(np.clip(np.diag(nw_cov(Xb, u, lags)), 0, None))
        except np.linalg.LinAlgError:
            continue
        if np.any(se == 0):
            continue
        exceed += np.abs((b - beta_hat) / se) >= np.abs(t_hat)
        valid += 1
    if valid == 0:
        raise RuntimeError("no valid bootstrap replications")
    return pd.Series(exceed / valid, index=base.params.index, name="p_mbb")
