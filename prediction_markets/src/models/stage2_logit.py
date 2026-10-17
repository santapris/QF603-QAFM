"""
Stage 2: is the calibration bias statistically real, not just noise in a
small sample?

Fit outcome ~ f(price) by logistic regression. A perfectly calibrated
market implies the fitted curve equals the 45-degree line; testing the
fitted coefficients against that line is the significance test for the
favorite-longshot bias (the discrete-choice analogue of a Newey-West t-test
on a VRP regression coefficient).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_calibration_logit(df: pd.DataFrame, include_price_sq: bool = True):
    """Logit of outcome on price (and optionally price^2, to let the curve
    bow away from a straight line the way the bucket chart typically does).

    Returns the fitted statsmodels result; `.summary()` gives coefficients,
    standard errors, and p-values directly.
    """
    work = df.dropna(subset=["implied_prob", "outcome"]).copy()
    X = pd.DataFrame({"price": work["implied_prob"]})
    if include_price_sq:
        X["price_sq"] = work["implied_prob"] ** 2
    X = sm.add_constant(X)
    y = work["outcome"]

    model = sm.Logit(y, X)
    return model.fit(disp=False)


def predicted_vs_raw_price(result, df: pd.DataFrame, include_price_sq: bool = True) -> pd.DataFrame:
    """Compare the logit-implied probability to the raw market price, per
    market -- this `fitted` column is the bias-corrected forecast that
    Stage 5's out-of-sample test should be scored against."""
    work = df.dropna(subset=["implied_prob", "outcome"]).copy()
    X = pd.DataFrame({"price": work["implied_prob"]})
    if include_price_sq:
        X["price_sq"] = work["implied_prob"] ** 2
    X = sm.add_constant(X)
    work["fitted_prob"] = result.predict(X)
    return work[["market_id", "platform", "implied_prob", "fitted_prob", "outcome"]]
