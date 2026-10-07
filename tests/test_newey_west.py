import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.utils.io import load_config
from src.utils.newey_west import hodrick_1b, nw_lags, ols_nw


def ar1(T, rho, rng):
    x = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x


def test_nw_lags_rule():
    cfg = load_config()
    assert nw_lags(1, 200, cfg=cfg) == 4          # floor(4*(2)**(2/9)) = 4
    assert nw_lags(6, 200, cfg=cfg) == 6
    assert nw_lags(6, 200, expost=True, cfg=cfg) == 7


@pytest.mark.parametrize("lags", [0, 3, 6])
def test_matches_statsmodels_hac(lags):
    rng = np.random.default_rng(0)
    T = 180
    X = pd.DataFrame({"a": ar1(T, 0.8, rng), "b": rng.standard_normal(T)})
    y = pd.Series(0.3 * X["a"] + ar1(T, 0.5, rng))
    ours = ols_nw(y, X, lags)
    ref = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    np.testing.assert_allclose(ours.params.values, ref.params.values, rtol=1e-10)
    np.testing.assert_allclose(ours.se.values, ref.bse.values, rtol=1e-8)
    assert ours.nobs == T and abs(ours.rsquared - ref.rsquared) < 1e-10


def test_drops_missing_rows():
    rng = np.random.default_rng(1)
    X = pd.Series(rng.standard_normal(50), name="x")
    y = pd.Series(rng.standard_normal(50))
    y.iloc[:5] = np.nan
    assert ols_nw(y, X, 2).nobs == 45


def test_hodrick_point_estimate_equals_ols():
    rng = np.random.default_rng(2)
    T, k = 200, 3
    x = pd.Series(ar1(T, 0.7, rng), name="x")
    r1 = pd.Series(rng.standard_normal(T) * 0.04)
    res = hodrick_1b(r1, x, k)
    y_k = sum(r1.shift(-j) for j in range(k))
    ref = ols_nw(y_k, x, 0)
    np.testing.assert_allclose(res.params.values, ref.params.values, rtol=1e-10)


def test_hodrick_size_under_null():
    """Under no predictability, the 1B t-test on the slope rejects at roughly the nominal 5% rate."""
    rng = np.random.default_rng(3)
    T, k, reps = 240, 6, 400
    rejections = 0
    for _ in range(reps):
        x = pd.Series(ar1(T, 0.6, rng), name="x")
        r1 = pd.Series(rng.standard_normal(T) * 0.04)
        rejections += hodrick_1b(r1, x, k).pvalue["x"] < 0.05
    assert 0.01 <= rejections / reps <= 0.10
