import numpy as np
import pytest
import pandas as pd

from src.measures.har import har_target, recursive_har
from src.measures.vrp import vrp_measures
from src.utils.dates import month_end_trading_days, trading_days


def synthetic_rv(seed=0, start="2004-01-01", end="2012-12-31"):
    days = trading_days(start, end)
    rng = np.random.default_rng(seed)
    logv = np.zeros(len(days))
    for i in range(1, len(days)):
        logv[i] = 0.98 * logv[i - 1] + 0.2 * rng.standard_normal()
    return pd.Series(np.exp(logv) * 1e-4 * rng.chisquare(1, len(days)), index=days, name="rv")


def test_har_forecast_has_no_lookahead():
    rv = synthetic_rv()
    origins = month_end_trading_days("2007-01-01", "2010-12-31")
    base = recursive_har(rv, origins)
    t = pd.Timestamp("2008-09-30")
    perturbed = rv.copy()
    perturbed.loc[perturbed.index > t] *= 50
    after = recursive_har(perturbed, origins)
    pd.testing.assert_series_equal(base.loc[:t, "har_fcst"], after.loc[:t, "har_fcst"])
    assert not np.allclose(base.loc["2008-12-31":, "har_fcst"], after.loc["2008-12-31":, "har_fcst"])


def test_har_uses_only_fully_observed_targets():
    rv = synthetic_rv()
    origins = month_end_trading_days("2007-01-01", "2007-12-31")
    out = recursive_har(rv, origins)
    p = rv.index.get_loc(origins[0])
    y = har_target(rv, 21)
    # rows s <= p - 21 with a full 22-day history and an observed target
    expected = int(y.iloc[21:p - 21 + 1].notna().sum())
    assert out["n_train"].iloc[0] == expected


def test_vrp_measures_windows():
    days = trading_days("2020-01-01", "2020-06-30")
    rv = pd.Series(1e-4, index=days)                    # constant daily variance → annualised 0.0252
    me = month_end_trading_days("2020-02-01", "2020-04-30")
    iv = pd.Series(0.04, index=me)
    har = pd.Series(0.03, index=me)
    v = vrp_measures(iv, har, rv)
    assert np.allclose(v["rv_fwd"], 0.0252) and np.allclose(v["rv_past"], 0.0252)
    assert np.allclose(v["vrp_exante"], 0.01)
    assert np.allclose(v["vrp_expost"], 0.04 - 0.0252)
    # ex-post observed 21 trading days after the month-end
    assert v.loc["2020-02-28", "obs_date_vrp_expost"] == days[days.get_loc(pd.Timestamp("2020-02-28")) + 21]
    assert np.isnan(v["rv_surprise"].iloc[0]) and np.allclose(v["rv_surprise"].iloc[1:], 0.0252 - 0.03)


def synthetic_iv(rv):
    return (252 * rv.rolling(22, min_periods=1).mean() * 1.3).rename("iv_daily")


def test_all_variants_have_no_lookahead():
    rv = synthetic_rv(seed=1)
    iv = synthetic_iv(rv)
    origins = month_end_trading_days("2007-01-01", "2010-12-31")
    t = pd.Timestamp("2009-03-31")
    rv2, iv2 = rv.copy(), iv.copy()
    rv2.loc[rv2.index > t] *= 30
    iv2.loc[iv2.index > t] *= 30
    for v in ["level", "log", "iv"]:
        a = recursive_har(rv, origins, variant=v, iv_daily=iv)
        b = recursive_har(rv2, origins, variant=v, iv_daily=iv2)
        pd.testing.assert_series_equal(a.loc[:t, "har_fcst"], b.loc[:t, "har_fcst"], obj=v)


def test_log_variant_forecasts_positive_and_roughly_unbiased():
    rv = synthetic_rv(seed=2, start="2003-01-01", end="2015-12-31")
    origins = month_end_trading_days("2006-01-01", "2015-06-30")
    f = recursive_har(rv, origins, variant="log")["har_fcst"]
    realised = har_target(rv, 21).reindex(origins)
    assert (f > 0).all()
    assert 0.8 < realised.mean() / f.mean() < 1.25


def test_iv_variant_loads_on_iv_when_rv_follows_iv():
    days = trading_days("2004-01-01", "2012-12-31")
    rng = np.random.default_rng(3)
    iv = pd.Series(np.exp(np.cumsum(0.05 * rng.standard_normal(len(days)))) * 0.04, index=days)
    rv = iv / 252 * rng.chisquare(20, len(days)) / 20        # daily RV noisy around IV/252
    f = recursive_har(rv, month_end_trading_days("2011-01-01", "2011-12-31"), variant="iv", iv_daily=iv)
    assert f["b_iv"].mean() > 0.5


def test_evaluate_perfect_forecast():
    from src.measures.har import evaluate
    idx = month_end_trading_days("2010-01-01", "2015-12-31")
    rng = np.random.default_rng(4)
    r = pd.Series(np.exp(rng.standard_normal(len(idx))) * 0.03, index=idx)
    t = evaluate(pd.DataFrame({"level": r, "noisy": r * np.exp(0.3 * rng.standard_normal(len(idx)))}), r)
    assert t.loc["level", "qlike"] == pytest.approx(0, abs=1e-12)
    assert t.loc["level", "mz_b"] == pytest.approx(1) and t.loc["level", "mse"] == pytest.approx(0, abs=1e-15)
    assert t.loc["noisy", "qlike"] > 0 and t.loc["noisy", "dm_t_qlike_vs_level"] > 2
