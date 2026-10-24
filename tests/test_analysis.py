"""Parts 2-3 tests: moments, seasonality, unit-root and cointegration helpers, rolling timing."""
import numpy as np
import pandas as pd
import pytest

from oilrv import diagnostics as dg
from oilrv import explore as ex


def _ar1(n, b, sd=1.0, seed=0, mean=0.0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = mean + b * (x[t - 1] - mean) + rng.normal(0, sd)
    return x


# ------------------------------------------------------------------ Part 2
def test_moments_on_normal_sample():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"x": rng.normal(0, 2, 20000)})
    m = ex.moments_table(d).loc["x"]
    assert m["std"] == pytest.approx(2, rel=0.03)
    assert abs(m["excess_kurt"]) < 0.15
    assert m["t_df"] > 30                       # a fitted t on normal data has large df


def test_moments_by_period_labels():
    idx = pd.bdate_range("2019-01-01", "2022-12-30")
    d = pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)
    t = ex.moments_by_period(d, [["2019-01-01", "2019-12-31"], ["2020-01-01", None]])
    assert set(t.index.get_level_values("period")) == {"2019", "2020–2022"}


def test_month_effect_detects_planted_seasonality():
    idx = pd.date_range("2007-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(2)
    noise = pd.Series(rng.normal(0, 1, 240), index=idx)
    seasonal = noise + 3 * np.sin(2 * np.pi * idx.month / 12)
    assert ex.detrended_by_month(seasonal).attrs["kruskal_p"] < 0.001
    assert ex.detrended_by_month(noise).attrs["kruskal_p"] > 0.01
    res = ex.stl_decompose(seasonal, period=12)
    assert ex.stl_strength(res)["seasonal_strength"] > 0.6
    assert ex.stl_strength(res)["seasonal_peak_to_trough"] == pytest.approx(6, rel=0.25)


# ------------------------------------------------------------------ Part 3
def test_ar1_half_life_recovers_truth():
    b_true = 0.5 ** (1 / 20)                    # half-life 20 days
    b, hl = dg.ar1(_ar1(5000, b_true, seed=3, mean=10))
    assert b == pytest.approx(b_true, abs=0.01)
    assert hl == pytest.approx(20, rel=0.25)
    b_rw, hl_rw = dg.ar1(np.cumsum(np.random.default_rng(14).normal(size=3000)))
    assert b_rw > 0.99 and hl_rw > 120              # a random walk falls outside the entry band


def test_rolling_stationarity_uses_only_past_data():
    idx = pd.bdate_range("2015-01-01", periods=900)
    x = pd.Series(_ar1(900, 0.95, seed=4), index=idx)
    a = dg.rolling_stationarity(x, window=300)
    y = x.copy()
    cut = a.index[3]
    y[y.index >= cut] += 50.0                   # change everything from that month start on
    b = dg.rolling_stationarity(y, window=300)
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])
    assert not np.isclose(a["mean"].iloc[-1], b["mean"].iloc[-1])


def test_unit_root_table_classifies():
    rng = np.random.default_rng(5)
    rw = pd.Series(np.cumsum(rng.normal(size=2000)))
    st = pd.Series(_ar1(2000, 0.5, seed=6))
    t = dg.unit_root_table({"rw": rw, "st": st})
    assert t.loc[("rw", "level"), "adf_says"] == "unit root"
    assert t.loc[("rw", "diff"), "adf_says"] == "stationary"
    assert t.loc[("st", "level"), "adf_says"] == "stationary"


def test_eg_pair_on_cointegrated_pair():
    rng = np.random.default_rng(7)
    x = pd.Series(50 + np.cumsum(rng.normal(size=3000)))
    y = x + 3 + pd.Series(_ar1(3000, 0.9, seed=8))
    r = dg.eg_pair(y, x)
    assert r["eg_p"] < 0.01 and r["beta"] == pytest.approx(1, abs=0.02)
    assert r["adf_p_fixed_1to1"] < 0.01


def test_distance_to_space():
    beta = np.array([[-1, -1], [1, 0], [0, 1]], dtype=float)     # gasoline and diesel cracks
    assert dg.distance_to_space(np.array([-3, 2, 1.0]), beta) == pytest.approx(0, abs=1e-12)
    assert dg.distance_to_space(np.array([1, 1, 1.0]), beta) == pytest.approx(1, abs=1e-12)
    assert dg.distance_to_space(np.array([1, 0, 0.0]), np.empty((3, 0))) == 1.0


def test_johansen_finds_two_product_cracks():
    """CL a random walk; RB and HO each CL plus a stationary crack: rank 2, and 3:2:1 in the space."""
    rng = np.random.default_rng(9)
    n = 2000
    cl = 70 + np.cumsum(rng.normal(0, 1.5, n))
    rb = cl + 15 + _ar1(n, 0.9, 1.0, seed=10)
    ho = cl + 25 + _ar1(n, 0.9, 1.0, seed=11)
    lv = pd.DataFrame({"cl": cl, "rb": rb, "ho": ho})
    r = dg.johansen(lv, 1, targets={"crack321": [-3, 2, 1], "outright_cl": [1, 0, 0]})
    assert r["rank"] == 2
    assert r["dist_crack321"] < 0.05
    assert r["dist_outright_cl"] > 0.5
    assert r["v1_cl"] == pytest.approx(-3.0)


def test_sarima_forecasts_use_only_past():
    idx = pd.date_range("2010-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(12)
    m = pd.Series(20 + 3 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 1, 48), index=idx)
    a = dg.sarima_vs_naive(m, train=36)
    m2 = m.copy()
    m2.iloc[-1] += 100
    b = dg.sarima_vs_naive(m2, train=36)
    pd.testing.assert_series_equal(a["sarima"].iloc[:-1], b["sarima"].iloc[:-1])
    assert (a["snaive"] == m.shift(12).loc[a.index]).all()
    assert (a["rw"] == m.shift(1).loc[a.index]).all()


def test_diebold_mariano_sign():
    rng = np.random.default_rng(13)
    e_good = pd.Series(rng.normal(0, 1, 500))
    e_bad = pd.Series(rng.normal(0, 2, 500))
    r = dg.diebold_mariano(e_good, e_bad)
    assert r["mean_d"] < 0 and r["p"] < 0.01
