"""Part 4 tests: seasonal fair value, GARCH filter, VECM loadings, Johansen weights, and no look-ahead
in the rolling fits and the daily point-in-time path."""
import numpy as np
import pandas as pd
import pytest

from oilrv import model as M
from oilrv.config import load_config

W = 300                                                  # short window keeps the tests fast


def _ar1(n, b, sd, rng):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = b * x[t - 1] + rng.normal(0, sd)
    return x


def synthetic_legs(n=900, seed=0, seasonal=0.0):
    """CL a random walk; RB and HO = CL + mean-reverting cracks (+ optional month effect on HO);
    Brent = CL + a mean-reverting spread. All in $/bbl, columns as in data/clean/spreads.csv."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cl = 70 + np.cumsum(rng.normal(0, 1.0, n))
    season = seasonal * np.sin(2 * np.pi * idx.month.to_numpy() / 12)
    return pd.DataFrame({
        "cl_bbl": cl,
        "rb_bbl": cl + 15 + _ar1(n, 0.95, 0.6, rng),
        "ho_bbl": cl + 25 + _ar1(n, 0.95, 0.6, rng) + season,
        "brent_bbl": cl + 4 + _ar1(n, 0.9, 0.4, rng),
    }, index=idx)


# ------------------------------------------------------------------ fair value
def test_seasonal_effects_recover_planted_pattern_and_fill_missing_months():
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    s = pd.Series(10 + 3 * (idx.month == 7) - 3 * (idx.month == 1), index=idx, dtype=float)
    seas = M.seasonal_effects(s)
    assert seas[7] - seas[3] == pytest.approx(3) and seas[1] - seas[3] == pytest.approx(-3)
    part = s.loc["2018-01-01":"2018-05-31"]
    assert M.seasonal_effects(part)[9] == 0.0                # September absent -> 0
    r = M.deseasonalise(s, seas)
    assert r.std() == pytest.approx(0, abs=1e-12)


def test_fit_window_z_definition_and_gate():
    cfg = load_config()
    legs = synthetic_legs(seasonal=4.0)
    p = M.fit_window(legs.iloc[:W], "crack321", cfg=cfg)
    S = M.spread_from_legs(legs, cfg["spreads"]["crack321"]["legs"], 3)
    t = legs.index[W]
    z = M.zscore(S.loc[t], t.month, p)
    assert z == pytest.approx((S.loc[t] - p[f"seas_{t.month}"] - p["mean"]) / p["sd"])
    assert p["gate"] == (5 <= p["half_life"] <= 120)
    q = M.fit_window(legs.iloc[:W], "brent_wti", cfg=cfg)   # not seasonally adjusted
    assert all(q[f"seas_{m}"] == 0.0 for m in range(1, 13))


# ------------------------------------------------------------------ GARCH
def test_garch_filter_matches_recursion_and_fallback():
    g = {"mu": 0.1, "omega": 0.05, "a": 0.1, "b": 0.85, "nu": 5.0, "sig2_next": 1.0, "ok": True}
    dx = np.array([0.5, -2.0, 1.0])
    s2, exp = 1.0, []
    for x in dx:
        s2 = 0.05 + 0.1 * (x - 0.1) ** 2 + 0.85 * s2
        exp.append(np.sqrt(s2))
    assert np.allclose(M.garch_filter(g, dx), exp)
    bad = {**g, "ok": False, "sig2_next": 4.0}
    assert np.allclose(M.garch_filter(bad, dx), 2.0)


def test_fit_garch_recovers_persistence_and_falls_back():
    rng = np.random.default_rng(1)
    n, w, a, b = 3000, 0.02, 0.1, 0.85
    x, s2 = np.empty(n), w / (1 - a - b)
    for t in range(n):
        x[t] = np.sqrt(s2) * rng.standard_t(6) / np.sqrt(6 / 4)
        s2 = w + a * x[t] ** 2 + b * s2
    g = M.fit_garch(x)
    assert g["ok"] and g["a"] + g["b"] == pytest.approx(a + b, abs=0.05)
    assert not M.fit_garch(np.zeros(100))["ok"]


# ------------------------------------------------------------------ VECM and Johansen
def test_vecm_known_vector_finds_correcting_leg():
    """Only RB pulls the gasoline crack back; CL and HO are random walks around it."""
    rng = np.random.default_rng(2)
    n = 3000
    cl = 70 + np.cumsum(rng.normal(0, 1, n))
    rb = np.empty(n)
    rb[0] = cl[0] + 15
    for t in range(1, n):
        rb[t] = rb[t - 1] + (cl[t] - cl[t - 1]) - 0.05 * (rb[t - 1] - cl[t - 1] - 15) + rng.normal(0, 0.5)
    legs = pd.DataFrame({"cl_bbl": cl, "rb_bbl": rb}, index=pd.bdate_range("2010-01-01", periods=n))
    v = M.fit_vecm_known(legs, {"cl": -1, "rb": 1}, 1)
    assert v["vecm_alpha_rb"] == pytest.approx(-0.05, abs=0.035) and v["vecm_t_rb"] < -4   # se is about 0.013
    assert abs(v["vecm_t_cl"]) < 2.5
    assert v["vecm_spread_speed"] == pytest.approx(-0.05, abs=0.035)


def test_johansen_weights_valid_on_cointegrated_and_none_on_random_walks():
    jw = M.johansen_weights(synthetic_legs(1500))
    assert jw["jo_rank"] >= 1 and jw["weights"]["cl"] == pytest.approx(-3)
    rng = np.random.default_rng(3)
    rw = pd.DataFrame(70 + np.cumsum(rng.normal(size=(1500, 3)), axis=0), columns=["cl_bbl", "rb_bbl", "ho_bbl"])
    assert M.johansen_weights(rw)["weights"] is None


# ------------------------------------------------------------------ no look-ahead
@pytest.mark.parametrize("name,variant", [("crack321", "fixed"), ("ho_rb", "fixed"), ("crack321", "johansen")])
def test_rolling_fits_and_path_ignore_the_future(name, variant):
    cfg = load_config()
    legs = synthetic_legs(seasonal=3.0)
    fits = M.rolling_fits(legs, name, variant, window=W, cfg=cfg)
    path = M.model_path(legs, fits, name, cfg)
    cut = path.index[len(path) // 2]                      # a day in the middle of some month
    legs2 = legs.copy()
    legs2.loc[legs2.index > cut] += np.random.default_rng(4).normal(0, 20, legs2.loc[legs2.index > cut].shape)
    fits2 = M.rolling_fits(legs2, name, variant, window=W, cfg=cfg)
    path2 = M.model_path(legs2, fits2, name, cfg)
    pd.testing.assert_frame_equal(fits.loc[:cut], fits2.loc[:cut])
    pd.testing.assert_frame_equal(path.loc[:cut], path2.loc[:cut])
    assert not path["z"].loc[cut:].iloc[1:].equals(path2["z"].loc[cut:].iloc[1:])


def test_fit_uses_window_strictly_before_month_start():
    cfg = load_config()
    legs = synthetic_legs()
    fits = M.rolling_fits(legs, "brent_wti", window=W, cfg=cfg)
    d, pos = fits.index[0], int(fits["pos"].iloc[0])
    assert legs.index[pos] == d
    manual = M.fit_window(legs.iloc[pos - W:pos], "brent_wti", cfg=cfg)
    assert fits.iloc[0]["mean"] == pytest.approx(manual["mean"])
    assert fits.iloc[0]["S_last"] == pytest.approx(legs["brent_bbl"].iloc[pos - 1] - legs["cl_bbl"].iloc[pos - 1])


def test_path_spread_and_sigma_timing():
    cfg = load_config()
    legs = synthetic_legs()
    fits = M.rolling_fits(legs, "brent_wti", window=W, cfg=cfg)
    path = M.model_path(legs, fits, "brent_wti", cfg)
    S = legs["brent_bbl"] - legs["cl_bbl"]
    assert np.allclose(path["S"], S.loc[path.index])
    assert np.allclose(path["dS"], S.diff().loc[path.index])
    # sigma_next on the first day of a month uses that month's fit and that day's dS
    p = fits.iloc[1]
    first = path.loc[fits.index[1]]
    g = M._garch_dict(p)
    assert first["sigma_next"] == pytest.approx(M.garch_filter(g, np.array([first["dS"]]))[0])


def test_vol_forecast_test_prefers_true_garch():
    rng = np.random.default_rng(5)
    n, s2 = 4000, 1.0
    rows = []
    for t in range(n):
        x = np.sqrt(s2) * rng.normal()
        s2_next = 0.05 + 0.15 * x ** 2 + 0.8 * s2
        rows.append((x, np.sqrt(s2_next)))
        s2 = s2_next
    dS, sig = np.array(rows).T
    path = pd.DataFrame({"dS": dS, "sigma_next": sig, "dS_sd": dS.std(), "fit_date": 0},
                        index=pd.RangeIndex(n))
    r = M.vol_forecast_test(path)
    assert r["mean_diff"] < 0 and r["p"] < 0.01
