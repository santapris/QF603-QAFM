"""Horizon-test estimators checked against simulated series with known answers.

`horizon.py` implements ADF, KPSS, AR selection and the Lo-MacKinlay variance
ratio directly rather than calling statsmodels, so each one is verified here
against a process whose correct answer is known by construction. Offline.
"""
import numpy as np
import pandas as pd
import pytest

from oilrv import horizon as H

N = 5000
SEED = 0


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(SEED)
    rw = np.cumsum(rng.standard_normal(N))
    phi = 0.99
    ar = np.zeros(N)
    for i in range(1, N):
        ar[i] = phi * ar[i - 1] + rng.standard_normal()
    return {"rw": rw, "ar": ar, "wn": rng.standard_normal(N)}


# ------------------------------------------------------------- unit root
def test_adf_does_not_reject_a_random_walk(series):
    t, p, _ = H.adf(series["rw"])
    assert t > -2.6, "a random walk should not look stationary"
    assert p > 0.10


def test_adf_rejects_a_stationary_ar1(series):
    t, p, _ = H.adf(series["ar"])
    assert t < -3.43, "AR(1) with phi=0.99 should reject at 1%"
    assert p < 0.01


def test_kpss_rejects_a_random_walk(series):
    stat, p = H.kpss(series["rw"])
    assert stat > 0.739 and p <= 0.01


def test_kpss_accepts_white_noise(series):
    stat, p = H.kpss(series["wn"])
    assert stat < 0.463 and p >= 0.05


# ---------------------------------------------------------- over-differencing
def test_ma1_detects_over_differencing(series):
    """Differencing an already-stationary series drives theta toward -1."""
    assert H.ma1_css(np.diff(series["wn"])) < -0.9


def test_ma1_near_zero_when_differencing_is_right(series):
    assert abs(H.ma1_css(np.diff(series["rw"]))) < 0.1


# -------------------------------------------------------- variance ratio
def test_vr_near_one_for_a_random_walk(series):
    v = H.variance_ratio(np.diff(series["rw"]), 21)
    assert 0.8 < v["vr"] < 1.2
    assert v["p2"] > 0.05


def test_vr_below_one_for_a_mean_reverting_level(series):
    v = H.variance_ratio(np.diff(series["ar"]), 252)
    assert v["vr"] < 0.8, "a stationary level reverses over long horizons"


def test_vr_above_one_when_increments_persist():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(N)
    d = np.zeros(N)
    for i in range(1, N):
        d[i] = 0.15 * d[i - 1] + e[i]
    assert H.variance_ratio(d, 21)["vr"] > 1.05


def test_vr_robust_z_is_finite_and_scaled(series):
    v = H.variance_ratio(np.diff(series["rw"]), 63)
    assert np.isfinite(v["z2"]) and abs(v["z2"]) < 5


# --------------------------------------------------------------- AR order
def test_bic_picks_ar0_on_random_walk_differences(series):
    assert H.ar_select(np.diff(series["rw"]), pmax=10)["p_bic"] == 0


def test_bic_recovers_a_real_ar1_in_the_differences():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(N)
    d = np.zeros(N)
    for i in range(1, N):
        d[i] = 0.4 * d[i - 1] + e[i]
    assert H.ar_select(d, pmax=10)["p_bic"] == 1


# ------------------------------------------------------------- regressions
def test_reversion_regression_finds_a_known_half_life():
    """A level with a 60-day half-life should give a clearly negative beta."""
    rng = np.random.default_rng(3)
    phi = 0.5 ** (1 / 60)
    x = np.zeros(3000)
    for i in range(1, 3000):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    r = H.reversion_regression(pd.Series(x), 63)
    assert r["beta"] < 0 and r["t"] < -3


def test_reversion_regression_is_biased_under_the_null():
    """The whole reason the module simulates a null: under a random walk the
    t-statistic is not centred on zero, so -1.645 is the wrong critical value."""
    null = H.reversion_rw_null(n=1500, sd=1.0, k=63, reps=60, seed=4)
    assert np.median(null) < -0.2
    assert np.percentile(null, 5) < -1.645


def test_signal_sharpe_recovers_a_perfect_signal():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2010-01-01", periods=1200)
    b = pd.Series(np.cumsum(rng.standard_normal(1200)), index=idx)
    perfect = np.sign(b.diff().shift(-1))          # tomorrow's move, known today
    assert H.signal_sharpe(b, perfect)["sharpe"] > 5


def test_signal_sharpe_near_zero_for_a_coin_flip():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2010-01-01", periods=1200)
    b = pd.Series(np.cumsum(rng.standard_normal(1200)), index=idx)
    coin = pd.Series(rng.choice([-1.0, 1.0], 1200), index=idx)
    assert abs(H.signal_sharpe(b, coin)["sharpe"]) < 1.5


# ------------------------------------------------------------ power audits
def test_ar_power_sim_shows_the_blind_spot():
    """The claim the report rests on: a level that genuinely reverts still
    selects AR(0) on its differences."""
    df = H.ar_power_sim(half_lives=(90,), n=2000, reps=6, pmax=10, seed=7)
    assert df["share_bic_ar0"].iloc[0] >= 0.8


def test_vr_power_curve_has_correct_size_and_rising_power():
    df = H.vr_power_curve(sd_etas=(0.0, 0.02), n=1500, reps=40, seed=8)
    assert df["reject_rate"].iloc[0] < 0.20, "size should be near nominal"
    assert df["reject_rate"].iloc[1] > df["reject_rate"].iloc[0]
    assert df["median_sharpe"].iloc[1] > df["median_sharpe"].iloc[0]
