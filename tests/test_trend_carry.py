"""Exploratory trend and carry (PREREGISTRATION_TREND_CARRY.md): timing, costs, signals, look-ahead."""
import numpy as np
import pandas as pd
import pytest

from oilrv import trend_carry as TC
from oilrv.config import load_config


def frame(n=400, seed=0, rolled_every=21):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    x = 50 + np.cumsum(rng.normal(0.05, 1.0, n))
    rolled = np.zeros(n, bool)
    rolled[::rolled_every] = True
    return pd.DataFrame({"x": x, "rolled": rolled, "M": 1000.0, "roll_usd": 10.0,
                         "m1": x, "m2": x - 0.5, "m3": x - 1.0, "m6": x + 1.0}, index=idx)


def test_fill_lag_cost_and_roll_accounting():
    cfg = load_config()
    f = frame()
    sigma = pd.Series(1.5, index=f.index)
    sig = pd.Series(1.0, index=f.index)
    sig.iloc[200:] = -1.0
    start = f.index[50]
    d = TC.simulate(f, sig, sigma, start, cfg)
    R, M = cfg["exploratory_trend_carry"]["risk_per_day_usd"], 1000.0
    u = R / (1.5 * M)
    assert d["pos"].iloc[0] == 0 and d["pos"].iloc[1] == 0        # decided at start, held from start+2
    assert d["pos"].iloc[2] == pytest.approx(u)
    flip = f.index[200]
    assert d.loc[flip, "pos"] == pytest.approx(u) and d["pos"].shift(-2).loc[flip] == pytest.approx(-u)
    c = cfg["exploratory_trend_carry"]["cost_usd_per_bbl"] * M
    assert d["cost"].iloc[1] == pytest.approx(u * c)               # entry filled at start+1
    assert d["cost"].shift(-1).loc[flip] == pytest.approx(2 * u * c)
    exp_roll = d["pos"].abs() * 10.0 * f.loc[start:, "rolled"]
    assert np.allclose(d["roll"], exp_roll)
    assert np.allclose(d["pnl"], d["pos"] * M * f["x"].diff().loc[start:] - d["cost"] - d["roll"])


def test_signals():
    f = frame()
    t = TC.trend_signal(f["x"], 10)
    assert np.isnan(t.iloc[9]) and t.iloc[10] == np.sign(f["x"].iloc[10] - f["x"].iloc[0])
    assert (TC.carry_signal(f, "m1_m2") == 1).all()                # M1 above M2: backwardation, long
    assert (TC.carry_signal(f, "m1_m6") == -1).all()


def test_position_changes_counts_entries_and_flips():
    s = pd.Series([0, 0, 1, 1, -1, -1, 0, 1, 1.5])
    assert TC.position_changes(s) == 3


def test_no_look_ahead():
    """Changing prices after a date leaves every P&L up to it unchanged."""
    cfg = load_config()
    f = frame(1200)
    window = cfg["model"]["window"]
    from oilrv.readout import garch_path
    sig = garch_path(f["x"], window)
    start = sig.first_valid_index()
    d1 = TC.simulate(f, TC.trend_signal(f["x"], 252), sig, start, cfg)
    cut = f.index[1000]
    f2 = f.copy()
    f2.loc[f2.index > cut, "x"] += np.linspace(0, 40, (f2.index > cut).sum())
    sig2 = garch_path(f2["x"], window)
    d2 = TC.simulate(f2, TC.trend_signal(f2["x"], 252), sig2, start, cfg)
    pd.testing.assert_series_equal(d1.loc[:cut, "pnl"], d2.loc[:cut, "pnl"])


def test_descriptive_helpers():
    idx = pd.bdate_range("2020-01-01", periods=10)
    pnl = pd.Series([1, 1, -3, -1, 2, 5, -1, -1, 0, 0], index=idx, dtype=float)
    dd = TC.drawdowns(pnl)
    assert len(dd) == 2
    first = dd.sort_values("peak").iloc[0]
    assert first["peak"] == idx[1] and first["trough"] == idx[3] and first["depth"] == -4
    assert first["recovered"] == idx[5] and first["days_to_trough"] == 2 and first["days_underwater"] == 4
    assert pd.isna(dd.sort_values("peak").iloc[1]["recovered"])     # the last drawdown has not recovered

    books = pd.DataFrame({"a": np.arange(600.0) % 7 - 3}, index=pd.bdate_range("2019-01-01", periods=600))
    ann = TC.annual(books)
    assert ann[("net", "a")].sum() == pytest.approx(books["a"].sum())
    assert ann["days"].sum() == 600


def test_daily_paths_and_current_book_cover_every_instrument():
    cfg = load_config()
    f = frame(300)
    sigma = pd.Series(1.0, index=f.index)
    d = TC.simulate(f, TC.trend_signal(f["x"], 20), sigma, f.index[30], cfg)
    res = {"instruments": {"trend|cl": d, "trend|crack321": d}}
    paths = TC.daily_paths(res)
    assert len(paths) == 2 * len(d) and set(paths["instrument"]) == {"cl", "crack321"}
    assert np.allclose(paths["pnl"], paths["gross"] - paths["cost"] - paths["roll"])
    now = TC.current_book(res, cfg)
    u = now.loc["trend|crack321", "target_units"]
    assert now.loc["trend|crack321", "contracts"] == f"CL {-3 * u:+.2f}, RB {2 * u:+.2f}, HO {u:+.2f}"  # 3:2:1
