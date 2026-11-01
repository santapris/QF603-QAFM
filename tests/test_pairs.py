"""Exploratory pairs (PREREGISTRATION_PAIRS.md), on synthetic data only: fit, gate, z, timing, costs, look-ahead,
engine parity, verdicts."""
import numpy as np
import pandas as pd
import pytest

from oilrv import backtest as B
from oilrv import pairs as P
from oilrv.config import load_config
from oilrv.diagnostics import month_starts

W = 300
SPEC = {"y": "ho", "x": "cl", "seasonal_adjust": False}
NAME = "ho/cl"


def pair_frame(n=1500, beta=1.3, trend=0.01, half_life=15.0, sd=2.0, seed=0, walk=False):
    """y = 20 + beta * x + trend * t + AR(1); x is a random walk. `walk` makes y an independent random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    x = 70 + np.cumsum(rng.normal(0, 1, n))
    b = 0.5 ** (1 / half_life)
    e = rng.normal(0, sd * np.sqrt(1 - b ** 2), n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = b * u[t - 1] + e[t]
    y = 70 + np.cumsum(rng.normal(0, 1, n)) if walk else 20 + beta * x + trend * np.arange(n) + u
    return pd.DataFrame({"ho_bbl": y, "cl_bbl": x, "ho_bbl_unadj": y, "cl_bbl_unadj": x}, index=idx)


def data_for(legs, rolled=None, bo=None):
    rolled = pd.Series(False, index=legs.index) if rolled is None else rolled
    bo = pd.Series(False, index=legs.index) if bo is None else bo
    return {"specs": {NAME: SPEC}, "legs": {NAME: legs}, "rolled": {NAME: rolled}, "blackout": bo, "stores": {NAME: {}}}


def run(legs, rolled=None, **kw):
    return P.run_pair(NAME, data_for(legs, rolled), load_config(), window=W, **kw)


def fit(legs, pos=600, signal="adjusted_with_trend"):
    win = legs.iloc[pos - W:pos]
    return P.fit_pair(win, "ho", "cl", False, pos, W, signal, load_config())


def test_fit_recovers_hedge_ratio_and_trend_and_grosses_to_two():
    p = fit(pair_frame(half_life=5.0))
    assert p["beta"] == pytest.approx(1.3, abs=0.05)
    w_y, w_x = p["w_ho"], p["w_cl"]
    assert abs(w_y) + abs(w_x) == pytest.approx(2.0)
    assert w_x / w_y == pytest.approx(-p["beta"])
    assert p["trend_c"] / w_y == pytest.approx(0.01, abs=0.005)
    assert p["trend_i0"] == 600 - W and p["eg_p"] < 0.10


def test_cost_is_25_usd_per_contract():
    legs = pair_frame(seed=1)
    r = run(legs)
    t = r["trades"]
    closed = t[t["exit_reason"] != "open_at_end"]
    assert len(closed) > 3
    contracts = closed["units"] * (closed["w_ho"].abs() + closed["w_cl"].abs())
    # entry and exit, each 2 contracts per unit
    assert np.allclose(closed["gross_usd"] - closed["net_usd"], 2 * 25.0 * contracts)


def test_gate_closes_on_random_walks_and_no_entry_when_closed():
    cfg = load_config()
    rate = {}
    for label, legs in {"coint": pair_frame(half_life=5.0, seed=2), "walk": pair_frame(seed=2, walk=True)}.items():
        c = P.PairFitCache(legs, SPEC, "adjusted_with_trend", 0.10, cfg)
        ms = [s for s in month_starts(legs.index) if s >= W]
        rate[label] = np.mean([np.isfinite(c.get(s, W)["half_life"]) for s in ms])
    assert rate["coint"] > 0.75 and rate["walk"] < 0.25
    t = run(pair_frame(seed=3))["trades"]
    assert len(t) and (t["eg_p"] < 0.10).all()                     # every entry had an open gate
    closed_only = run(pair_frame(seed=3), eg_p_max=1e-12)["trades"]
    assert len(closed_only) == 0


def test_failed_gate_keeps_raw_half_life_and_blocks_engine_gate():
    p = fit(pair_frame(seed=4))
    closed = P.with_gate({**p, "eg_p": 0.5}, 0.10)
    assert np.isnan(closed["half_life"]) and closed["half_life_raw"] == p["half_life"]
    assert not B.gate_open(closed, {"hl_min": 5, "hl_max": 120})
    assert P.with_gate({**p, "eg_p": 0.5}, None)["half_life"] == p["half_life"]     # no_gate


def test_z_removes_the_trend_and_extrapolates_it():
    cfg = load_config()
    legs = pair_frame(trend=0.05, seed=5)
    pos = 600
    p = fit(legs, pos)
    d = data_for(legs)
    sim = P.run_pair(NAME, d, cfg, window=W)["sim"]
    for i in (pos - 10, pos + 50):                                 # inside the window and past its end
        s = p["w_ho"] * legs["ho_bbl"].iloc[i] + p["w_cl"] * legs["cl_bbl"].iloc[i]
        exp = (s - p["trend_c"] * (i - (pos - W)) - p["mean"]) / p["sd"]
        assert sim.z(p, i) == pytest.approx(exp)
    # in the window the detrended residual is centred; without the trend term z drifts upward out of it
    zs = np.array([sim.z(p, i) for i in range(pos, pos + 100)])
    no_trend = fit(legs, pos, "adjusted")
    z0 = np.array([sim.z(no_trend, i) for i in range(pos, pos + 100)])
    assert abs(zs.mean()) < abs(z0.mean())


def test_pnl_reconciles_to_trades_and_roll_costs():
    cfg = load_config()
    legs = pair_frame(seed=6)
    rolled = pd.Series(False, index=legs.index)
    rolled.iloc[::21] = True
    r = run(legs, rolled)
    daily, t = r["daily"], r["trades"]
    assert daily["pnl_usd"].sum() == pytest.approx(t["net_usd"].sum())
    closed = t[t["exit_reason"] != "open_at_end"]
    assert len(closed) > 3
    M = cfg["exploratory_pairs"]["usd_per_1bbl_per_unit"]
    tick = cfg["strategy"]["tick_usd"]
    for _, tr in closed.iterrows():
        s = tr["w_ho"] * legs["ho_bbl"] + tr["w_cl"] * legs["cl_bbl"]
        gross = tr["side"] * tr["units"] * M * (s[tr["exit_fill_date"]] - s[tr["fill_date"]])
        assert tr["gross_usd"] == pytest.approx(gross)
        n = rolled.loc[tr["fill_date"]:tr["exit_fill_date"]].iloc[1:].sum()
        rolls = n * tr["units"] * (abs(tr["w_ho"]) * tick["ho"] + abs(tr["w_cl"]) * tick["cl"])
        entry_exit = 2 * cfg["strategy"]["cost_usd_per_bbl"] * M * tr["units"]
        assert tr["gross_usd"] - tr["net_usd"] == pytest.approx(rolls + entry_exit)


def test_fill_lag_one_day():
    r = run(pair_frame(seed=7))
    t = r["trades"].iloc[0]
    idx = r["daily"].index
    assert idx.get_loc(t["fill_date"]) == idx.get_loc(t["decision_date"]) + 1


def test_no_look_ahead():
    """Changing prices after a date leaves every P&L up to it unchanged."""
    legs = pair_frame(seed=8)
    a = run(legs)["daily"]
    cut = legs.index[1000]
    moved = legs.copy()
    later = moved.index > cut
    moved.loc[later, "ho_bbl"] += np.linspace(0, 40, later.sum())
    moved.loc[later, "ho_bbl_unadj"] = moved.loc[later, "ho_bbl"]
    b = run(moved)["daily"]
    pd.testing.assert_series_equal(a.loc[:cut, "pnl_usd"], b.loc[:cut, "pnl_usd"])
    pd.testing.assert_series_equal(a.loc[:cut, "z"], b.loc[:cut, "z"])


def test_market_signal_reads_unadjusted_but_pnl_stays_adjusted():
    legs = pair_frame(seed=9)
    rng = np.random.default_rng(0)
    jump = np.cumsum(rng.normal(0, 0.3, len(legs)) * (np.arange(len(legs)) % 21 == 0))
    legs["ho_bbl_unadj"] = legs["ho_bbl"] + jump
    r = run(legs, mode="market")
    p = r["sim"].fit
    assert p["market_signal"]
    i = len(legs) - 1
    s = p["w_ho"] * legs["ho_bbl_unadj"].iloc[i] + p["w_cl"] * legs["cl_bbl_unadj"].iloc[i]
    assert r["sim"].z(p, i) == pytest.approx((s - p[f"seas_{legs.index[i].month}"] - p["mean"]) / p["sd"])
    tr = r["trades"].iloc[0]
    adj = tr["w_ho"] * legs["ho_bbl"] + tr["w_cl"] * legs["cl_bbl"]
    if tr["exit_reason"] != "open_at_end":
        gross = tr["side"] * tr["units"] * 1000 * (adj[tr["exit_fill_date"]] - adj[tr["fill_date"]])
        assert tr["gross_usd"] == pytest.approx(gross)


def test_engine_parity_with_registered_spreadsim():
    """With no trend and the registered 1:1 fit, PairSim trades exactly like SpreadSim."""
    cfg = load_config()
    rng = np.random.default_rng(10)
    n, b = 1500, 0.5 ** (1 / 20)
    idx = pd.bdate_range("2012-01-02", periods=n)
    cl = 70 + np.cumsum(rng.normal(0, 1, n))
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = b * u[t - 1] + rng.normal(0, 2 * np.sqrt(1 - b ** 2))
    legs = pd.DataFrame({"cl_bbl": cl, "brent_bbl": cl + 4 + u}, index=idx)
    rolled, bo = pd.Series(False, index=idx), pd.Series(False, index=idx)
    rules = lambda: B.Rules(use_logit=False, window=W)
    p_hat = pd.Series(dtype=float)
    results = []
    for cls in (B.SpreadSim, P.PairSim):
        sim = cls("brent_wti", legs, B.FitCache(legs, "brent_wti", "fixed", cfg), p_hat, bo, rules(), cfg, rolled)
        for i in range(sim.first, n):
            sim.step(i)
        results.append(sim.results())
    (d0, t0), (d1, t1) = results
    assert len(t0) > 3
    pd.testing.assert_frame_equal(d0, d1)
    pd.testing.assert_frame_equal(t0, t1[t0.columns])


def test_cost_override_and_entry_grid_reach_the_engine():
    legs = pair_frame(seed=11)
    r = run(legs, entry_z=1.5, cost_bbl=0.0)
    assert r["sim"].r["entry_z"] == 1.5 and r["sim"].r["rearm_z"] == 1.5 and r["sim"].r["cost_bbl"] == 0.0
    assert r["trades"]["gross_usd"].sum() == pytest.approx(r["trades"]["net_usd"].sum())


def test_book_is_the_average_of_the_pairs():
    legs = pair_frame(seed=12)
    other = pair_frame(seed=13)
    specs = {NAME: SPEC, "rb/cl": {"y": "rb", "x": "cl", "seasonal_adjust": False}}
    d = {"specs": specs, "legs": {NAME: legs, "rb/cl": other.rename(columns=lambda c: c.replace("ho_", "rb_"))},
         "rolled": {n: pd.Series(False, index=legs.index) for n in specs}, "blackout": pd.Series(False, index=legs.index),
         "stores": {n: {} for n in specs}}
    res = P.run_book(d, load_config(), window=W)
    exp = (res["pairs"][NAME]["daily"]["pnl_usd"] + res["pairs"]["rb/cl"]["daily"]["pnl_usd"]) / 2
    pd.testing.assert_series_equal(res["book"], exp, check_names=False)
    assert res["round_trips"] == sum(P.round_trips(r["trades"]) for r in res["pairs"].values())


def test_verdict_gates():
    cfg = load_config()
    idx = pd.bdate_range("2010-02-01", "2026-10-26")
    rng = np.random.default_rng(14)
    good = pd.Series(rng.normal(400, 3000, len(idx)), index=idx)
    v = P.verdict(good, 30, {"a": 0.5, "b": 0.1}, cfg)
    assert v["PASS"]
    assert not P.verdict(good, 19, {"a": 0.5}, cfg)["G2_round_trips"]
    assert not P.verdict(good, 30, {"a": 0.5, "b": -0.1}, cfg)["G4_grid"]
    hit = good.copy()
    hit.loc["2022-06-01"] = -500000
    assert not P.verdict(hit, 30, {"a": 0.5}, cfg)["G5_stress"]
    assert not P.verdict(pd.Series(rng.normal(0, 3000, len(idx)), index=idx), 30, {"a": 0.5}, cfg)["G1_sharpe"]


def test_single_pair_needs_all_gates_bonferroni_and_a_new_pair():
    cfg = load_config()
    ok, no = {"PASS": True}, {"PASS": False}
    assert P.single_pair_evidence("rb/cl", ok, 2.7, cfg)
    assert not P.single_pair_evidence("rb/cl", ok, 2.6, cfg)
    assert not P.single_pair_evidence("rb/cl", no, 5.0, cfg)
    assert not P.single_pair_evidence("brent/cl", ok, 5.0, cfg)      # same legs as a registered spread


def test_carry_split_and_sides():
    r = run(pair_frame(trend=0.05, seed=15))
    s = P.carry_split(r["trades"], load_config())
    assert np.allclose(s["carry_usd"] + s["convergence_usd"], s["gross_usd"])
    tr = s.iloc[0]
    assert tr["carry_usd"] == pytest.approx(tr["side"] * tr["units"] * 1000 * tr["trend_c"] * tr["hold_days"])
    side = P.by_side(r["trades"])
    assert side["trades"].sum() == len(r["trades"]) and side["net"].sum() == pytest.approx(r["trades"]["net_usd"].sum())


def test_pair_legs_same_delivery_month_and_units():
    cfg = load_config()
    idx = pd.bdate_range("2020-01-02", "2020-08-31")
    months = pd.period_range("2020-02", "2021-03", freq="M")
    cl = pd.DataFrame({m: 50.0 + i for i, m in enumerate(months)}, index=idx)
    rb = pd.DataFrame({m: 2.0 + i / 100 for i, m in enumerate(months)}, index=idx)
    legs, rolled = P.pair_legs("rb", "cl", {"rb": rb, "cl": cl}, idx, cfg)
    assert set(legs.columns) == {"rb_bbl", "cl_bbl", "rb_bbl_unadj", "cl_bbl_unadj"}
    # flat curves in time: back-adjusted changes are zero, so roll gaps never reach the adjusted level
    assert legs["cl_bbl"].nunique() == 1 and legs["rb_bbl"].nunique() == 1
    assert rolled.sum() >= 5
    ratio = (legs["rb_bbl_unadj"] / 42 - 2.0) * 100
    assert np.allclose(ratio, legs["cl_bbl_unadj"] - 50.0)           # both legs always on the same month


def test_fitted_residual_reproduces_the_pair_fit():
    from oilrv.diagnostics import ar1, fitted_residual
    legs = pair_frame(trend=0.05, seed=16)
    pos = 600
    p = fit(legs, pos)
    r = fitted_residual(legs.iloc[pos - W:pos], p)
    assert ar1(r)[1] == pytest.approx(p["half_life_raw"], rel=1e-10)
    assert r.std(ddof=1) == pytest.approx(p["sd"], rel=1e-10)
    t = np.arange(W)
    assert abs(np.polyfit(t, r, 1)[0]) < 0.1 * abs(p["trend_c"])   # the fitted trend is gone
