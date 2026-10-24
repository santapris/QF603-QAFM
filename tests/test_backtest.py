"""Part 6 tests: sizing (the plan's worked examples), the walk-forward engine on synthetic spreads,
timing, costs, accounting, filters, look-ahead, the integer book, and the gate verdicts."""
import numpy as np
import pandas as pd
import pytest

from oilrv import backtest as B
from oilrv import metrics as MT
from oilrv import sizing as SZ
from oilrv.config import load_config

W = 300


def legs_ar1(n=1500, half_life=20.0, sd=2.0, seed=0, random_walk=False):
    """Brent - WTI is an AR(1) with the given half-life (or a random walk); CL is a random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    cl = 70 + np.cumsum(rng.normal(0, 1, n))
    b = 1.0 if random_walk else 0.5 ** (1 / half_life)
    e = rng.normal(0, sd * np.sqrt(1 - b ** 2) if not random_walk else 0.4, n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = b * u[t - 1] + e[t]
    return pd.DataFrame({"cl_bbl": cl, "brent_bbl": cl + 4 + u}, index=idx)


def run(legs, rules=None, p=None, bo=None, mode="research", rolled=None):
    cfg = load_config()
    cache = B.FitCache(legs, "brent_wti", "fixed", cfg)
    p = p if p is not None else pd.Series(np.nan, index=legs.index)
    bo = bo if bo is not None else pd.Series(False, index=legs.index)
    return B.run_backtest("brent_wti", legs, cache, p, bo, rules or B.Rules(window=W), cfg, mode, rolled)


# ------------------------------------------------------------------ sizing
def test_worked_example_a():
    cfg = load_config()
    r = SZ.integer_units(sigma=0.80, abs_z=2.3, sd_w=7.0, usd_per_bbl=3000, cfg=cfg)
    assert r["vol_limit"] == pytest.approx(2.5)
    assert r["stop_limit"] == pytest.approx(40000 / (1.7 * 7 * 3000))
    assert r["units"] == 1


def test_worked_example_b_and_drawdown_rules():
    cfg = load_config()
    assert SZ.integer_units(sigma=2.50, abs_z=2.3, sd_w=7.0, usd_per_bbl=3000, cfg=cfg)["units"] == 0
    assert SZ.integer_units(sigma=0.30, abs_z=2.0, sd_w=1.0, usd_per_bbl=1000, cfg=cfg)["units"] == 3   # cap
    assert SZ.integer_units(sigma=0.30, abs_z=2.0, sd_w=1.0, usd_per_bbl=1000, size_mult=0.5, cfg=cfg)["units"] == 1
    assert [SZ.drawdown_multiplier(d, cfg) for d in (0.05, 0.10, 0.15, 0.20)] == [1.0, 0.5, 0.5, 0.0]
    assert SZ.research_units(1.5, 1000, cfg) == pytest.approx(4.0)


# ------------------------------------------------------------------ engine on synthetic spreads
def test_mean_reverting_spread_makes_money_random_walk_does_not():
    rules = B.Rules(window=W, use_blackout=False, use_logit=False)
    d, t = run(legs_ar1(n=3000), rules)
    s = MT.summarize(d, t)
    assert s["n_trades"] >= 10 and s["sharpe"] > 0.5 and s["total"] > 0
    d2, t2 = run(legs_ar1(n=3000, random_walk=True, seed=1), rules)
    s2 = MT.summarize(d2, t2)
    assert not (s2["sharpe"] > 0.5)                           # NaN (never trades) or small


def test_fill_lag_and_trade_accounting():
    d, t = run(legs_ar1(), B.Rules(window=W, use_blackout=False, use_logit=False))
    closed = t[t["exit_reason"] != "open_at_end"]
    first = closed.iloc[0]
    dates = d.index
    assert dates.get_loc(first["fill_date"]) == dates.get_loc(first["decision_date"]) + 1
    assert dates.get_loc(first["exit_fill_date"]) == dates.get_loc(first["exit_decision_date"]) + 1
    assert d.loc[first["decision_date"], "side"] == 0 and d.loc[first["fill_date"], "side"] == first["side"]
    # daily P&L and trade P&L reconcile exactly (both streams)
    assert d["pnl_usd"].sum() == pytest.approx(t["net_usd"].sum())
    assert d["pnl_bbl"].sum() == pytest.approx(t["net_bbl"].sum())


def test_cost_charged_on_every_position_change():
    legs = legs_ar1()
    base = B.Rules(window=W, use_blackout=False, use_logit=False)
    d0, t0 = run(legs, B.Rules(window=W, use_blackout=False, use_logit=False, cost_bbl=0.0))
    d1, t1 = run(legs, base)
    closed = t1[t1["exit_reason"] != "open_at_end"]
    sides = 2 * len(closed) + (len(t1) - len(closed))
    assert d1["cost_usd"].sum() == pytest.approx((0.05 * t1["units"] * 1000).sum() * 2
                                                 - (0.05 * t1.loc[t1["exit_reason"] == "open_at_end", "units"] * 1000).sum())
    assert (d0["pnl_bbl"].sum() - d1["pnl_bbl"].sum()) == pytest.approx(0.05 * sides)
    pd.testing.assert_series_equal(d0["side"], d1["side"])        # costs never change decisions


def test_research_units_fixed_at_entry():
    d, t = run(legs_ar1(), B.Rules(window=W, use_blackout=False, use_logit=False))
    tr = t.iloc[0]
    held = d.loc[tr["fill_date"]:tr.get("exit_decision_date", d.index[-1])]
    assert (held["units"].iloc[:-1] == tr["units"]).all()
    assert tr["units"] == pytest.approx(6000 / (tr["sigma_entry"] * 1000))


def test_blackout_and_logit_block_entries():
    legs = legs_ar1()
    d_free, t_free = run(legs, B.Rules(window=W, use_logit=False))
    d_bo, t_bo = run(legs, B.Rules(window=W, use_logit=False), bo=pd.Series(True, index=legs.index))
    assert len(t_free) > 0 and len(t_bo) == 0
    low = pd.Series(0.5, index=legs.index)                    # P below 0.6: no entries
    assert len(run(legs, B.Rules(window=W, use_blackout=False), p=low)[1]) == 0
    high = pd.Series(0.9, index=legs.index)                   # P above 0.6: same trades as no filter
    t_hi = run(legs, B.Rules(window=W, use_blackout=False), p=high)[1]
    t_nofilter = run(legs, B.Rules(window=W, use_blackout=False, use_logit=False))[1]
    assert t_hi["fill_date"].tolist() == t_nofilter["fill_date"].tolist()


def test_regime_break_exit():
    legs = legs_ar1()
    t0 = run(legs, B.Rules(window=W, use_blackout=False, use_logit=False))[1]
    entry = t0.iloc[0]["decision_date"]
    p = pd.Series(0.9, index=legs.index)
    p.loc[p.index > entry] = 0.3                               # P collapses the day after entry
    t = run(legs, B.Rules(window=W, use_blackout=False), p=p)[1]
    assert t.iloc[0]["exit_reason"] in ("regime_break", "take_profit", "stop_loss") and "regime_break" in t.iloc[0]["all_exit_reasons"]
    assert t.iloc[0]["hold_days"] == 1


def test_no_look_ahead():
    legs = legs_ar1()
    rules = B.Rules(window=W, use_blackout=False, use_logit=False)
    d, _ = run(legs, rules)
    cut = d.index[len(d) // 2]
    legs2 = legs.copy()
    legs2.loc[legs2.index > cut, "brent_bbl"] += 25.0
    d2, _ = run(legs2, rules)
    pd.testing.assert_frame_equal(d.loc[:cut], d2.loc[:cut])


def test_roll_cost_charged_only_in_position():
    legs = legs_ar1()
    rules = B.Rules(window=W, use_blackout=False, use_logit=False)
    d0, t0 = run(legs, rules)
    rolled = pd.Series(True, index=legs.index)
    d1, _ = run(legs, rules, rolled=rolled)
    in_pos = (d0["side"].shift(1).fillna(0) != 0)               # held over (i-1, i]
    extra = (d0["pnl_usd"] - d1["pnl_usd"])
    assert (extra[~in_pos].abs() < 1e-9).all()
    assert (extra[in_pos] > 0).all()


def test_integer_book_equity_and_halt():
    cfg = load_config()
    legs = legs_ar1()
    sims = {"brent_wti": B.SpreadSim("brent_wti", legs, B.FitCache(legs, "brent_wti", "fixed", cfg),
                                     pd.Series(np.nan, index=legs.index), pd.Series(False, index=legs.index),
                                     B.Rules(window=W, use_blackout=False, use_logit=False), cfg)}
    book, res = B.run_book(sims, cfg)
    assert book["equity"].iloc[-1] == pytest.approx(cfg["sizing"]["book_usd"] + book["pnl_usd"].sum())
    d, t = res["brent_wti"]
    assert (t["units"] == t["units"].round()).all() and (t["units"] <= 3).all()
    # a book that starts at its stop level never trades
    cfg2 = {**cfg, "sizing": {**cfg["sizing"], "drawdown_stop": -1.0}}
    sims2 = {"brent_wti": B.SpreadSim("brent_wti", legs, B.FitCache(legs, "brent_wti", "fixed", cfg2),
                                      pd.Series(np.nan, index=legs.index), pd.Series(False, index=legs.index),
                                      B.Rules(window=W, use_blackout=False, use_logit=False), cfg2)}
    book2, res2 = B.run_book(sims2, cfg2)
    assert len(res2["brent_wti"][1]) == 0 and book2["halted"].all()


# ------------------------------------------------------------------ gate verdicts
def test_gate_verdict_g5():
    cfg = load_config()
    summ = {"sharpe": 0.8, "n_trades": 25}
    sub = pd.DataFrame({"net": [1.0, 2.0, -1.0]})
    grid = {1.5: 0.2, 2.0: 0.8, 2.5: 0.1}
    good = {"ukraine": {"filtered": -1000, "unfiltered": -5000}, "hormuz": {"filtered": 0, "unfiltered": 0}}
    v = MT.gate_verdict(summ, sub, grid, good, cfg)
    assert v["PASS"]
    bad = {**good, "hormuz": {"filtered": -50000, "unfiltered": -60000}}     # beats unfiltered but < -L
    assert not MT.gate_verdict(summ, sub, grid, bad, cfg)["G5_stress"]
    worse = {**good, "ukraine": {"filtered": -2000, "unfiltered": -1000}}
    assert not MT.gate_verdict(summ, sub, grid, worse, cfg)["G5_stress"]
