"""Exploratory ratio mean reversion (PREREGISTRATION_RATIO.md), on synthetic data only: weights, fit, z read from market
prices while P&L uses adjusted legs, costs, look-ahead, comparisons, verdicts."""
import numpy as np
import pandas as pd
import pytest

from oilrv import ratio as R
from oilrv.config import load_config

W = 300
SPEC = {"a": {"ho": 1}, "b": {"rb": 1}, "seasonal_adjust": False}
NAME = "ho_rb"


def frame(n=1500, k=1.25, half_life=15.0, sd=0.08, seed=0, walk=False, adj_offset=0.0):
    """B is a random walk; log(A/B) = log k + AR(1) (or a random walk if `walk`). Adjusted legs equal market legs
    plus a constant `adj_offset` on A, as a roll gap would add."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    b = 70 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    phi = 0.5 ** (1 / half_life)
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = (u[t - 1] + rng.normal(0, 0.01)) if walk else phi * u[t - 1] + e[t]
    a = b * k * np.exp(u)
    return pd.DataFrame({"ho_bbl": a + adj_offset, "rb_bbl": b, "ho_bbl_unadj": a, "rb_bbl_unadj": b}, index=idx)


def data_for(legs, rolled=None, bo=None):
    rolled = pd.Series(False, index=legs.index) if rolled is None else rolled
    bo = pd.Series(False, index=legs.index) if bo is None else bo
    return {"specs": {NAME: SPEC}, "legs": {NAME: legs}, "rolled": {NAME: rolled}, "blackout": bo, "stores": {NAME: {}}}


def run(legs, **kw):
    return R.run_spread(NAME, data_for(legs), load_config(), window=W, **kw)


def fit(legs, pos=600, mode="base"):
    return R.fit_ratio(legs.iloc[pos - W:pos], SPEC, pos, mode, load_config())


# --------------------------------------------------------------------------- weights and fit
def test_neutral_weights_equal_notional_and_gross():
    a, b = {"rb": 2, "ho": 1}, {"cl": 3}
    w = R.neutral_weights(a, b, p_a=210.0, p_b=180.0, gross=2.0)
    assert sum(abs(v) for v in w.values()) == pytest.approx(2.0)
    # side A holds 2:1 of rb:ho, side B is short cl
    assert w["rb"] == pytest.approx(2 * w["ho"]) and w["cl"] < 0 < w["ho"]
    # equal notional per unit of index price: (sA * P_A) == (sB * P_B)
    s_a, s_b = w["ho"], -w["cl"] / 3
    assert s_a * 210.0 == pytest.approx(s_b * 180.0)


def test_fit_recovers_k_half_life_and_neutral_weights():
    legs = frame(half_life=10.0)
    p = fit(legs)
    assert p["k_hat"] == pytest.approx(1.25, abs=0.05)
    assert 4 < p["half_life_raw"] < 25
    w_a, w_b = p["w_ho"], p["w_rb"]
    assert abs(w_a) + abs(w_b) == pytest.approx(2.0)
    last = legs.iloc[600 - 1]
    assert w_a * last["ho_bbl_unadj"] == pytest.approx(-w_b * last["rb_bbl_unadj"])


def test_fit_uses_only_rows_before_position():
    legs = frame(seed=3)
    a = fit(legs, 700)
    changed = legs.copy()
    changed.iloc[700:, :] *= 3.0                      # corrupt the future
    b = fit(changed, 700)
    for k in ("mean", "sd", "half_life_raw", "w_ho", "w_rb", "dS_sd", "garch_sig2_next"):
        assert a[k] == pytest.approx(b[k])


# --------------------------------------------------------------------------- market signal, adjusted money
def test_z_reads_market_prices_not_adjusted_legs():
    legs = frame(seed=1)
    shifted = frame(seed=1, adj_offset=25.0)           # same market prices, adjusted A shifted by a constant
    sim1 = run(legs)["sim"]
    sim2 = run(shifted)["sim"]
    p1, p2 = sim1.fit, sim2.fit
    for i in (700, 900, 1100):
        assert sim1.z(p1, i) == pytest.approx(sim2.z(p2, i))
    a = legs["ho_bbl_unadj"].iloc[900]
    b = legs["rb_bbl_unadj"].iloc[900]
    assert sim1.x(p1, 900) == pytest.approx(np.log(a / b))


def test_pnl_uses_adjusted_legs():
    base = run(frame(seed=2))
    shifted = run(frame(seed=2, adj_offset=25.0))
    # A constant shift of the adjusted level changes no daily change, so the trades, their dates and their P&L per unit
    # are identical. Unit sizes can differ by under 1%: the GARCH optimiser reacts to 1e-15 noise in its inputs.
    assert len(base["trades"]) == len(shifted["trades"]) > 3
    for c in ("decision_date", "fill_date", "exit_fill_date"):
        assert (base["trades"][c].fillna(pd.Timestamp(0)) == shifted["trades"][c].fillna(pd.Timestamp(0))).all()
    per_unit = lambda t: (t["gross_usd"] / t["units"]).to_numpy()
    assert np.allclose(per_unit(base["trades"]), per_unit(shifted["trades"]))
    assert np.allclose(base["trades"]["units"], shifted["trades"]["units"], rtol=0.02)


def test_cost_is_25_usd_per_contract():
    r = run(frame(seed=1))
    t = r["trades"]
    closed = t[t["exit_reason"] != "open_at_end"]
    assert len(closed) > 3
    contracts = closed["units"] * (closed["w_ho"].abs() + closed["w_rb"].abs())
    assert np.allclose(closed["gross_usd"] - closed["net_usd"], 2 * 25.0 * contracts)


# --------------------------------------------------------------------------- gate and comparisons
def test_every_entry_has_half_life_inside_the_registered_band():
    """The registered gate is a band check on the AR(1) half-life, not a stationarity test: it closes on very fast and
    on very slow estimates, and a random walk can sit inside the band."""
    m = load_config()["model"]
    for legs in (frame(half_life=20.0, seed=4), frame(seed=4, walk=True)):
        t = run(legs)["trades"]
        assert len(t) > 0
        assert t["half_life"].between(m["half_life_min"], m["half_life_max"]).all()


def test_gate_flag_matches_the_band():
    m = load_config()["model"]
    sim = run(frame(half_life=20.0, seed=4))["sim"]
    p = sim.fit
    assert (m["half_life_min"] <= p["half_life"] <= m["half_life_max"]) == bool(
        run(frame(half_life=20.0, seed=4))["daily"]["gate_open"].iloc[-1])


def test_no_gate_clips_half_life_into_band():
    m = load_config()["model"]
    p = fit(frame(seed=5, walk=True), mode="no_gate")
    assert m["half_life_min"] <= p["half_life"] <= m["half_life_max"]
    assert p["half_life_raw"] == p["half_life_raw"]    # raw value kept


def test_dollar_market_reads_the_dollar_spread():
    legs = frame(seed=6)
    sim = run(legs, mode="dollar_market")["sim"]
    p = sim.fit
    i = 900
    expect = p["w_ho"] * legs["ho_bbl_unadj"].iloc[i] + p["w_rb"] * legs["rb_bbl_unadj"].iloc[i]
    assert sim.x(p, i) == pytest.approx(expect)
    assert p["mode"] == "dollar_market"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        run(frame(), mode="nope")


# --------------------------------------------------------------------------- timing and verdicts
def test_fill_is_one_close_after_decision():
    t = run(frame(seed=7))["trades"]
    closed = t[t["exit_reason"] != "open_at_end"]
    assert len(closed) > 0
    assert (closed["fill_date"] > closed["decision_date"]).all()


def test_verdict_applies_registered_gates():
    cfg = load_config()
    idx = pd.bdate_range("2010-01-04", "2026-10-26")
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(150, 1000, len(idx)), index=idx)
    v = R.verdict(good, 40, {"1.5": 1.0, "2.0": 0.8, "2.5": 0.6}, cfg)
    assert v["G2_round_trips"] and v["G4_grid"]
    assert not R.verdict(good, 10, {"1.5": 1.0}, cfg)["G2_round_trips"]
    assert not R.verdict(good, 40, {"1.5": -0.1, "2.0": 0.8}, cfg)["G4_grid"]
    crash = good.copy()
    crash.loc["2022-03-01":"2022-03-31"] -= 5000
    assert not R.verdict(crash, 40, {"2.0": 1.0}, cfg)["G5_stress"]


def test_single_spread_evidence_needs_pass_and_bonferroni():
    cfg = load_config()
    assert R.single_spread_evidence({"PASS": True}, 2.5, cfg)
    assert not R.single_spread_evidence({"PASS": True}, 2.0, cfg)
    assert not R.single_spread_evidence({"PASS": False}, 3.0, cfg)


def test_round_trips_excludes_open_at_end():
    t = pd.DataFrame({"exit_reason": ["take_profit", "stop_loss", "open_at_end"]})
    assert R.round_trips(t) == 2 and R.round_trips(pd.DataFrame()) == 0


# --------------------------------------------------------------------------- real data (skipped if absent)
def test_log_ratio_matches_clean_futures_file():
    from oilrv import futures
    try:
        f = futures.load_futures()
    except Exception:
        pytest.skip("no futures file")
    legs = R.build_data(load_config(), f, blackout=pd.Series(False, index=f.index))["legs"]["ho_rb"]
    lr = np.log(R.side_value(legs, {"ho": 1}) / R.side_value(legs, {"rb": 1}))
    ref = np.log(f["ho_rb__ho_bbl_unadj"] / f["ho_rb__rb_bbl_unadj"])
    assert np.allclose(lr, ref)


# --------------------------------------------------------------------------- slow decisions (PREREGISTRATION_RATIO_SLOW.md)
def month_starts_of(idx):
    m = idx.to_period("M")
    return set(idx[np.flatnonzero(np.r_[True, m[1:] != m[:-1]])])


def test_decision_months_none_is_the_daily_strategy():
    legs = frame(seed=8)
    a = run(legs)
    b = run(legs, decision_months=None)
    assert np.allclose(a["daily"]["pnl_usd"], b["daily"]["pnl_usd"])


def test_monthly_decisions_only_on_first_trading_day_of_a_month():
    legs = frame(seed=8, half_life=15.0)
    r = run(legs, decision_months=1)
    t = r["trades"]
    starts = month_starts_of(legs.index)
    assert len(t) >= 2
    assert t["decision_date"].isin(starts).all()                      # every entry is decided on a month start
    ex = t[t["exit_reason"] != "open_at_end"]
    by_calendar = ex["exit_reason"].isin(["time_stop", "model_break"])
    assert ex.loc[~by_calendar, "exit_decision_date"].isin(starts).all()   # take-profit and stop only on month starts


def test_quarterly_entries_only_in_jan_apr_jul_oct():
    legs = frame(seed=9, half_life=15.0, n=2200)
    t = run(legs, decision_months=3)["trades"]
    assert len(t) > 1
    assert set(pd.DatetimeIndex(t["decision_date"]).month) <= {1, 4, 7, 10}


def test_slower_decisions_trade_less_on_the_same_data():
    legs = frame(seed=10, half_life=15.0, n=2200)
    n_daily = R.round_trips(run(legs)["trades"])
    n_month = R.round_trips(run(legs, decision_months=1)["trades"])
    n_quarter = R.round_trips(run(legs, decision_months=3)["trades"])
    assert n_daily >= n_month >= n_quarter


def test_z_raw_is_available_on_non_decision_days():
    sim = run(frame(seed=11), decision_months=1)["sim"]
    p = sim.fit
    non_start = next(i for i in range(sim.first + 5, len(sim.idx)) if not sim.decide[i])
    assert np.isnan(sim.z(p, non_start)) and np.isfinite(sim.z_raw(p, non_start))
