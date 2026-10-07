import numpy as np
import pandas as pd
import pytest

from src.strategy import metrics
from src.strategy.backtest import run_cycle, variance_swap_pnl
from src.strategy.costs import CostModel
from src.strategy.instruments import black76, choose_expiry, settlement_value, stress_loss
from src.strategy.signals import multipliers, tiers
from src.utils.dates import trading_days
from src.utils.io import load_config


@pytest.fixture
def cfg():
    return load_config()


def contract(**kw):
    c = dict(entry=pd.Timestamp("2019-03-01"), settle=pd.Timestamp("2019-03-29"), am_settlement=0, T=28 / 365,
             r=0.02, forward=2800.0, strike=2800.0, wing_strike=2600.0,
             call_optionid=1, put_optionid=2, wing_optionid=3,
             call_mid=60.0, put_mid=60.0, wing_mid=8.0, call_best_bid=59.5, call_best_offer=60.5,
             put_best_bid=59.5, put_best_offer=60.5, wing_best_bid=7.8, wing_best_offer=8.2,
             call_delta=0.52, put_delta=-0.48, wing_delta=-0.10,
             call_impl_volatility=0.15, put_impl_volatility=0.15, wing_impl_volatility=0.22)
    c.update(kw)
    return pd.Series(c)


def test_black76_put_call_parity():
    c, _ = black76(100.0, 95.0, 0.1, 0.02, 0.2, "C")
    p, _ = black76(100.0, 95.0, 0.1, 0.02, 0.2, "P")
    assert c - p == pytest.approx(np.exp(-0.002) * (100 - 95), rel=1e-10)


def test_settlement_and_stress_loss():
    c = contract().to_dict()
    assert settlement_value(c, 2800) == 0
    assert settlement_value(c, 2500) == pytest.approx(-(300) + 100)          # wing caps the put loss
    L_down = stress_loss(c, 2800, -0.15, 2.0)
    assert L_down > 0 and stress_loss(c, 2800, -0.05, 1.0) < L_down


def test_choose_expiry_latest_before_next_month_end_prefers_pm():
    day = pd.DataFrame({"exdate": pd.to_datetime(["2019-03-15", "2019-03-15", "2019-03-29", "2019-04-05"]),
                        "am_settlement": [1, 0, 0, 0]})
    assert choose_expiry(day, pd.Timestamp("2019-03-01"), pd.Timestamp("2019-03-29")) == (pd.Timestamp("2019-03-29"), 0)
    assert choose_expiry(day, pd.Timestamp("2019-03-01"), pd.Timestamp("2019-03-28")) == (pd.Timestamp("2019-03-15"), 0)
    assert choose_expiry(day, pd.Timestamp("2019-03-10"), pd.Timestamp("2019-03-20")) is None   # < 14 days


def test_zero_cost_cycle_reconciles_and_costs_are_additive():
    c = contract()
    days = trading_days("2019-03-01", "2019-03-29")
    spx = pd.DataFrame({"open": 2800.0, "close": 2800.0}, index=days)
    quotes = {(oid, d): (mid, dl) for d in days[1:-1]
              for oid, mid, dl in [(1, 60.0, 0.52), (2, 60.0, -0.48), (3, 8.0, -0.10)]}
    gross = run_cycle(c, quotes, spx, days, CostModel.zero(), scale=1.0, band=0.0)
    # flat market, constant quotes: only the settlement changes value → P&L = premium kept = 60 + 60 − 8
    assert gross["pnl_pts"].sum() == pytest.approx(112.0)
    net = run_cycle(c, quotes, spx, days, CostModel(0.5, 1.0, 1.0), scale=1.0, band=0.0)
    entry_cost = 0.5 * (0.5 + 0.5 + 0.2) + 3 * 0.01
    hedge_cost = 1e-4 * abs(0.52 - 0.48 + 0.10) * 2800 * 2                  # open + close, no rebalancing
    assert gross["pnl_pts"].sum() - net["pnl_pts"].sum() == pytest.approx(entry_cost + hedge_cost)


def test_variance_swap_reconciles():
    v = pd.Series([0.01, -0.2, 0.005])
    assert (variance_swap_pnl(v, 2.0) == 2 * v).all()


def test_tiers_and_multipliers(cfg):
    idx = pd.date_range("2010-01-31", periods=40, freq="ME")
    z = pd.Series(np.linspace(0, 1, 40), index=idx)
    t = tiers(z, cfg)
    assert (t.iloc[:24] == 2).all() and t.iloc[-1] == 3
    mu = pd.Series(0.01, index=idx)
    mu.iloc[5] = -0.01
    df = multipliers(mu, pd.Series(0.02, index=idx), pd.Series(10.0, index=idx), pd.Series(1.0, index=idx), cfg)
    assert df["m"].iloc[5] == 0 and df["m"].iloc[0] == 1.0
    df2 = multipliers(mu, pd.Series(0.02, index=idx), pd.Series(1.0, index=idx), pd.Series(1.0, index=idx), cfg)
    assert (df2["m"] == 0).all()                                            # expected P&L < 1.5 × cost


def test_metrics_basics():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2010-01", periods=120, freq="M")
    r = pd.Series(0.01 + 0.03 * rng.standard_normal(120), index=idx)
    rf = pd.Series(0.001, index=idx)
    s = metrics.summary(r, r - rf)
    assert s["max_drawdown"] <= 0 and s["cvar99"] <= s["cvar95"]
    same = metrics.sharpe_diff_test(r - rf, r - rf)
    assert same["diff_ann"] == pytest.approx(0) and np.isnan(same["t"]) or abs(same["t"]) < 1e-6
    better = metrics.sharpe_diff_test(r - rf + 0.01, r - rf)
    assert better["diff_ann"] > 0 and better["t"] > 3
    assert metrics.ce_gain(r + 0.005, r, 3) > 0
    assert metrics.mppm(r, rf, rho=3) < 12 * (r - rf).mean()                # risk penalty
    d1 = metrics.deflated_sharpe(r - rf, 1, 0.0)["dsr"]
    d50 = metrics.deflated_sharpe(r - rf, 50, 0.01)["dsr"]
    assert d50 < d1
