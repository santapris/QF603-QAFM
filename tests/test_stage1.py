import numpy as np
import pandas as pd
import pytest

from src.models.stage1 import adf_table, ar2, local_projections, persistence, single_predictors, zscore_train
from src.utils.dates import month_end_trading_days
from src.utils.io import load_config


@pytest.fixture
def cfg():
    return load_config()


def synthetic_panel(seed=0, rho=0.6, beta_x=0.0, lp_beta=0.0):
    rng = np.random.default_rng(seed)
    idx = month_end_trading_days("2008-01-01", "2025-08-31")
    n = len(idx)
    shock = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = rho * v[t - 1] + beta_x * x[t - 1] + lp_beta * shock[t - 1] + rng.standard_normal()
    rw = np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({"vrp_exante": v, "x": x, "rv_surprise": shock, "rw": rw}, index=idx)


def test_persistence_recovers_rho_and_respects_holdout(cfg):
    p = persistence(synthetic_panel(rho=0.6), "vrp_exante", [1, 3, 6], cfg)
    assert p.loc[1, "rho"] == pytest.approx(0.6, abs=0.12)
    assert pd.Timestamp(p.loc[6, "end"]) <= pd.Timestamp("2020-06-30")      # target at t+6 ≤ 2020-12
    assert not p["differs_from_ar1"].any()


def test_ar2_implied_coefficient(cfg):
    a = ar2(synthetic_panel(rho=0.6), "vrp_exante", [1, 3], cfg)
    assert a.loc[1, "implied_coef_on_vrp_t"] == pytest.approx(a.loc[1, "phi1"])
    assert abs(a.loc[1, "t_phi2"]) < 3


def test_local_projection_detects_one_period_effect(cfg):
    lp = local_projections(synthetic_panel(lp_beta=1.0), "vrp_exante", "rv_surprise", [0, 1, 2, 3], cfg)
    assert lp.loc[1, "beta"] == pytest.approx(1.0, abs=0.3) and lp.loc[1, "t_nw"] > 4
    assert abs(lp.loc[0, "t_nw"]) < 3


def test_single_predictors_and_holm(cfg):
    panel = synthetic_panel(beta_x=0.8, seed=2)
    panel["noise"] = np.random.default_rng(9).standard_normal(len(panel))
    t = single_predictors(panel, "vrp_exante", ["x", "noise"], [1, 3, 6], cfg, bootstrap=True)
    assert len(t) == 6 and (t["p_holm"] >= t["p_nw"]).all()
    x1 = t[(t["predictor"] == "x") & (t["k"] == 1)].iloc[0]
    assert x1["p_holm"] < 0.01 and x1["p_mbb"] < 0.05 and x1["incr_r2"] > 0


def test_zscore_uses_training_window_only(cfg):
    x = pd.Series(np.r_[np.zeros(96), np.full(116, 100.0)], index=month_end_trading_days("2008-01-01", "2025-08-31"))
    z = zscore_train(x + np.random.default_rng(0).standard_normal(len(x)), cfg["oos"]["train_end"])
    assert abs(z.loc[:"2015-12-31"].mean()) < 1e-9          # standardised on 2008–2015 only


def test_adf_flags_random_walk(cfg):
    t = adf_table(synthetic_panel(), ["vrp_exante", "rw"], cfg)
    assert t.loc["rw", "persistent"] and not t.loc["vrp_exante", "persistent"]
