import numpy as np
import pandas as pd
import pytest

from src.models.stage3 import (ex_crisis, high_vol_dummy, interactions, regime_note, rolling_betas, sup_wald,
                               supwald_null)
from src.utils.dates import month_end_trading_days
from src.utils.io import load_config


@pytest.fixture
def cfg():
    c = load_config()
    c["stage3"]["supwald_sim_reps"] = 4000
    c["stage3"]["supwald_sim_grid"] = 400
    return c


def regime_panel(seed=0, gamma=1.5, break_at=None):
    rng = np.random.default_rng(seed)
    idx = month_end_trading_days("2008-01-01", "2025-08-31")
    n = len(idx)
    vix = 15 + 8 * np.abs(rng.standard_normal(n))
    x = rng.standard_normal(n)
    hi = vix > np.percentile(vix[:96], 80)
    v = np.zeros(n)
    for t in range(1, n):
        slope = 0.3 + gamma * hi[t - 1]
        if break_at is not None and t >= break_at:
            slope += 1.5
        v[t] = 0.3 * v[t - 1] + slope * x[t - 1] + 0.5 * rng.standard_normal()
    return pd.DataFrame({"vrp_exante": v, "x": x, "vix": vix}, index=idx)


def test_supwald_null_matches_andrews_p1(cfg):
    null = supwald_null(1, 0.15, 20000, 1000, 603)
    assert np.quantile(null, 0.95) == pytest.approx(8.85, abs=0.5)      # Andrews (1993), p = 1, π0 = 0.15


def test_threshold_uses_training_window_only(cfg):
    p = regime_panel()
    p.loc["2016-01-01":, "vix"] += 100                                   # later data must not move the threshold
    _, thr = high_vol_dummy(p["vix"], cfg)
    assert thr == pytest.approx(np.percentile(p["vix"].loc[:"2015-12-31"], 80))


def test_interaction_detects_regime_slope(cfg):
    inter = interactions(regime_panel(gamma=1.5), "vrp_exante", ["x"], [1], cfg)
    d = inter[inter["kind"] == "dummy_p80"].iloc[0]
    assert d["gamma"] == pytest.approx(1.5, abs=0.4) and d["t_gamma"] > 3
    assert d["beta_stress"] == pytest.approx(1.8, abs=0.4)
    assert bool(regime_note(inter)["unstable_in_high_vix"].iloc[0])


def test_supwald_detects_break_not_stability(cfg):
    stable = regime_panel(gamma=0.0, seed=5)
    broken = regime_panel(gamma=0.0, seed=5, break_at=80)
    y1, y2 = stable["vrp_exante"].shift(-1), broken["vrp_exante"].shift(-1)
    X = lambda p: pd.DataFrame({"lag": p["vrp_exante"], "x": p["x"]}).iloc[:-1].loc[:"2020-11-30"]  # noqa: E731
    r_stable = sup_wald(y1.loc[X(stable).index], X(stable), cfg, reps=199)
    r_broken = sup_wald(y2.loc[X(broken).index], X(broken), cfg, reps=199)
    assert r_broken["p_value"] < 0.01 and r_stable["p_value"] > 0.05
    assert abs((pd.Timestamp(r_broken["break_date"]) - broken.index[79]).days) < 400


def test_rolling_betas_window(cfg):
    rb = rolling_betas(regime_panel(), "vrp_exante", ["x"], cfg)
    ar = rb[rb["spec"] == "AR(1)"]
    assert len(ar) == 155 - cfg["rolling_window_months"] + 1         # pre-holdout rows at k = 1: 2008-01 … 2020-11


def test_ex_crisis_drops_2008_09(cfg):
    out = ex_crisis(regime_panel(), "vrp_exante", ["x"], [1], cfg)
    assert pd.Timestamp(out["persistence"].loc[1, "start"]) >= pd.Timestamp("2010-01-01")


def test_supwald_bootstrap_size(cfg):
    """Wild-bootstrap sup-Wald keeps roughly nominal size where the asymptotic test over-rejects (D035)."""
    rej = 0
    for s in range(40):
        p = regime_panel(gamma=0.0, seed=300 + s)
        X = pd.DataFrame({"lag": p["vrp_exante"], "x": p["x"]}).iloc[:-1].loc[:"2020-11-30"]
        y = p["vrp_exante"].shift(-1).loc[X.index]
        rej += sup_wald(y, X, cfg, reps=99)["p_value"] < 0.05
    assert rej / 40 <= 0.15
