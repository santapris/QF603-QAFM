import numpy as np
import pandas as pd
import pytest

from src.models.stage1 import control_series
from src.models.stage4 import (bridge_insample, model_specs, oos_forecasts, oos_metrics, selection,
                               validation_origins)
from src.utils.dates import month_end_trading_days, trading_days
from src.utils.io import load_config


@pytest.fixture
def cfg():
    c = load_config()
    c["stage2"]["ridge_log10_alpha"] = [-1, 2, 4]
    c["stage4"]["pca_m"] = [1, 2]
    return c


def panel(seed=0, beta=0.6):
    rng = np.random.default_rng(seed)
    idx = month_end_trading_days("2008-01-01", "2025-08-31")
    n = len(idx)
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = 0.4 * v[t - 1] + beta * x[t - 1] + 0.5 * rng.standard_normal()
    days = trading_days("2008-01-01", "2025-12-31")
    obs = [days[days.searchsorted(d) + 21] for d in idx]
    payoff = v + 0.5 * x + 0.5 * rng.standard_normal(n)          # predictor x forecasts the payoff beyond v
    return pd.DataFrame({"vrp_exante": v, "vrp_expost": payoff, "obs_date_vrp_expost": obs,
                         "x": x, "noise": noise}, index=idx)


def test_control_for_expost_is_exante():
    p = panel()
    assert control_series(p, "vrp_expost") is p["vrp_exante"]


def test_oos_forecast_has_no_lookahead(cfg):
    p = panel()
    specs = model_specs(["x"], cfg)
    t = pd.Timestamp("2017-06-30")
    base = oos_forecasts(p, "vrp_exante", 3, specs, [t], cfg)
    q = p.copy()
    q.loc[q.index > t, ["vrp_exante", "x"]] *= 100                  # everything unobservable at t
    after = oos_forecasts(q, "vrp_exante", 3, specs, [t], cfg)
    cols = [s["name"] for s in specs]
    pd.testing.assert_series_equal(base.loc[t, cols], after.loc[t, cols])


def test_bridge_uses_exact_observation_dates(cfg):
    p = panel()
    origins = validation_origins(p, "vrp_expost", 0, cfg)
    assert (p.loc[origins, "obs_date_vrp_expost"] <= pd.Timestamp(cfg["oos"]["validate_end"])).all()
    assert origins.min() > pd.Timestamp(cfg["oos"]["train_end"])
    fc = oos_forecasts(p, "vrp_expost", 0, model_specs(["x"], cfg, bridge=True), origins[:3], cfg)
    assert np.allclose(fc["naive"], p.loc[origins[:3], "vrp_exante"])


def test_metrics_and_selection(cfg):
    p = panel(beta=0.8)
    specs = model_specs(["x", "noise"], cfg)
    fc = oos_forecasts(p, "vrp_exante", 1, specs, validation_origins(p, "vrp_exante", 1, cfg), cfg)
    m = oos_metrics(fc, "ar1", 1, specs).set_index("model")
    assert m.loc["ar1+x", "oos_r2"] > 0.2 and m.loc["ar1+x", "cw_p"] < 0.01
    assert m.loc["ar1+noise", "oos_r2"] < 0.05
    assert m.loc["ridge0.1", "tuned"] and not m.loc["ar1+x", "tuned"]
    e = (fc["y"] - fc["ar1+x"]) ** 2
    assert m.loc["ar1+x", "mse"] == pytest.approx(e.dropna().mean())
    sel = selection(m.reset_index())
    assert sel.loc[0, "best_ridge"].startswith("ridge") and sel.loc[0, "best_pca"].startswith("pca")


def test_bridge_insample_detects_unbiasedness_and_predictor(cfg):
    t = bridge_insample(panel(), ["x", "noise"], cfg).set_index("spec")
    assert t.loc["vrp_exante only", "b"] == pytest.approx(1.0, abs=0.2)
    assert abs(t.loc["+ x", "t_x"]) > 4 and t.loc["+ x", "p_wald"] < 0.01
