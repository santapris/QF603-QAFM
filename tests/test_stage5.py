import numpy as np
import pandas as pd
import pytest

from src.models.stage5 import contiguous_tail, insample, oos
from src.utils.dates import month_end_trading_days
from src.utils.io import load_config


@pytest.fixture
def cfg():
    return load_config()


def returns_panel(seed=0, beta=0.5):
    rng = np.random.default_rng(seed)
    idx = month_end_trading_days("2008-01-01", "2025-08-31")
    n = len(idx)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    r1 = np.r_[beta * 0.01 * x[:-1] + 0.005 + 0.04 * rng.standard_normal(n - 1), np.nan]   # r1[t]: t → t+1
    p = pd.DataFrame({"vrp_trail": x, "noise": rng.standard_normal(n), "exret_1": r1}, index=idx)
    for k in (3, 6):
        p[f"exret_{k}"] = sum(p["exret_1"].shift(-j) for j in range(k))
    return p


def test_insample_nw_and_hodrick_agree_and_respect_holdout(cfg):
    t = insample(returns_panel(beta=1.5), ["vrp_trail"], [1, 3, 6], cfg).set_index("k")
    assert t.loc[1, "t_nw"] > 3 and t.loc[1, "t_hodrick"] > 3
    assert t.loc[1, "t_hodrick"] == pytest.approx(t.loc[1, "t_nw"], rel=0.35)
    assert pd.Timestamp(t.loc[6, "end"]) <= pd.Timestamp("2020-06-30")


def test_oos_campbell_thompson_restrictions(cfg):
    met, fc = oos(returns_panel(beta=1.5), ["vrp_trail", "noise"], [1], cfg)
    m = met.set_index(["predictor", "variant"])
    assert m.loc[("vrp_trail", "unrestricted"), "r2_os"] > 0.05
    assert m.loc[("vrp_trail", "unrestricted"), "cw_p"] < 0.05
    f = fc[fc["predictor"] == "noise"]
    neg = f["b_hat"] < 0
    assert np.allclose(f.loc[neg, "slope_restricted"], f.loc[neg, "mean"])
    assert (f["slope_and_premium"] >= 0).all()
    assert fc["date"].min() > pd.Timestamp(cfg["oos"]["train_end"])


def test_contiguous_tail():
    idx = month_end_trading_days("2010-01-01", "2010-12-31")
    df = pd.DataFrame({"y": range(12)}, index=idx).drop(idx[4])
    assert contiguous_tail(df).index[0] == idx[5]
