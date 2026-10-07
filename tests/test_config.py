import pandas as pd

from src.utils.io import load_config


def test_config_is_consistent():
    cfg = load_config()
    s, o = cfg["sample"], cfg["oos"]
    order = [s["start"], o["train_end"], o["validate_end"], o["holdout_end"]]
    assert [pd.Timestamp(d) for d in order] == sorted(pd.Timestamp(d) for d in order)
    assert pd.Timestamp(o["holdout_end"]) == pd.Timestamp(s["end"])
    assert cfg["horizons"] == [1, 3, 6]
    assert set(cfg["target_observation_lag_months"]) >= {"vrp_exante", "vrp_expost", "excess_return"}
    assert cfg["strategy"]["option_fee_per_contract"] == {"base": 1.0, "stress": 2.0}   # D043


def test_load_config_returns_copy():
    a = load_config()
    a["horizons"].append(99)
    assert load_config()["horizons"] == [1, 3, 6]
