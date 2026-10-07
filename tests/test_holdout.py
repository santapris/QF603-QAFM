import pandas as pd
import pytest

from src.utils.dates import month_end_trading_days
from src.utils.holdout import HoldoutViolation, guard, restrict, training_rows
from src.utils.io import load_config


@pytest.fixture
def cfg():
    return load_config()


def panel(start, end):
    idx = month_end_trading_days(start, end)
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_pre_cutoff_rows_pass(cfg):
    df = panel("2008-01-01", "2020-06-30")
    assert guard(df, "vrp_exante", k=6, cfg=cfg) is df


def test_row_after_cutoff_raises(cfg):
    # (a) a plain row dated after validate_end
    with pytest.raises(HoldoutViolation):
        guard(panel("2020-01-01", "2021-01-31"), cfg=cfg)


def test_origin_2020_12_k6_raises(cfg):
    # (b) the row is in 2020 but its target VRP_{t+6} is observed in 2021
    df = panel("2020-01-01", "2020-12-31")
    with pytest.raises(HoldoutViolation):
        guard(df, "vrp_exante", k=6, cfg=cfg)
    assert guard(df.loc[:"2020-06-30"], "vrp_exante", k=6, cfg=cfg) is not None


def test_expost_target_boundary_raises(cfg):
    # (c) ex-post target at origin 2020-11, k = 1 is observed after 2020-12
    df = panel("2020-11-01", "2020-11-30")
    with pytest.raises(HoldoutViolation):
        guard(df, "vrp_expost", k=1, cfg=cfg)
    guard(df, "vrp_exante", k=1, cfg=cfg)  # the ex-ante target at the same origin is fine


def test_final_true_passes_but_not_beyond_sample(cfg):
    # (d) final=True allows the holdout ...
    end = pd.Timestamp(cfg["oos"]["holdout_end"])
    df = panel("2020-01-01", end - pd.DateOffset(months=6))
    guard(df, "vrp_exante", k=6, final=True, cfg=cfg)
    # ... but never targets that rely on data after the sample end
    with pytest.raises(HoldoutViolation):
        guard(panel(end - pd.DateOffset(months=3), end), "vrp_exante", k=1, final=True, cfg=cfg)


def test_exact_obs_dates_override(cfg):
    df = panel("2020-11-01", "2020-11-30")
    guard(df, "vrp_expost", k=0, cfg=cfg, obs_dates=[pd.Timestamp("2020-12-30")])
    with pytest.raises(HoldoutViolation):
        guard(df, "vrp_expost", k=0, cfg=cfg, obs_dates=[pd.Timestamp("2021-01-04")])


def test_unknown_target_raises(cfg):
    with pytest.raises(ValueError):
        guard(panel("2010-01-01", "2010-12-31"), "not_a_target", k=1, cfg=cfg)


def test_restrict_then_guard(cfg):
    df = panel("2008-01-01", "2025-12-31")
    kept = restrict(df, "vrp_exante", k=3, cfg=cfg)
    assert kept.index.max() == pd.Timestamp("2020-09-30")
    guard(kept, "vrp_exante", k=3, cfg=cfg)


def test_training_rows_only_observed_targets(cfg):
    df = panel("2008-01-01", "2020-12-31")
    origin = pd.Timestamp("2015-12-31")
    rows = training_rows(df, origin, "vrp_exante", k=3, cfg=cfg)
    assert rows.index.max() == pd.Timestamp("2015-09-30")          # s + 3 <= t
    rows = training_rows(df, origin, "vrp_expost", k=1, cfg=cfg)
    assert rows.index.max() == pd.Timestamp("2015-09-30")          # s + 1 + 2 <= t
    rows = training_rows(df, origin, None, k=0, cfg=cfg)
    assert rows.index.max() == origin
