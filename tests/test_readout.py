"""Part 7 tests: point-in-time volatility percentile, switch-on trigger logic, the watch list."""
import numpy as np
import pandas as pd
import pytest

from oilrv import readout as RO
from oilrv.config import load_config


def test_expanding_pct_is_point_in_time():
    x = pd.Series([1.0, 3.0, 2.0, np.nan, 5.0])
    p = RO.expanding_pct(x)
    assert p.tolist()[:3] == [1.0, 1.0, pytest.approx(2 / 3)]
    assert np.isnan(p.iloc[3]) and p.iloc[4] == 1.0


def test_garch_path_ignores_the_future():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=900)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 900)), index=idx, name="d")
    a = RO.garch_path(x, 300)
    y = x.copy()
    cut = idx[600]
    y[y.index > cut] += rng.normal(0, 10, (y.index > cut).sum())
    b = RO.garch_path(y, 300)
    pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])
    assert a.loc[:idx[299]].isna().all() and a.iloc[-1] > 0


def test_triggers():
    cfg = load_config()
    ok = RO.triggers("ho_rb", 0.7, 0.5, 40, 2.5, cfg=cfg)
    assert ok["all_met"]
    assert not RO.triggers("ho_rb", 0.7, 0.85, 40, 2.5, cfg=cfg)["all_met"]       # volatility still high
    assert not RO.triggers("ho_rb", 0.7, 0.5, 27, 2.5, cfg=cfg)["all_met"]        # closure within 4 weeks
    assert not RO.triggers("crack321", np.nan, 0.5, 40, 2.5, cfg=cfg)["all_met"]  # no P yet
    assert not RO.triggers("brent_wti", 0.7, 0.1, 40, 2.5, source="spot", cfg=cfg)["all_met"]
    assert RO.triggers("brent_wti", 0.7, 0.1, 0, -2.5, source="futures", cfg=cfg)["all_met"]


def test_wpsr_shifts_after_midweek_holiday():
    d = RO.wpsr_wednesdays(pd.Timestamp("2026-10-24"), 4)
    assert d == [pd.Timestamp(x) for x in ["2026-10-28", "2026-11-04", "2026-11-12", "2026-11-18"]]  # Veterans Day


def test_watch_list_window_and_order():
    ev = pd.DataFrame({"date": pd.to_datetime(["2026-10-04", "2026-11-01", "2027-06-01"]), "type": "OPEC",
                       "name": ["past", "next", "far"], "source": "x"})
    wl = RO.watch_list(ev, pd.Timestamp("2026-10-24"), [{"date": "2026-11-10", "item": "STEO", "source": "plan"}])
    assert wl["date"].is_monotonic_increasing
    assert "OPEC: next" in wl["item"].tolist() and "STEO" in wl["item"].tolist()
    assert not wl["item"].str.contains("past|far").any()
