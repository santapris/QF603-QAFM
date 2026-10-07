import numpy as np
import pandas as pd
import pytest

from src.utils.dates import (lag_business_days, lag_months, month_end_trading_days, to_month_end,
                             trading_days)


def test_month_ends_known_dates():
    me = month_end_trading_days("2008-01-01", "2025-12-31")
    assert len(me) == 18 * 12
    for d in ["2008-10-31", "2012-10-31", "2020-03-31", "2025-02-28"]:
        assert pd.Timestamp(d) in me
    # Good Friday on the last weekday of March
    assert pd.Timestamp("2018-03-29") in me and pd.Timestamp("2018-03-30") not in me
    assert pd.Timestamp("2024-03-28") in me and pd.Timestamp("2024-03-29") not in me


def test_sandy_closure_not_a_trading_day():
    days = trading_days("2012-10-26", "2012-11-02")
    assert pd.Timestamp("2012-10-29") not in days and pd.Timestamp("2012-10-30") not in days


def test_to_month_end_asof_and_staleness():
    s = pd.Series([1.0, 2.0, np.nan, 4.0],
                  index=pd.to_datetime(["2020-01-15", "2020-01-30", "2020-01-31", "2020-03-02"]))
    me = month_end_trading_days("2020-01-01", "2020-03-31")
    out = to_month_end(s, me)
    assert out.loc["2020-01-31"] == 2.0          # NaN on the month-end is skipped
    assert out.loc["2020-02-28"] == 2.0          # carried forward
    assert out.loc["2020-03-31"] == 4.0
    stale = to_month_end(s, me, max_stale_days=10)
    assert np.isnan(stale.loc["2020-02-28"])


def test_lag_business_days_skips_weekend_and_holiday():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-31", "2020-02-14"]))   # Fri, Fri
    out = lag_business_days(s, 1)
    assert list(out.index) == [pd.Timestamp("2020-02-03"), pd.Timestamp("2020-02-18")]  # 17th = Presidents Day
    # A value published on the last trading day of the month is not usable at that month-end
    me = month_end_trading_days("2020-01-01", "2020-02-29")
    assert np.isnan(to_month_end(out, me).loc["2020-01-31"])


def test_lag_months_maps_to_trading_month_end():
    s = pd.Series([1.0], index=pd.to_datetime(["2017-12-31"]))
    out = lag_months(s, 3)
    assert out.index[0] == pd.Timestamp("2018-03-29")   # Good Friday month


def test_negative_lag_rejected():
    s = pd.Series([1.0], index=pd.to_datetime(["2020-01-31"]))
    with pytest.raises(ValueError):
        lag_business_days(s, -1)
