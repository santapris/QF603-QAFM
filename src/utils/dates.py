"""NYSE trading calendar, month-end sampling and publication lags.

Conventions
-----------
* A "month-end" is the last NYSE trading day of the calendar month.
* A daily series is sampled at month-end as the last observation on or before that day.
* Publication lags [F4] move each observation's index forward to the date it became available,
  *before* month-end sampling, so a value is only used once it was public.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=32)
def _trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    days = mcal.get_calendar("NYSE").valid_days(start_date=start, end_date=end)
    return pd.DatetimeIndex(days.tz_localize(None).normalize())


def trading_days(start, end) -> pd.DatetimeIndex:
    """NYSE trading days in [start, end]."""
    return _trading_days(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())


def month_end_trading_days(start, end) -> pd.DatetimeIndex:
    """Last NYSE trading day of every calendar month touching [start, end]."""
    first = pd.Timestamp(start).to_period("M").start_time
    last = pd.Timestamp(end).to_period("M").end_time.normalize()
    days = trading_days(first, last)
    last_per_month = pd.Series(days, index=days).groupby(days.to_period("M")).max()
    return pd.DatetimeIndex(last_per_month.values, name="date")


def to_month_end(series: pd.Series, month_ends: pd.DatetimeIndex,
                 max_stale_days: int | None = None) -> pd.Series:
    """Sample a daily series at month-ends: last non-missing value on or before each month-end.

    If ``max_stale_days`` is given, a month-end whose latest observation is older than that many
    calendar days is set to NaN (so gaps are visible, not silently filled).
    """
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    month_ends = pd.DatetimeIndex(month_ends)
    pos = s.index.searchsorted(month_ends, side="right") - 1
    valid = pos >= 0
    out = np.full(len(month_ends), np.nan)
    out[valid] = s.to_numpy(dtype=float)[pos[valid]]
    if max_stale_days is not None:
        obs_dates = pd.DatetimeIndex(np.where(valid, s.index[np.clip(pos, 0, None)], pd.NaT))
        stale = (month_ends - obs_dates).days > max_stale_days
        out[np.asarray(stale)] = np.nan
    return pd.Series(out, index=pd.DatetimeIndex(month_ends, name="date"), name=series.name)


def lag_business_days(series: pd.Series, n: int) -> pd.Series:
    """Re-date each observation to the n-th NYSE trading day after it (value becomes public then).

    Observations on non-trading days are first moved to the next trading day.
    """
    if n == 0:
        return series.copy()
    if n < 0:
        raise ValueError("publication lag must be non-negative")
    s = series.sort_index()
    idx = pd.DatetimeIndex(s.index).normalize()
    cal = trading_days(idx.min(), idx.max() + pd.Timedelta(days=10 + 3 * n))
    pos = cal.searchsorted(idx, side="left") + n
    out = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(cal[pos], name=s.index.name), name=s.name)
    return out[~out.index.duplicated(keep="last")]


def lag_months(series: pd.Series, n: int) -> pd.Series:
    """Re-date a monthly series so the value for month m is available at the month-end trading day
    of month m + n."""
    if n < 0:
        raise ValueError("publication lag must be non-negative")
    s = series.sort_index()
    periods = pd.DatetimeIndex(s.index).to_period("M") + n
    me = month_end_trading_days(periods.min().start_time, periods.max().end_time)
    lookup = pd.Series(me, index=me.to_period("M"))
    out = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(lookup.loc[periods].values, name="date"),
                    name=s.name)
    return out[~out.index.duplicated(keep="last")]


def add_months(dates, k: int) -> pd.PeriodIndex:
    """Calendar month k months after each date (as a monthly Period)."""
    return pd.DatetimeIndex(dates).to_period("M") + k
