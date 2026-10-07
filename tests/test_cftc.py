import numpy as np
import pandas as pd

from src.data.pull_cftc import build_monthly, release_dates
from src.utils.io import load_config

OVERRIDES = load_config()["cftc"]["shutdown_release_overrides"]


def rel(*dates):
    return list(release_dates(pd.Series(pd.to_datetime(list(dates))), OVERRIDES).dt.strftime("%Y-%m-%d"))


def test_normal_week_released_friday():
    assert rel("2019-06-04") == ["2019-06-07"]


def test_holiday_weeks_shift_to_next_business_day():
    assert rel("2019-11-26") == ["2019-12-02"]      # Thanksgiving Thu → Monday
    assert rel("2020-12-22") == ["2020-12-28"]      # Christmas Fri → Monday
    assert rel("2013-12-31") == ["2014-01-06"]      # New Year Wed between as-of and Friday → Monday
    assert rel("2018-12-31") == ["2019-03-08"]      # inside the 2018-19 shutdown window


def test_shutdown_overrides_are_upper_bounds():
    assert rel("2013-10-01", "2013-10-29") == ["2013-11-08", "2013-11-08"]
    assert rel("2019-01-08") == ["2019-03-08"]
    assert rel("2013-11-05") == ["2013-11-08"]      # first normal report after the backlog


def test_month_end_uses_last_released_report_and_staleness():
    cfg = load_config()
    weekly = pd.DataFrame({"as_of": pd.to_datetime(["2008-12-16", "2009-06-02"]),
                           "release": pd.to_datetime(["2008-12-19", "2009-06-05"]),
                           "cftc_pos": [0.1, 0.2], "cftc_pos_lev": [0.0, 0.0], "open_interest": [1.0, 1.0]})
    m = build_monthly(weekly, cfg)
    assert m.loc["2008-12-31", "cftc_pos"] == 0.1
    assert np.isnan(m.loc["2009-01-30", "cftc_pos"])     # release 42 days old
    assert m.loc["2009-06-30", "cftc_pos"] == 0.2
    assert np.isnan(m.loc["2008-01-31", "cftc_pos"])     # nothing released yet
