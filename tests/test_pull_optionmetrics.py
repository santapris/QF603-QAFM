import pandas as pd

from src.data.pull_optionmetrics import BUFFER_DAYS, request_dates
from src.utils.dates import month_end_trading_days
from src.utils.io import load_config


def test_request_dates_cover_month_ends_with_buffer():
    cfg = load_config()
    dates = request_dates(cfg)
    me = month_end_trading_days(pd.Timestamp(cfg["sample"]["start"]) - pd.DateOffset(months=1), cfg["sample"]["end"])
    assert me.isin(dates).all()
    # 2020-03-31 (Tue): three trading days either side
    for d in ["2020-03-26", "2020-03-27", "2020-03-30", "2020-04-01", "2020-04-02", "2020-04-03"]:
        assert pd.Timestamp(d) in dates
    assert pd.Timestamp("2020-03-25") not in dates and pd.Timestamp("2020-04-06") not in dates
    assert dates.max() <= pd.Timestamp(cfg["sample"]["end"])
    assert dates.is_unique and dates.is_monotonic_increasing
    assert len(dates) <= len(me) * (2 * BUFFER_DAYS + 1)


def test_request_dates_skip_holidays():
    dates = request_dates(load_config())
    # Good Friday 2018-03-30 is not a trading day; month-end is 2018-03-29
    assert pd.Timestamp("2018-03-30") not in dates and pd.Timestamp("2018-03-29") in dates
