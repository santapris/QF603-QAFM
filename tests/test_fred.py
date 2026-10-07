import numpy as np
import pandas as pd
import pytest

from src.data.pull_fred import tbill_holding_return
from src.utils.dates import lag_business_days, month_end_trading_days, to_month_end


def test_tbill_holding_return():
    # 5% discount rate: investment yield 365*0.05/(360-91*0.05) = 5.1344%; one 30-day month
    r = tbill_holding_return(pd.Series([5.0]), pd.Series([30]))
    y = 365 * 0.05 / (360 - 91 * 0.05)
    assert y == pytest.approx(0.051344, abs=1e-6)
    assert r.iloc[0] == pytest.approx((1 + y) ** (30 / 365) - 1, rel=1e-12)
    assert tbill_holding_return(pd.Series([0.0]), pd.Series([31])).iloc[0] == 0


def test_month_end_value_respects_publication_lag_and_staleness():
    # Value dated on the month-end trading day is only public the next trading day.
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-03-26", "2020-03-31"]))
    me = month_end_trading_days("2020-03-01", "2020-04-30")
    out = to_month_end(lag_business_days(s, 1), me, max_stale_days=14)
    assert out.loc["2020-03-31"] == 1.0          # 03-26 value, public 03-27
    assert np.isnan(out.loc["2020-04-30"])       # latest public value (04-01) is 29 days old
