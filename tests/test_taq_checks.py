import pandas as pd

from src.data.check_taq import expected_buckets


def test_expected_buckets_full_and_early_close_days():
    e = expected_buckets("2019-11-27", "2019-12-31")
    assert e.loc["2019-11-27"] == 78
    assert e.loc["2019-11-29"] == 42      # day after Thanksgiving, 13:00 close
    assert e.loc["2019-12-24"] == 42      # Christmas Eve, 13:00 close
    assert pd.Timestamp("2019-11-28") not in e.index
