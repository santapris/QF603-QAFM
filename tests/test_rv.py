import numpy as np
import pandas as pd
import pytest

from src.measures.rv import clean_bad_prints, daily_rv
from src.utils.dates import trading_days


def gbm_bars(sigma=0.2, n_days=300, overnight_share=0.2, seed=0, early_close_day=None):
    """1-minute GBM over 390 minutes plus an overnight jump; returns 5-minute bars and expected buckets."""
    rng = np.random.default_rng(seed)
    days = trading_days("2019-01-02", "2021-12-31")[:n_days]
    daily_var = sigma ** 2 / 252
    rows, logp, expected = [], np.log(100.0), {}
    for d in days:
        logp += rng.normal(0, np.sqrt(overnight_share * daily_var))           # overnight
        K = 42 if d == early_close_day else 78
        expected[d] = K
        step = np.sqrt((1 - overnight_share) * daily_var / 390)
        path = logp + np.cumsum(rng.normal(0, step, 390))
        first = logp + rng.normal(0, step * 0.1)                              # first trade just after 09:30
        for b in range(78):                                                   # trades may run past an early close
            rows.append(dict(date=d, b=b, first_price=np.exp(first if b == 0 else path[5 * b - 1]),
                             last_price=np.exp(path[5 * b + 4])))
        logp = path[5 * K - 1]
    return pd.DataFrame(rows), pd.Series(expected)


def test_rv_recovers_gbm_variance():
    sigma = 0.2
    bars, expected = gbm_bars(sigma=sigma)
    rv = daily_rv(bars, expected)
    assert rv["rv"].iloc[1:].mean() == pytest.approx(sigma ** 2 / 252, rel=0.05)
    assert rv["rv"].isna().iloc[0]                                            # no previous close
    assert (rv["n_buckets"] == 78).all() and (rv["n_empty_buckets"] == 0).all()


def test_early_close_uses_short_grid_without_padding():
    days = trading_days("2019-01-02", "2019-03-29")
    bars, expected = gbm_bars(n_days=40, early_close_day=days[10])
    rv = daily_rv(bars, expected)
    assert rv.loc[days[10], "n_buckets"] == 42 and rv.loc[days[10], "early_close"]
    # close on the early-close day is the last trade of bucket 41, not of the post-close buckets
    b41 = bars[(bars["date"] == days[10]) & (bars["b"] == 41)]["last_price"].iloc[0]
    assert rv.loc[days[10], "close"] == pytest.approx(b41)


def test_empty_bucket_carries_previous_price():
    bars, expected = gbm_bars(n_days=5)
    d = bars["date"].unique()[2]
    full = daily_rv(bars, expected)
    thinned = bars[~((bars["date"] == d) & (bars["b"] == 30))]
    rv = daily_rv(thinned, expected)
    assert rv.loc[d, "n_empty_buckets"] == 1
    # returns over buckets 29→31 are merged into one: RV changes but stays close
    assert rv.loc[d, "rv"] == pytest.approx(full.loc[d, "rv"], rel=0.5)


def test_bad_print_replaced_but_real_move_kept():
    bars = pd.DataFrame({"date": pd.Timestamp("2022-09-14"), "b": [0, 1, 2, 3],
                         "first_price": [400.0, 400.1, 400.2, 420.0], "last_price": [400.1, 415.0, 400.2, 420.5],
                         "vwap": [400.05, 400.2, 400.2, 419.0]})
    cleaned, fixed = clean_bad_prints(bars)
    assert list(fixed["b"]) == [1]                                   # 3.6% print reversed by the next trade
    assert cleaned.loc[cleaned["b"] == 1, "last_price"].iloc[0] == 400.2
    assert cleaned.loc[cleaned["b"] == 3, "last_price"].iloc[0] == 420.5   # big move with no reversal stays
