import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.measures.mfiv import interpolate_30d, mfiv_on_date, minutes_to_expiry, otm_tail, term_variance


def bs(S, K, T, r, sigma, cp):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def chain(date, exdate, S, r, sigma, am=0, step=1.0, half_spread=0.0):
    T = minutes_to_expiry(pd.Timestamp(date), pd.Series([pd.Timestamp(exdate)]), pd.Series([am])).iloc[0] / 525_600
    rows = []
    for K in np.arange(0.4 * S, 2.0 * S + step, step):
        for cp in "CP":
            p = bs(S, K, T, r, sigma, cp)
            bid = max(round(p - half_spread, 2), 0.0)
            rows.append(dict(date=pd.Timestamp(date), exdate=pd.Timestamp(exdate), am_settlement=am, cp_flag=cp,
                             strike_price=K * 1000, best_bid=bid, best_offer=max(p + half_spread, 0.01)))
    return pd.DataFrame(rows), T


def test_minutes_to_expiry_am_pm_and_saturday():
    d = pd.Timestamp("2008-01-31")
    ex = pd.Series(pd.to_datetime(["2008-02-16", "2008-02-15", "2008-02-29"]))   # Sat (AM), Fri (AM), Fri (PM)
    m = minutes_to_expiry(d, ex, pd.Series([1, 1, 0]))
    assert list(m) == [15 * 1440 - 390, 15 * 1440 - 390, 29 * 1440]


def test_term_variance_recovers_flat_vol():
    S, r, sigma = 1000.0, 0.02, 0.20
    q, T = chain("2020-01-31", "2020-02-28", S, r, sigma)
    q["strike"] = q["strike_price"] / 1000
    q["mid"] = (q["best_bid"] + q["best_offer"]) / 2
    res = term_variance(q, T, r)
    assert res["sigma2"] == pytest.approx(sigma ** 2, rel=0.01)
    assert res["F"] == pytest.approx(S * np.exp(r * T), rel=1e-3)


def test_mfiv_on_date_flat_vol_interpolated():
    S, r, sigma = 3000.0, 0.01, 0.25
    q1, _ = chain("2020-01-31", "2020-02-21", S, r, sigma, step=5)
    q2, _ = chain("2020-01-31", "2020-03-06", S, r, sigma, step=5)
    zero = pd.DataFrame({"days": [7, 30, 60, 365], "rate": [r * 100] * 4})
    res = mfiv_on_date(pd.concat([q1, q2]), zero)
    assert res["method"] == "interpolate"
    assert res["T1_days"] < 30 < res["T2_days"]
    assert res["iv"] == pytest.approx(sigma ** 2, rel=0.01)


def test_interpolation_is_exact_for_flat_variance():
    assert interpolate_30d(20 / 365, 0.04, 40 / 365, 0.04, 30) == pytest.approx(0.04 * 365 / 365, rel=1e-12)


def test_otm_tail_skips_single_zero_and_stops_after_two():
    side = pd.DataFrame({"best_bid": [1.0, 0.5, 0.0, 0.2, 0.0, 0.0, 0.3], "mid": [1.1, 0.6, 0.05, 0.3, 0.05, 0.05, 0.4]},
                        index=[101.0, 102, 103, 104, 105, 106, 107])
    out = otm_tail(side, side.index)
    assert [k for k, _ in out] == [101, 102, 104]      # 103 skipped; stop at 105–106; 107 never reached
