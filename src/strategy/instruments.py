"""Phase 6 instruments: contract selection, Black-76 pricing for stress tests, settlement values.

The traded unit (fixed across strategy variants, PLAN 6.2) is, per entry:
* short 1 ATM call + short 1 ATM put (same strike = listed strike closest to the forward, both bids > 0),
* long 1 OTM put wing with delta closest to −``strategy.wing_delta`` (strike below the straddle, bid > 0),
* delta-hedged daily with index futures (proxied by the SPX index).
Expiry rule (D041): entry on the first trading day after month-end t; the listed expiry whose settlement is the
latest on or before the next month-end t', with at least ``MIN_DAYS`` calendar days to settlement; AM-settled
contracts settle at the SPX open on the settlement day (proxy for the special opening quotation), PM-settled at
the close. Saturday exdates (pre-2015) are moved to Friday (D028). No overlapping positions: between settlement
and the next entry the capital sits in T-bills.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.measures.mfiv import clean_quotes, term_variance, zero_rate, MIN_PER_YEAR, minutes_to_expiry

MIN_DAYS = 14


def settle_date(exdate: pd.Series) -> pd.Series:
    return exdate - pd.to_timedelta((exdate.dt.weekday == 5).astype(int), unit="D")


def black76(F, K, T, r, sigma, cp: str):
    """Black-76 price and delta w.r.t. the forward (discounted), for arrays or scalars."""
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    disc = np.exp(-r * T)
    if cp == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2)), disc * norm.cdf(d1)
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1)), -disc * norm.cdf(-d1)


def choose_expiry(day: pd.DataFrame, entry: pd.Timestamp, next_month_end: pd.Timestamp) -> tuple | None:
    """(exdate, am_settlement) of the latest settlement ≤ next month-end with ≥ MIN_DAYS to go; PM preferred on ties."""
    d = day[["exdate", "am_settlement"]].drop_duplicates().copy()
    d["settle"] = settle_date(d["exdate"])
    d = d[((d["settle"] - entry).dt.days >= MIN_DAYS) & (d["settle"] <= next_month_end)]
    if d.empty:
        return None
    d = d.sort_values(["settle", "am_settlement"], ascending=[True, False])      # PM (0) after AM (1)
    last = d.iloc[-1]
    return last["exdate"], int(last["am_settlement"])


def select_contracts(day: pd.DataFrame, zero_day: pd.DataFrame, entry: pd.Timestamp, next_month_end: pd.Timestamp,
                     wing_delta: float) -> dict | None:
    """Pick the straddle strike and put wing on ``entry`` from that day's quotes (raw OptionMetrics rows)."""
    exp = choose_expiry(day, entry, next_month_end)
    if exp is None:
        return None
    exdate, am = exp
    q = clean_quotes(day[(day["exdate"] == exdate) & (day["am_settlement"] == am)])
    minutes = float(minutes_to_expiry(entry, pd.Series([exdate]), pd.Series([am])).iloc[0])
    T = minutes / MIN_PER_YEAR
    R = zero_rate(zero_day, minutes / 1440)
    tv = term_variance(q, T, R)
    if tv is None:
        return None
    F = tv["F"]
    calls = q[(q["cp_flag"] == "C") & (q["best_bid"] > 0)].set_index("strike")
    puts = q[(q["cp_flag"] == "P") & (q["best_bid"] > 0)].set_index("strike")
    both = calls.index.intersection(puts.index)
    if len(both) == 0:
        return None
    K = both[np.argmin(np.abs(both - F))]
    wings = puts[(puts.index < K) & puts["delta"].notna()]
    if wings.empty:
        return None
    Kw = (wings["delta"] + wing_delta).abs().idxmin()
    leg = lambda row, name: {f"{name}_{c}": row[c] for c in  # noqa: E731
                             ["optionid", "best_bid", "best_offer", "mid", "impl_volatility", "delta", "gamma", "vega"]}
    return dict(entry=entry, exdate=exdate, settle=settle_date(pd.Series([exdate])).iloc[0], am_settlement=am,
                T=T, r=R, forward=F, strike=float(K), wing_strike=float(Kw),
                **leg(calls.loc[K], "call"), **leg(puts.loc[K], "put"), **leg(wings.loc[Kw], "wing"))


def unit_value(F, T, r, c: dict, iv_mult: float = 1.0, spx_move: float = 0.0) -> float:
    """Value (index points) of the unit book: −call −put +wing, Black-76 at the contracts' implied vols."""
    F1 = F * (1 + spx_move)
    call, _ = black76(F1, c["strike"], T, r, c["call_impl_volatility"] * iv_mult, "C")
    put, _ = black76(F1, c["strike"], T, r, c["put_impl_volatility"] * iv_mult, "P")
    wing, _ = black76(F1, c["wing_strike"], T, r, c["wing_impl_volatility"] * iv_mult, "P")
    return float(-call - put + wing)


def stress_loss(c: dict, S: float, spx_move: float, iv_mult: float) -> float:
    """Loss (index points, > 0) of the delta-hedged unit book under an instantaneous (spx_move, iv × iv_mult)."""
    F, T, r = c["forward"], c["T"], c["r"]
    v0 = unit_value(F, T, r, c)
    v1 = unit_value(F, T, r, c, iv_mult=iv_mult, spx_move=spx_move)
    hedge = -(-c["call_delta"] - c["put_delta"] + c["wing_delta"])            # index units offsetting book delta
    return float(-(v1 - v0 + hedge * S * spx_move))


def settlement_value(c: dict, s_settle: float) -> float:
    """Intrinsic value of the unit book at settlement (index points)."""
    K, Kw = c["strike"], c["wing_strike"]
    return float(-(max(s_settle - K, 0) + max(K - s_settle, 0)) + max(Kw - s_settle, 0))
