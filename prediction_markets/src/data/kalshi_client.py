"""
Thin client for Kalshi's public market-data REST API.

No API key is needed for the endpoints used here (GET /markets and the
candlesticks endpoint are public, read-only). An API key is only required
for placing orders, which this project never does.

Docs: https://docs.kalshi.com/api-reference/market/get-markets
      https://docs.kalshi.com/api-reference/market/get-market-candlesticks

Verify against the live docs before relying on this long-term -- Kalshi has
changed its base host and response field names (e.g. *_dollars fixed-point
fields) more than once.
"""
from __future__ import annotations

import time
from typing import Iterator

import pandas as pd
import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_SESSION = requests.Session()


def _get(path: str, params: dict | None = None) -> dict:
    resp = _SESSION.get(f"{BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def iter_settled_markets(
    page_limit: int = 1000,
    max_pages: int | None = None,
    extra_params: dict | None = None,
) -> Iterator[dict]:
    """Yield raw market dicts for every settled (resolved) market.

    Paginates via the cursor Kalshi returns. `extra_params` lets you narrow
    by e.g. series_ticker or min_settled_ts/max_settled_ts (unix seconds) --
    useful once you want a specific category or date range rather than
    pulling the whole history on every run.
    """
    params = {"status": "settled", "limit": page_limit}
    if extra_params:
        params.update(extra_params)

    cursor = None
    pages = 0
    while True:
        if cursor:
            params["cursor"] = cursor
        data = _get("/markets", params=params)
        for m in data.get("markets", []):
            yield m

        cursor = data.get("cursor") or None
        pages += 1
        if not cursor or (max_pages and pages >= max_pages):
            break
        time.sleep(0.1)  # be polite -- see rate limit docs


def get_settled_markets_df(
    page_limit: int = 1000,
    max_pages: int | None = None,
    extra_params: dict | None = None,
) -> pd.DataFrame:
    """Flatten settled markets into a DataFrame with the fields we need downstream.

    Pass extra_params={"mve_filter": "exclude"} to drop Kalshi's
    auto-generated multivariate/combinatorial "cross category" markets --
    in an unfiltered pull these vastly outnumber ordinary single-event
    markets and are almost all short-duration, which otherwise swamps any
    date-based sample with one narrow expiry window (confirmed empirically:
    an unfiltered 15k-market pull spanned about 3 hours of close_time).

    Includes `floor_strike`/`event_ticker`/`open_time` (None for markets that
    don't have a numeric strike, e.g. plain binary event markets) -- needed to
    reconstruct the near-the-money strike per day on a ladder-style series
    (many threshold strikes settling on the same day/event, e.g. the
    KXAAAGASDxx gas-price or KXUST*A Treasury-yield series) -- see
    `src/data/ladder_atm.py`.
    """
    rows = []
    for m in iter_settled_markets(page_limit=page_limit, max_pages=max_pages, extra_params=extra_params):
        rows.append(
            {
                "ticker": m.get("ticker"),
                "series_ticker": m.get("series_ticker") or m.get("event_ticker"),
                "event_ticker": m.get("event_ticker"),
                "title": m.get("title"),
                "category": m.get("category"),
                "close_time": m.get("close_time"),
                "open_time": m.get("open_time"),
                "floor_strike": m.get("floor_strike"),
                "result": m.get("result"),  # "yes" / "no" / "scalar" / ""
                "last_price_dollars": m.get("last_price_dollars"),
                "yes_bid_dollars": m.get("yes_bid_dollars"),
                "yes_ask_dollars": m.get("yes_ask_dollars"),
                "volume_fp": m.get("volume_fp"),
            }
        )
    return pd.DataFrame(rows)


def get_candlestick_snapshot(series_ticker: str, ticker: str, open_time, close_time,
                             frac: float = 0.4) -> dict | None:
    """Early-life price snapshot for one market, at `frac` of the way between
    `open_time` and `close_time` (e.g. frac=0.4 -- the point this was tuned
    against empirically shows real two-sided quotes on a thinly-traded
    near-the-money ladder strike, well before the outcome has effectively
    resolved near market close).

    Returns None if there's no candlestick at or before the target time (the
    market hadn't traded yet) -- callers should skip, not impute, those.
    Ladder strikes trade thinly (often 0-1 contracts/hour on the strikes that
    matter most, i.e. the ones nearest the money) -- the midpoint of
    yes_bid/yes_ask is a real, if noisy, implied-probability estimate, not a
    volume-weighted trade price; this is a real caveat on the resulting
    `implied_prob`, not an implementation detail to hide.
    """
    open_ts = pd.Timestamp(open_time)
    close_ts = pd.Timestamp(close_time)
    target_ts = open_ts + frac * (close_ts - open_ts)
    data = _get(
        f"/series/{series_ticker}/markets/{ticker}/candlesticks",
        params={
            "start_ts": int(open_ts.timestamp()),
            "end_ts": int(close_ts.timestamp()),
            "period_interval": 60,
        },
    )
    candles = data.get("candlesticks", [])
    if not candles:
        return None
    on_or_before = [c for c in candles if pd.Timestamp(c["end_period_ts"], unit="s", tz="UTC") <= target_ts]
    chosen = on_or_before[-1] if on_or_before else candles[0]
    bid = (chosen.get("yes_bid") or {}).get("close_dollars")
    ask = (chosen.get("yes_ask") or {}).get("close_dollars")
    if bid is None or ask is None:
        return None
    bid, ask = float(bid), float(ask)
    return dict(
        snapshot_time=pd.Timestamp(chosen["end_period_ts"], unit="s", tz="UTC"),
        yes_bid=bid, yes_ask=ask, mid=(bid + ask) / 2,
        volume_fp=float((chosen.get("volume_fp") or 0)),
    )


def get_candlesticks(
    series_ticker: str,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1440,
) -> pd.DataFrame:
    """Daily (or hourly/1-min) OHLC price history for one market.

    period_interval is in minutes: 1, 60, or 1440 (daily) are the only
    valid values Kalshi accepts.
    """
    data = _get(
        f"/series/{series_ticker}/markets/{ticker}/candlesticks",
        params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
    )
    rows = []
    for c in data.get("candlesticks", []):
        price = c.get("price", {}) or {}
        rows.append(
            {
                "ticker": ticker,
                "end_period_ts": c.get("end_period_ts"),
                "price_open": price.get("open_dollars"),
                "price_high": price.get("high_dollars"),
                "price_low": price.get("low_dollars"),
                "price_close": price.get("close_dollars"),
                "volume_fp": c.get("volume_fp"),
            }
        )
    return pd.DataFrame(rows)
