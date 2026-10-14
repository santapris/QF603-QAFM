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


def get_settled_markets_df(page_limit: int = 1000, max_pages: int | None = None) -> pd.DataFrame:
    """Flatten settled markets into a DataFrame with the fields we need downstream."""
    rows = []
    for m in iter_settled_markets(page_limit=page_limit, max_pages=max_pages):
        rows.append(
            {
                "ticker": m.get("ticker"),
                "series_ticker": m.get("series_ticker") or m.get("event_ticker"),
                "title": m.get("title"),
                "category": m.get("category"),
                "close_time": m.get("close_time"),
                "result": m.get("result"),  # "yes" / "no" / "scalar" / ""
                "last_price_dollars": m.get("last_price_dollars"),
                "yes_bid_dollars": m.get("yes_bid_dollars"),
                "yes_ask_dollars": m.get("yes_ask_dollars"),
                "volume_fp": m.get("volume_fp"),
            }
        )
    return pd.DataFrame(rows)


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
