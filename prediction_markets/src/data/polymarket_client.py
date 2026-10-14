"""
Thin client for Polymarket's public data APIs:
  - Gamma API (gamma-api.polymarket.com)   -> market metadata, outcomes, closed status
  - CLOB API  (clob.polymarket.com)        -> per-token price history

Both are public / read-only for market data, no API key needed.

Docs: https://docs.polymarket.com/api-spec/gamma-openapi.yaml
      https://docs.polymarket.com/api-spec/clob-openapi.yaml

The exact pagination parameter names on the Gamma API (cursor- vs
offset-based) have shifted between doc versions -- this client tries the
documented cursor field first and falls back to offset paging. Confirm
against the live response shape before a long pull; print one raw page
and check for `next_cursor` vs needing `offset` yourself.
"""
from __future__ import annotations

import time
from typing import Iterator

import pandas as pd
import requests

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
_SESSION = requests.Session()


def iter_closed_markets(page_limit: int = 500, max_pages: int | None = None) -> Iterator[dict]:
    """Yield raw market dicts for closed (resolved) markets."""
    params = {"closed": "true", "limit": page_limit}
    cursor = None
    offset = 0
    pages = 0

    while True:
        if cursor:
            params["after_cursor"] = cursor
        else:
            params["offset"] = offset

        resp = _SESSION.get(f"{GAMMA_URL}/markets", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Gamma has returned either a bare list or {"data": [...]} depending
        # on version -- handle both.
        batch = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        if not batch:
            break
        for m in batch:
            yield m

        cursor = data.get("next_cursor") if isinstance(data, dict) else None
        pages += 1
        if max_pages and pages >= max_pages:
            break
        if cursor:
            continue
        if len(batch) < page_limit:
            break
        offset += page_limit
        time.sleep(0.1)


def get_closed_markets_df(page_limit: int = 500, max_pages: int | None = None) -> pd.DataFrame:
    rows = []
    for m in iter_closed_markets(page_limit=page_limit, max_pages=max_pages):
        outcomes = m.get("outcomes")
        outcome_prices = m.get("outcomePrices")
        token_ids = m.get("clobTokenIds")
        rows.append(
            {
                "id": m.get("id"),
                "slug": m.get("slug"),
                "question": m.get("question"),
                "category": m.get("category") or m.get("groupItemTitle"),
                "closed_time": m.get("closedTime") or m.get("endDate"),
                "outcomes": outcomes,          # e.g. '["Yes","No"]' (often a JSON string)
                "outcome_prices": outcome_prices,  # e.g. '["1","0"]' -- final settled prices
                "clob_token_ids": token_ids,   # needed to pull CLOB price history per outcome
                "volume": m.get("volume") or m.get("volumeNum"),
            }
        )
    return pd.DataFrame(rows)


def get_price_history(token_id: str, interval: str = "max", fidelity: int = 1440) -> pd.DataFrame:
    """Historical YES-token price series for one market outcome.

    fidelity is in minutes (1440 = daily). interval is a coarse window
    shortcut ("max", "1d", "1w", ...) Polymarket applies server-side.
    """
    resp = _SESSION.get(
        f"{CLOB_URL}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    history = data.get("history", [])
    df = pd.DataFrame(history)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.rename(columns={"p": "price"})[["timestamp", "price"]]
    return df
