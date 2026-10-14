"""
Harmonize Kalshi and Polymarket raw pulls into one common schema so the
measurement/modelling code never has to know which platform a row came from.

Common schema (one row per resolved market):
    platform        "kalshi" | "polymarket"
    market_id       platform-native id/ticker (string)
    title           question text
    category        best-effort category label (NOT guaranteed comparable
                     across platforms -- see note in README)
    close_time      UTC timestamp the market resolved
    outcome         1.0 if YES happened, 0.0 if NO, NaN if scalar/ambiguous
    implied_prob    market price just before close, rescaled to [0, 1]
    volume          platform-native volume (NOT comparable in level across
                     platforms -- Kalshi counts contracts, Polymarket
                     reports notional-ish volume. Use within-platform only,
                     or a rank/percentile, until this is reconciled)
    platform_fee_bps  rough placeholder for cost-of-trading, fill in before
                     the strategy/backtest stage -- see README

Why keep `platform` as its own column rather than just pooling everything:
Kalshi is CFTC-regulated and settles on its own stated rules; Polymarket
resolves via a UMA oracle with a dispute window. Fee structure, typical
liquidity, and category mix also differ. None of that invalidates pooling
them into one calibration dataset, but every regression stage should
control for `platform` (or run the analysis per-platform first and check
the two agree) rather than assume the two series are interchangeable draws
from the same process.
"""
from __future__ import annotations

import pandas as pd


def harmonize_kalshi(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["platform"] = "kalshi"
    out["market_id"] = df["ticker"]
    out["title"] = df["title"]
    out["category"] = df["category"]
    out["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")

    result = df["result"].str.lower()
    out["outcome"] = result.map({"yes": 1.0, "no": 0.0}).astype(float)

    # last_price_dollars is already a probability-like price in [0, 1]
    # (Kalshi contracts settle between $0 and $1).
    out["implied_prob"] = pd.to_numeric(df["last_price_dollars"], errors="coerce")
    out["volume"] = pd.to_numeric(df["volume_fp"], errors="coerce")
    return out


def harmonize_polymarket(df: pd.DataFrame) -> pd.DataFrame:
    import json

    out = pd.DataFrame(index=df.index)
    out["platform"] = "polymarket"
    out["market_id"] = df["id"]
    out["title"] = df["question"]
    out["category"] = df["category"]
    out["close_time"] = pd.to_datetime(df["closed_time"], utc=True, errors="coerce")

    def _final_yes_prob(row) -> float:
        # outcomes / outcome_prices often arrive as JSON-encoded strings,
        # e.g. outcomes='["Yes","No"]', outcome_prices='["1","0"]'.
        outcomes, prices = row["outcomes"], row["outcome_prices"]
        try:
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(prices, str):
                prices = json.loads(prices)
            idx = [o.lower() for o in outcomes].index("yes")
            return float(prices[idx])
        except (ValueError, TypeError, KeyError, IndexError):
            return float("nan")

    out["outcome"] = df.apply(_final_yes_prob, axis=1)  # settled price IS the outcome (0 or 1)
    out["implied_prob"] = float("nan")  # fill in from get_price_history() a few days pre-close
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return out


def combine(*frames: pd.DataFrame) -> pd.DataFrame:
    """Stack harmonized platform frames and drop rows we can't use.

    Markets that didn't resolve cleanly to 0/1 (scalar markets, disputed
    Polymarket markets, early-close edge cases) are dropped here rather
    than silently coerced -- check `n_dropped` before trusting downstream
    stats on a small pull.
    """
    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)
    combined = combined.dropna(subset=["outcome", "implied_prob"])
    combined = combined[combined["outcome"].isin([0.0, 1.0])]
    n_dropped = n_before - len(combined)
    if n_dropped:
        print(f"[align.combine] dropped {n_dropped}/{n_before} rows "
              f"(missing price/outcome, or non-binary settlement)")
    return combined.reset_index(drop=True)
