"""
Reconstruct genuinely-uncertain, at-the-money (ATM) observations out of a
ladder-style Kalshi series (many threshold-strike markets settling on the
same day/event -- e.g. KXAAAGASDxx gas prices, KXUST*A Treasury yields,
KX*AD FX pairs, KX*D commodities). Most strikes on any given day are far
from the realized value and settle almost deterministically; using their
settlement-time price as `implied_prob` (as `kalshi_client.get_settled_
markets_df` does for a plain pull) makes the whole series look trivially
predictable -- not because the market is uninformative, but because the
settlement-time price has already mostly resolved.

The fix, in two separate steps (kept separate so the selection logic below
is testable without a network call):

1. **Select** (this module, pure/offline): for each day's ladder, find the
   1-2 strikes bracketing the realized value -- the ones that were actually
   close to the margin, out of what's often 15-20 strikes most of which were
   never seriously in play.
2. **Snapshot** (`kalshi_client.get_candlestick_snapshot`, network): pull an
   EARLY price (a fraction of the way through the market's open-to-close
   life, before the outcome has effectively resolved) for just those
   selected strikes, instead of the near-certain settlement-time price.

Caveat carried forward, not hidden: the strikes nearest the realized value
are also often the most thinly traded (a handful of contracts across the
whole day) -- the resulting `implied_prob` is a real but noisy bid/ask
midpoint, not a volume-weighted trade price. Selecting "nearest the FINAL
realized value" (not nearest 50% at the early snapshot time) is itself a
deliberate compromise to avoid pulling every strike's candlesticks just to
find which one was early-ATM; it does not bias the (price, outcome) pair
itself, since the early price is read independently of the later outcome,
but it does mean this is not a perfectly representative sample of "the"
ATM strike at the snapshot time -- see the module docstring in
`scripts/pull_kalshi_ladder_atm.py` for the full reasoning.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd


def select_boundary_strikes(day_markets: pd.DataFrame, max_strikes: int = 2) -> pd.DataFrame:
    """Given all settled markets for ONE ladder day (one `event_ticker`),
    sorted by `floor_strike`, return the `max_strikes` markets bracketing the
    single no/yes transition point -- the strikes that were actually close to
    the margin that day. Returns an empty frame if there's no numeric
    `floor_strike`, fewer than 2 strikes, or no single clean transition
    (e.g. a non-monotonic or all-one-result day, which isn't a well-formed
    ladder and isn't a safe case to guess a boundary for).
    """
    day = day_markets.dropna(subset=["floor_strike"]).sort_values("floor_strike").reset_index(drop=True)
    if len(day) < 2:
        return day.iloc[0:0]
    results = day["result"].str.lower()
    if results.nunique() != 2 or not set(results.unique()) <= {"yes", "no"}:
        return day.iloc[0:0]
    flips = (results != results.shift()).to_numpy().copy()  # pandas 3's CoW can hand back a read-only array
    flips[0] = False
    flip_idx = flips.nonzero()[0]
    if len(flip_idx) != 1:
        return day.iloc[0:0]  # not a single clean transition -- not a well-formed ladder day
    i = flip_idx[0]
    lo = max(0, i - max_strikes // 2)
    return day.iloc[lo: lo + max_strikes].reset_index(drop=True)


def build_atm_panel(markets: pd.DataFrame, snapshot_fn: Callable[[str, str, object, object], dict | None],
                    frac: float = 0.4, max_strikes_per_day: int = 2) -> pd.DataFrame:
    """Assemble a common-schema panel (platform, market_id, title, category,
    close_time, outcome, implied_prob, volume) from a ladder series' settled
    markets, using `select_boundary_strikes` per day and `snapshot_fn`
    (`kalshi_client.get_candlestick_snapshot`, injected so this is testable
    without a network call) for each selected strike's early price.

    Skips (does not impute) any strike `snapshot_fn` returns None for -- a
    market that genuinely never traded before the target time is missing
    data, not a zero.
    """
    rows = []
    for event_ticker, day_markets in markets.groupby("event_ticker"):
        boundary = select_boundary_strikes(day_markets, max_strikes=max_strikes_per_day)
        for _, m in boundary.iterrows():
            row = _snapshot_row(m, snapshot_fn, frac)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows, columns=_PANEL_COLUMNS)


def build_full_ladder_panel(markets: pd.DataFrame, snapshot_fn: Callable[[str, str, object, object], dict | None],
                            frac: float = 0.4) -> pd.DataFrame:
    """Assemble a common-schema panel from EVERY settled strike (no selection
    based on the realized outcome at all) -- the methodologically clean
    alternative to `build_atm_panel`.

    `build_atm_panel` selects the 1-2 strikes bracketing each day's realized
    value, which conditions the sample on the outcome: by construction,
    exactly one bracketing strike resolves YES and one resolves NO every
    single day, regardless of what the early price said. That's a real
    selection-bias problem (the econometric analogue of selecting on the
    dependent variable), not something a different `frac` or a wider
    bracket fixes. This function instead keeps every strike -- far-OTM ones
    priced near 0 or 1 early on are legitimate calibration evidence (if they
    really do resolve as priced, that's good calibration, not noise to
    filter out), not a problem the way a *settlement-time* price would be.

    Costs roughly 10x more snapshot calls than `build_atm_panel` for the
    same series (every strike instead of ~2/day) -- the price of removing
    the selection bias rather than tuning around it.
    """
    rows = []
    for _, m in markets.iterrows():
        row = _snapshot_row(m, snapshot_fn, frac)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=_PANEL_COLUMNS)


_PANEL_COLUMNS = ["platform", "market_id", "title", "category", "close_time", "outcome",
                 "implied_prob", "volume"]


def _snapshot_row(m: pd.Series, snapshot_fn: Callable[[str, str, object, object], dict | None],
                  frac: float) -> dict | None:
    snap = snapshot_fn(m["series_ticker"], m["ticker"], m["open_time"], m["close_time"], frac)
    if snap is None:
        return None
    return dict(
        platform="kalshi", market_id=m["ticker"], title=m.get("title"),
        category=m["series_ticker"], close_time=m["close_time"],
        outcome=1.0 if str(m["result"]).lower() == "yes" else 0.0,
        implied_prob=snap["mid"], volume=snap["volume_fp"],
    )
