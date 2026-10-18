"""
Pull a genuinely-uncertain panel out of Kalshi's ladder-style series (gas
prices, Treasury yields, FX pairs, commodities -- see src/data/ladder_atm.py's
module docstring for why this is needed: their settlement-time price makes
almost every strike look trivially predictable, which isn't a property of
the market, just of reading the price too late).

Two modes, trading off statistical cleanliness against API call volume:
  --mode full   (recommended): snapshot EVERY settled strike's early price.
                No selection on the outcome at all -- the methodologically
                clean choice, at ~10x the API calls of --mode atm.
  --mode atm:   snapshot only the 1-2 strikes bracketing each day's REALIZED
                value. Far cheaper, but conditions the sample on the outcome
                (exactly one bracketing strike resolves YES, one resolves NO,
                every day, by construction) -- a real selection-bias problem,
                not just a cruder approximation. Useful only for a quick,
                cheap first look; `full` is the one to trust.

Usage:
  python scripts/pull_kalshi_ladder_atm.py --mode full --series KXAAAGASDIL KXAAAGASDTX --out data/raw_kalshi_ladder_full.parquet
  python scripts/pull_kalshi_ladder_atm.py --mode atm --series KXAAAGASDIL --max-pages 1   # quick smoke test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.kalshi_client import get_candlestick_snapshot, get_settled_markets_df  # noqa: E402
from src.data.ladder_atm import build_atm_panel, build_full_ladder_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "atm"], default="full")
    parser.add_argument("--series", nargs="+", required=True, help="one or more series_ticker values")
    parser.add_argument("--max-pages", type=int, default=None, help="cap pages per series for a smoke test")
    parser.add_argument("--frac", type=float, default=0.4,
                        help="fraction of the way between open_time and close_time for the early snapshot")
    parser.add_argument("--max-strikes-per-day", type=int, default=2, help="--mode atm only")
    parser.add_argument("--out", default="data/raw_kalshi_ladder_full.parquet")
    parser.add_argument("--sleep", type=float, default=0.1, help="politeness delay between candlestick calls")
    args = parser.parse_args()

    def snapshot_fn(series_ticker, ticker, open_time, close_time, frac):
        time.sleep(args.sleep)
        try:
            return get_candlestick_snapshot(series_ticker, ticker, open_time, close_time, frac)
        except Exception as e:
            print(f"  WARN: snapshot failed for {ticker}: {e}")
            return None

    frames = []
    for series_ticker in args.series:
        print(f"pulling settled markets for {series_ticker} ...")
        markets = get_settled_markets_df(max_pages=args.max_pages, extra_params={"series_ticker": series_ticker})
        markets["close_time"] = pd.to_datetime(markets["close_time"], utc=True, errors="coerce")
        markets["open_time"] = pd.to_datetime(markets["open_time"], utc=True, errors="coerce")
        n_days = markets["event_ticker"].nunique()
        print(f"  {len(markets)} settled markets across {n_days} days -- "
              f"pulling {'every strike' if args.mode == 'full' else 'ATM'} snapshots ...")
        if args.mode == "full":
            panel = build_full_ladder_panel(markets, snapshot_fn, frac=args.frac)
        else:
            panel = build_atm_panel(markets, snapshot_fn, frac=args.frac, max_strikes_per_day=args.max_strikes_per_day)
        print(f"  -> {len(panel)} observations ({len(panel) / max(n_days, 1):.1f}/day)")
        frames.append(panel)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"\nwrote {len(combined)} total observations -> {out_path}")
    if len(combined):
        print(f"implied_prob distribution:\n{combined['implied_prob'].describe()}")
