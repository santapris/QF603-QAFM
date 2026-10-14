"""Pull settled Kalshi markets and cache to data/raw_kalshi.parquet.

Usage: python scripts/pull_kalshi.py [--max-pages N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kalshi_client import get_settled_markets_df  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=None, help="cap pages for a quick smoke test")
    parser.add_argument("--out", default="data/raw_kalshi.parquet")
    parser.add_argument(
        "--include-multivariate",
        action="store_true",
        help="include Kalshi's auto-generated combo/cross-category markets "
             "(off by default -- they're overwhelmingly short-duration and "
             "swamp any date-based sample; see kalshi_client.py docstring)",
    )
    args = parser.parse_args()

    extra_params = None if args.include_multivariate else {"mve_filter": "exclude"}
    df = get_settled_markets_df(max_pages=args.max_pages, extra_params=extra_params)
    out_path = Path(__file__).resolve().parents[1] / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"pulled {len(df)} settled Kalshi markets -> {out_path}")
