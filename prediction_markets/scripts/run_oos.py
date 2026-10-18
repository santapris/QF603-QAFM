"""
Stage 5 end-to-end: does the Stage 2 calibration correction actually beat
trusting the raw market price, out of sample?

Usage:
  python scripts/run_oos.py --platform kalshi --split fraction --train-frac 0.7
  python scripts/run_oos.py --platform kalshi --split time --cutoff 2024-06-01

See src/models/stage5_oos.py's module docstring for an important caveat
about Kalshi sample date-diversity before trusting a --split time result.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.align import combine, harmonize_kalshi, harmonize_polymarket, load_panel  # noqa: E402
from src.models.stage5_oos import run_oos_test, split_by_fraction, split_by_time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["both", "kalshi", "polymarket"], default="kalshi")
    parser.add_argument("--split", choices=["time", "fraction"], default="fraction")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--cutoff", type=str, default=None, help="ISO date, required if --split time")
    parser.add_argument("--panel-path", default=None,
                        help="load an already common-schema panel directly (e.g. from "
                             "pull_kalshi_ladder_atm.py), skipping --platform/harmonize/combine")
    args = parser.parse_args()

    if args.panel_path:
        panel = load_panel(ROOT / args.panel_path)
    else:
        frames = []
        if args.platform in ("both", "kalshi"):
            frames.append(harmonize_kalshi(pd.read_parquet(ROOT / "data" / "raw_kalshi.parquet")))
        if args.platform in ("both", "polymarket"):
            frames.append(harmonize_polymarket(pd.read_parquet(ROOT / "data" / "raw_polymarket.parquet")))
        panel = combine(*frames)

    span = panel["close_time"].max() - panel["close_time"].min()
    print(f"{len(panel)} usable markets, close_time span = {span}")
    if span < pd.Timedelta(days=14):
        print("WARNING: sample spans less than 2 weeks -- a time-based split here is "
              "closer to a random split than a genuine across-time generalization test. "
              "See stage5_oos.py module docstring.")

    if args.split == "time":
        if not args.cutoff:
            parser.error("--cutoff is required when --split time")
        cutoff = pd.Timestamp(args.cutoff, tz="UTC")
        train, test = split_by_time(panel, cutoff)
    else:
        train, test = split_by_fraction(panel, train_frac=args.train_frac)

    result = run_oos_test(train, test)
    print(f"\nn_train={result['n_train']}  n_test={result['n_test']}")
    print(f"Brier (raw market price, test-only):      {result['brier_naive_price']:.4f}")
    print(f"Brier (logit-corrected, test-only):        {result['brier_corrected']:.4f}")
    print(f"Brier (base-rate benchmark, test-only):    {result['brier_base_rate_benchmark']:.4f}")
    print(f"Improvement vs. naive (positive = correction wins OOS): "
          f"{result['improvement_vs_naive']:+.4f}")
