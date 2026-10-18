"""
Stage 4 end-to-end: is this period's average favorite-longshot bias
predictive of next period's, and does the answer survive a proper
identify/fit/diagnose cycle rather than assuming AR(1)?

Usage:
  python scripts/run_stage4.py --platform kalshi
  python scripts/run_stage4.py --platform kalshi --max-p 2 --min-periods 18 --force-difference
  python scripts/run_stage4.py --platform kalshi --freq D --min-periods 20 \
      --kalshi-path data/raw_kalshi_trumpapprove.parquet

See src/models/stage4_persistence.py's module docstring for the full
Box-Jenkins cycle this runs, and stage5_oos.py's module docstring for the
same Kalshi sample date-diversity caveat that applies here too: if the
pull doesn't span enough distinct periods at the chosen --freq, this
script's own min-periods guard will raise ValueError rather than report
an untrustworthy number. --freq defaults to "M" (monthly); use "D" for a
genuinely daily-cadence series (e.g. a recurring daily market) instead of
forcing it into near-empty monthly buckets.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.align import combine, harmonize_kalshi, harmonize_polymarket, load_panel  # noqa: E402
from src.models.stage4_persistence import run_stage4_persistence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["both", "kalshi", "polymarket"], default="kalshi")
    parser.add_argument("--freq", default="M", help="pandas period alias, e.g. M (monthly), D (daily), W (weekly)")
    parser.add_argument("--min-markets-per-period", type=int, default=10)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--min-periods", type=int, default=24)
    parser.add_argument("--adf-alpha", type=float, default=0.05)
    parser.add_argument("--force-difference", action="store_true")
    parser.add_argument("--drop-flagged", action="store_true",
                        help="exclude periods below --min-markets-per-period instead of just flagging them")
    parser.add_argument("--kalshi-path", default="data/raw_kalshi.parquet",
                        help="which cached Kalshi pull to read -- point this at a series-filtered pull "
                             "(see pull_kalshi.py --series-ticker) for a slower/daily-cadence series")
    parser.add_argument("--polymarket-path", default="data/raw_polymarket.parquet")
    parser.add_argument("--panel-path", default=None,
                        help="load an already common-schema panel directly (e.g. from "
                             "pull_kalshi_ladder_atm.py), skipping --platform/harmonize/combine")
    args = parser.parse_args()

    if args.panel_path:
        panel = load_panel(ROOT / args.panel_path)
    else:
        frames = []
        if args.platform in ("both", "kalshi"):
            frames.append(harmonize_kalshi(pd.read_parquet(ROOT / args.kalshi_path)))
        if args.platform in ("both", "polymarket"):
            frames.append(harmonize_polymarket(pd.read_parquet(ROOT / args.polymarket_path)))
        panel = combine(*frames)

    span = panel["close_time"].max() - panel["close_time"].min()
    close_time = panel["close_time"]
    n_distinct_periods = (close_time.dt.tz_convert(None) if close_time.dt.tz is not None
                        else close_time).dt.to_period(args.freq).nunique()
    print(f"{len(panel)} usable markets, close_time span = {span}, "
          f"{n_distinct_periods} distinct period(s) at freq={args.freq!r}")
    if n_distinct_periods < args.min_periods:
        print(f"WARNING: only {n_distinct_periods} distinct period(s) at freq={args.freq!r} -- Stage 4 "
              "needs many independent observations to fit an AR model at all, let alone trust it. See "
              "stage5_oos.py's module docstring for why Kalshi's settled-markets feed tends to cluster "
              "like this, or try a coarser/finer --freq or a slower/faster-cadence --series-ticker.")

    result = run_stage4_persistence(
        panel, freq=args.freq, min_markets_per_period=args.min_markets_per_period, max_p=args.max_p,
        adf_alpha=args.adf_alpha, min_periods=args.min_periods,
        force_difference=args.force_difference, drop_flagged=args.drop_flagged,
    )

    print(f"\n{result.n_periods} period(s) of bias data at freq={result.freq!r} "
          f"({result.n_periods_flagged_small} flagged as thin, <{args.min_markets_per_period} markets)")
    print(result.periods[["period", "n_markets", "bias", "flag_small_n"]].to_string(index=False))

    def _print_cycle(name: str, cycle: dict) -> None:
        print(f"\n-- {name} --")
        print(f"ADF: stat={cycle['adf']['stat']:.3f}  p={cycle['adf']['p_value']:.4f}  "
              f"stationary={cycle['adf']['stationary']}")
        print(f"AR order selected (AIC): {cycle['ar_order']}")
        print(f"Coefficients: { {k: round(v, 4) for k, v in cycle['ar_fit']['coefs'].items()} }")
        print(f"Ljung-Box white noise (all tested lags p>0.05): "
              f"{cycle['diagnostics']['residual_white_noise']}")
        print(f"ARCH effects present: {cycle['diagnostics']['arch_effects_present']}")

    _print_cycle("Level series", result.level)
    if result.differencing_applied:
        print("\nLevel series was NOT stationary -- differenced once and re-ran the full cycle:")
        _print_cycle("Differenced series", result.differenced)
    else:
        print("\nLevel series was already stationary -- no differencing needed.")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
