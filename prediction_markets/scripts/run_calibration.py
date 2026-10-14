"""
End-to-end Stage 1 + Stage 2 run: load cached raw pulls, harmonize, compute
the calibration curve, and fit the logit.

Run pull_kalshi.py / pull_polymarket.py first. If only one platform has
been pulled, pass --platform kalshi (or polymarket) to skip the other.

Usage: python scripts/run_calibration.py [--platform both|kalshi|polymarket]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.align import combine, harmonize_kalshi, harmonize_polymarket  # noqa: E402
from src.measures.calibration import bucket_calibration, brier_score, brier_score_baseline  # noqa: E402
from src.models.stage2_logit import fit_calibration_logit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["both", "kalshi", "polymarket"], default="both")
    args = parser.parse_args()

    frames = []
    if args.platform in ("both", "kalshi"):
        raw_kalshi = pd.read_parquet(ROOT / "data" / "raw_kalshi.parquet")
        frames.append(harmonize_kalshi(raw_kalshi))
    if args.platform in ("both", "polymarket"):
        raw_poly = pd.read_parquet(ROOT / "data" / "raw_polymarket.parquet")
        frames.append(harmonize_polymarket(raw_poly))

    panel = combine(*frames)
    print(f"\n{len(panel)} usable resolved markets "
          f"({panel['platform'].value_counts().to_dict()})\n")

    print("--- calibration curve (price bucket vs realized frequency) ---")
    print(bucket_calibration(panel).to_string(index=False))

    print(f"\nBrier score (raw price):            {brier_score(panel):.4f}")
    print(f"Brier score (base-rate benchmark):  {brier_score_baseline(panel):.4f}")

    print("\n--- logit fit: outcome ~ price + price^2 ---")
    result = fit_calibration_logit(panel)
    print(result.summary())
