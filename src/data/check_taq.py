"""Phase 1c checks on the raw SPY 5-minute TAQ bars (reads data/raw/taq, writes a summary table).

Checks
* every NYSE trading day in [har_burnin_start, sample.end] is present;
* buckets per day: 78 on full days, 42 on 13:00 early-close days (from the NYSE calendar);
* empty buckets inside the session (no qualifying trade in 5 minutes);
* price outliers: last trade more than OUTLIER_VWAP from the bucket VWAP, or a bucket-to-bucket jump
  larger than OUTLIER_JUMP;
* SPY close-to-close log returns (last trade before the close) vs SPX index close-to-close returns
  from OptionMetrics: correlation and days differing by more than RETURN_GAP (dividends ex-dates
  explain small differences).

Run:  python -m src.data.check_taq
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.utils.io import load_config, project_path

OUTLIER_VWAP = 0.005
OUTLIER_JUMP = 0.03
RETURN_GAP = 0.01


def expected_buckets(start, end) -> pd.Series:
    """Number of 5-minute buckets per NYSE session (78 on full days, fewer on early closes)."""
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    close_et = sched["market_close"].dt.tz_convert("America/New_York")
    minutes = (close_et.dt.hour * 60 + close_et.dt.minute) - (9 * 60 + 30)
    return pd.Series((minutes // 5).to_numpy(), index=pd.DatetimeIndex(sched.index).normalize(), name="expected")


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    start, end = pd.Timestamp(cfg["sample"]["har_burnin_start"]), pd.Timestamp(cfg["sample"]["end"])
    files = sorted(project_path("data", "raw", "taq").glob("spy_5min_[0-9][0-9][0-9][0-9].parquet"))
    bars = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    expected = expected_buckets(start, end)

    per_day = bars.groupby("date").agg(buckets=("b", "nunique"), b_max=("b", "max"), trades=("n", "sum"))
    per_day = per_day.join(expected, how="right")
    missing_days = per_day.index[per_day["buckets"].isna()]
    have = per_day.dropna(subset=["buckets"])
    wrong_count = have[have["buckets"] != have["expected"]]
    beyond_close = have[have["b_max"] >= have["expected"]]

    bars = bars.sort_values(["date", "b"])
    dev = (bars["last_price"] / bars["vwap"] - 1).abs()
    jump = np.log(bars["last_price"]).groupby(bars["date"]).diff().abs()
    outliers = bars.loc[(dev > OUTLIER_VWAP) | (jump > OUTLIER_JUMP), ["date", "b", "last_price", "vwap"]]

    close = bars.groupby("date")["last_price"].last()
    spx = pd.read_parquet(project_path("data", "raw", "optionmetrics", "spx_index_daily.parquet")).set_index("date")
    r = pd.concat({"spy": np.log(close).diff(), "spx": np.log(spx["close"]).diff()}, axis=1).dropna()
    gap = r[(r["spy"] - r["spx"]).abs() > RETURN_GAP]

    summary = bars.groupby(bars["date"].dt.year).agg(
        days=("date", "nunique"), trades=("n", "sum"), median_trades_per_bucket=("n", "median"))
    summary["days_expected"] = expected.groupby(expected.index.year).size()
    summary["early_close_days"] = (expected < 78).groupby(expected.index.year).sum()
    summary["days_wrong_bucket_count"] = wrong_count.groupby(wrong_count.index.year).size()
    summary["outlier_buckets"] = outliers.groupby(outliers["date"].dt.year).size()
    summary["return_gap_days"] = gap.groupby(gap.index.year).size()
    summary = summary.fillna(0).astype({c: "int64" for c in summary.columns if c != "median_trades_per_bucket"})
    summary.index.name = "year"
    summary.to_csv(project_path("outputs", "tables", "taq_pull_summary.csv"))

    with pd.option_context("display.width", 250):
        print(summary.to_string())
    print(f"\nSPY vs SPX daily log-return correlation: {r.corr().iloc[0, 1]:.4f} over {len(r)} days")
    problems = []
    if len(missing_days):
        problems.append(f"{len(missing_days)} trading days missing, e.g. {[str(d.date()) for d in missing_days[:10]]}")
    if len(wrong_count):
        problems.append(f"{len(wrong_count)} days with unexpected bucket counts, e.g. "
                        f"{wrong_count.head(10)[['buckets', 'expected']].to_dict('index')}")
    if len(beyond_close):
        problems.append(f"{len(beyond_close)} days with trades after the scheduled close")
    if len(outliers):
        problems.append(f"{len(outliers)} outlier buckets (listed in taq_outliers.csv)")
        outliers.to_csv(project_path("outputs", "tables", "taq_outliers.csv"), index=False)
    if len(gap):
        problems.append(f"{len(gap)} days where SPY and SPX returns differ by > {RETURN_GAP:.0%}: "
                        f"{[str(d.date()) for d in gap.index[:10]]}")
    print("\nFlags:" if problems else "\nNo problems found.")
    for p in problems:
        print("  -", p)
    return summary, problems


if __name__ == "__main__":
    main()
