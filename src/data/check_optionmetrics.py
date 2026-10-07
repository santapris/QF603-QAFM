"""Phase 1b checks on the raw OptionMetrics pull (reads data/raw/optionmetrics, writes a summary table).

Checks
* every sample month-end has option quotes and a zero curve;
* no duplicate (date, optionid);
* quote sanity: crossed markets (bid > offer), negative bids, zero offers;
* strike scaling: median strike / 1000 relative to the SPX close is near 1 (catches a missed ×1000);
* each month-end has expiries bracketing 30 calendar days, both with calls and puts (preview of 2b);
* zero curve and SPX index coverage.

Run:  python -m src.data.check_optionmetrics
"""

from __future__ import annotations

import pandas as pd

from src.utils.dates import month_end_trading_days, trading_days
from src.utils.io import load_config, project_path

RAW = ("data", "raw", "optionmetrics")


def _bracketing(df: pd.DataFrame, days: int) -> bool:
    """True if there is an expiry strictly below and one at/above ``days`` calendar days with calls and puts."""
    dte = (df["exdate"] - df["date"]).dt.days
    ok = df.assign(dte=dte).groupby("exdate").filter(lambda g: set(g["cp_flag"]) >= {"C", "P"})
    dte_ok = (ok["exdate"] - ok["date"]).dt.days
    return bool(((dte_ok > 0) & (dte_ok < days)).any() and (dte_ok >= days).any())


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    days = cfg["vrp_tenor"]["iv_calendar_days"]
    month_ends = month_end_trading_days(start, end)

    index = pd.read_parquet(project_path(*RAW, "spx_index_daily.parquet")).set_index("date")
    zero = pd.read_parquet(project_path(*RAW, "zerocd_request_dates.parquet"))

    rows, problems = [], []
    for path in sorted(project_path(*RAW).glob("spx_options_*.parquet")):
        df = pd.read_parquet(path)
        year = int(path.stem[-4:])
        me = month_ends[month_ends.year == year]
        at_me = df[df["date"].isin(me)]
        missing_me = me.difference(pd.DatetimeIndex(at_me["date"].unique()))
        no_bracket = [d for d, g in at_me.groupby("date") if not _bracketing(g, days)]
        med_strike = df.groupby("date")["strike_price"].median() / 1000
        ratio = (med_strike / index["close"].reindex(med_strike.index)).median()
        rows.append(dict(
            year=year, rows=len(df), dates=df["date"].nunique(), month_ends_expected=len(me),
            month_ends_missing=len(missing_me), duplicates=int(df.duplicated(["date", "optionid"]).sum()),
            crossed=int((df["best_bid"] > df["best_offer"]).sum()), negative_bid=int((df["best_bid"] < 0).sum()),
            zero_offer=int((df["best_offer"] <= 0).sum()), iv_missing_share=round(df["impl_volatility"].isna().mean(), 3),
            am_settled_share=round(float((df["am_settlement"] == 1).mean()), 3),
            expiries_per_month_end=round(at_me.groupby("date")["exdate"].nunique().mean(), 1),
            month_ends_without_30d_bracket=len(no_bracket), median_strike_to_spot=round(float(ratio), 3)))
        problems += [f"{year}: month-end {d.date()} has no option quotes" for d in missing_me]
        problems += [f"{year}: month-end {d.date()} lacks expiries bracketing {days}d" for d in no_bracket]
        if not 0.7 < ratio < 1.3:
            problems.append(f"{year}: median strike/1000 vs SPX close = {ratio:.2f} (scaling?)")
        if rows[-1]["duplicates"]:
            problems.append(f"{year}: {rows[-1]['duplicates']} duplicate (date, optionid) rows")

    summary = pd.DataFrame(rows)
    zmiss = month_ends.difference(pd.DatetimeIndex(zero["date"].unique()))
    problems += [f"zero curve missing at month-end {d.date()}" for d in zmiss]
    rates = zero["rate"]
    if not rates.between(-1, 10).all():
        problems.append(f"zero-curve rates outside [-1, 10]%: {int((~rates.between(-1, 10)).sum())} rows")
    idx_missing = trading_days(cfg["sample"]["har_burnin_start"], end).difference(index.index)
    problems += [f"SPX index missing on {len(idx_missing)} trading days (first: {idx_missing[:5].date.tolist()})"] \
        if len(idx_missing) else []
    if (index["open"] <= 0).any() or index["open"].isna().any():
        problems.append(f"SPX index open missing/non-positive on {int((index['open'].fillna(0) <= 0).sum())} days")

    out = project_path("outputs", "tables", "optionmetrics_pull_summary.csv")
    summary.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print(f"\nZero curve: {zero['date'].nunique()} dates, rate range {rates.min():.3f}–{rates.max():.3f}; "
          f"SPX index: {len(index)} days {index.index.min().date()} → {index.index.max().date()}")
    print("\nProblems:" if problems else "\nNo problems found.")
    for p in problems:
        print("  -", p)
    return summary, problems


if __name__ == "__main__":
    main()
