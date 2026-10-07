"""Phase 6.3: choose the strategy's contracts each month and pull their daily history (only those optionids).

1. For each month-end t in the sample, entry = next trading day (``strategy.execution_lag_days`` = 1); pick
   the expiry and contracts on the entry day from the Phase 1b pull (``src.strategy.instruments``).
   → ``data/processed/strategy_contracts.parquet`` (one row per cycle).
2. Pull daily quotes/Greeks of the selected optionids from entry to settlement from ``optionm.opprcdYYYY``
   (a few dozen ids per year — a small query) → ``data/processed/strategy_options_daily.parquet``.
Cycles whose settlement falls after ``sample.end`` are not selected.

Run:  python -m src.data.pull_optionmetrics_daily [--force]
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.strategy.instruments import select_contracts
from src.utils.dates import month_end_trading_days, trading_days
from src.utils.io import load_config, project_path, write_parquet

RAW = ("data", "raw", "optionmetrics")
COLS = ["date", "exdate", "optionid", "am_settlement", "cp_flag", "strike_price", "best_bid", "best_offer",
        "impl_volatility", "delta", "gamma", "vega"]
DAILY_COLS = ["date", "optionid", "best_bid", "best_offer", "impl_volatility", "delta", "gamma", "vega",
              "open_interest", "volume"]


def build_contracts(cfg) -> pd.DataFrame:
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    lag = cfg["strategy"]["execution_lag_days"]
    me = month_end_trading_days(start, end)
    cal = trading_days(me.min(), end + pd.Timedelta(days=10))
    zero = dict(tuple(pd.read_parquet(project_path(*RAW, "zerocd_request_dates.parquet")).groupby("date")))
    plan = []
    for t in me:
        i = cal.get_loc(t)
        if i + lag < len(cal) and me.get_loc(t) + 1 < len(me):
            plan.append((t, cal[i + lag], me[me.get_loc(t) + 1]))
    rows, skipped = [], []
    for year in sorted({entry.year for _, entry, _ in plan}):
        df = pd.read_parquet(project_path(*RAW, f"spx_options_{year}.parquet"), columns=COLS)
        by_date = dict(tuple(df.groupby("date")))
        for t, entry, nxt in [p for p in plan if p[1].year == year]:
            if entry not in by_date:
                skipped.append((t, "no quotes on entry day"))
                continue
            c = select_contracts(by_date[entry], zero[entry], entry, nxt, cfg["strategy"]["wing_delta"])
            if c is None:
                skipped.append((t, "no eligible expiry/contracts"))
                continue
            rows.append(dict(signal_date=t, **c))
    out = pd.DataFrame(rows)
    out = out[out["settle"] <= end].reset_index(drop=True)
    if skipped:
        print("skipped cycles:", [(str(d.date()), why) for d, why in skipped])
    return out


def pull_daily(contracts: pd.DataFrame) -> pd.DataFrame:
    from src.utils.wrds_conn import connect
    legs = pd.concat([contracts[["entry", "settle", f"{leg}_optionid"]].rename(columns={f"{leg}_optionid": "optionid"})
                      for leg in ["call", "put", "wing"]])
    legs["optionid"] = legs["optionid"].astype("int64")
    frames = []
    db = connect()
    try:
        for year in range(legs["entry"].dt.year.min(), legs["settle"].dt.year.max() + 1):
            active = legs[(legs["entry"].dt.year <= year) & (legs["settle"].dt.year >= year)]
            if active.empty:
                continue
            ids = ", ".join(str(i) for i in sorted(active["optionid"].unique()))
            q = (f"select {', '.join(DAILY_COLS)} from optionm.opprcd{year} where secid = 108105 "
                 f"and optionid in ({ids}) and date between '{active['entry'].min().date()}' and '{active['settle'].max().date()}'")
            frames.append(db.raw_sql(q, date_cols=["date"]))
            print(f"  {year}: {len(frames[-1]):,} rows", flush=True)
    finally:
        db.close()
    daily = pd.concat(frames, ignore_index=True)
    daily["optionid"] = daily["optionid"].astype("int64")
    # keep only rows inside each option's own holding window
    daily = daily.merge(legs, on="optionid")
    daily = daily[(daily["date"] >= daily["entry"]) & (daily["date"] <= daily["settle"])]
    return daily.drop(columns=["entry", "settle"]).drop_duplicates(["date", "optionid"]).sort_values(["optionid", "date"])


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    cpath = project_path("data", "processed", "strategy_contracts.parquet")
    dpath = project_path("data", "processed", "strategy_options_daily.parquet")
    if dpath.exists() and cpath.exists() and not force:
        print("[skip] strategy contracts and daily data exist")
        return
    contracts = build_contracts(cfg)
    write_parquet(contracts, "data/processed/strategy_contracts.parquet")
    days = (contracts["settle"] - contracts["entry"]).dt.days
    print(f"[ok  ] strategy_contracts.parquet: {len(contracts)} cycles {contracts['entry'].min().date()} → "
          f"{contracts['settle'].max().date()}; holding days {days.min()}–{days.max()} (median {days.median():.0f}); "
          f"AM-settled {int((contracts['am_settlement'] == 1).sum())}")
    daily = pull_daily(contracts)
    write_parquet(daily, "data/processed/strategy_options_daily.parquet")
    print(f"[ok  ] strategy_options_daily.parquet: {len(daily):,} rows for {daily['optionid'].nunique()} contracts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)
