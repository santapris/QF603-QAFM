"""Phase 1b: pull SPX options (month-end ± buffer days), zero curve and SPX index prices from OptionMetrics.

Outputs (raw, untouched values — note ``strike_price`` is ×1000):
* ``data/raw/optionmetrics/spx_options_YYYY.parquet`` — all SPX (secid 108105) option quotes on the
  request dates falling in year YYYY (one file per ``optionm.opprcdYYYY`` table; resumable).
* ``data/raw/optionmetrics/zerocd_request_dates.parquet`` — zero curve on the same request dates.
* ``data/raw/optionmetrics/spx_index_daily.parquet`` — SPX daily open/high/low/close/return from
  ``sample.har_burnin_start`` to ``sample.end`` (prototype RV stand-in, delta hedging, AM settlement proxy).
* ``data/raw/optionmetrics/_pull_log.json`` — row counts, request dates without data, timestamps.

Request dates = every month-end trading day from one month before ``sample.start`` to ``sample.end``,
plus ``BUFFER_DAYS`` NYSE trading days either side (prior days: MFIV fallback for bad month-ends;
following days: t+1 strategy execution). Dates after ``sample.end`` are not requested.

Run:  python -m src.data.pull_optionmetrics [--force]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src.utils.dates import month_end_trading_days, trading_days
from src.utils.io import load_config, project_path

SPX_SECID = 108105
BUFFER_DAYS = 3
RAW_DIR = ("data", "raw", "optionmetrics")

# PLAN 1b fields plus cheap extras that identify series/settlement (D019).
OPTION_COLUMNS = [
    "date", "exdate", "last_date", "optionid", "symbol", "root", "suffix", "expiry_indicator",
    "am_settlement", "ss_flag", "cp_flag", "strike_price", "best_bid", "best_offer", "volume",
    "open_interest", "impl_volatility", "delta", "gamma", "vega", "theta", "forward_price",
    "contract_size", "cfadj",
]


def request_dates(cfg: dict) -> pd.DatetimeIndex:
    """Month-end trading days (from the month before sample.start) ± BUFFER_DAYS, capped at sample.end."""
    start = pd.Timestamp(cfg["sample"]["start"]) - pd.DateOffset(months=1)
    end = pd.Timestamp(cfg["sample"]["end"])
    me = month_end_trading_days(start, end)
    cal = trading_days(me.min() - pd.Timedelta(days=15), me.max() + pd.Timedelta(days=15))
    pos = cal.get_indexer(me)
    idx = sorted({p + o for p in pos for o in range(-BUFFER_DAYS, BUFFER_DAYS + 1)})
    dates = cal[idx]
    return pd.DatetimeIndex(dates[dates <= end], name="date")


def _date_list(dates) -> str:
    return ", ".join(f"'{d:%Y-%m-%d}'" for d in dates)


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    df["last_date"] = pd.to_datetime(df["last_date"])
    df["optionid"] = df["optionid"].astype("int64")
    df["am_settlement"] = df["am_settlement"].astype("Int8")
    for c in ["symbol", "root", "suffix", "expiry_indicator", "ss_flag", "cp_flag"]:
        df[c] = df[c].astype("string")
    return df.sort_values(["date", "exdate", "cp_flag", "strike_price"]).reset_index(drop=True)


def pull_options(db, dates: pd.DatetimeIndex, force: bool, log: dict) -> None:
    out_dir = project_path(*RAW_DIR)
    available = {int(t[6:]) for t in db.list_tables("optionm") if t.startswith("opprcd") and t[6:].isdigit()}
    for year, ds in pd.Series(dates, index=dates).groupby(dates.year):
        path = out_dir / f"spx_options_{year}.parquet"
        if path.exists() and not force:
            print(f"[skip] {path.name} exists")
            continue
        if year not in available:
            raise RuntimeError(f"optionm.opprcd{year} not available on WRDS")
        q = (f"select {', '.join(OPTION_COLUMNS)} from optionm.opprcd{year} "
             f"where secid = {SPX_SECID} and date in ({_date_list(ds)})")
        t0 = datetime.now()
        df = _typed(db.raw_sql(q))
        df.to_parquet(path, index=False)
        missing = sorted(set(ds.dt.date.astype(str)) - set(df["date"].dt.date.astype(str)))
        log["options"][str(year)] = dict(rows=len(df), request_dates=len(ds), dates_with_data=df["date"].nunique(),
                                         request_dates_without_data=missing,
                                         seconds=round((datetime.now() - t0).total_seconds(), 1))
        print(f"[ok  ] {path.name}: {len(df):,} rows, {df['date'].nunique()}/{len(ds)} dates", flush=True)


def pull_zero_curve(db, dates: pd.DatetimeIndex, force: bool, log: dict) -> None:
    path = project_path(*RAW_DIR, "zerocd_request_dates.parquet")
    if path.exists() and not force:
        print(f"[skip] {path.name} exists")
        return
    df = db.raw_sql(f"select date, days, rate from optionm.zerocd where date in ({_date_list(dates)})",
                    date_cols=["date"]).sort_values(["date", "days"])
    df.to_parquet(path, index=False)
    log["zero_curve"] = dict(rows=len(df), dates_with_data=df["date"].nunique(), request_dates=len(dates))
    print(f"[ok  ] {path.name}: {len(df):,} rows")


def pull_index(db, cfg: dict, force: bool, log: dict) -> None:
    path = project_path(*RAW_DIR, "spx_index_daily.parquet")
    if path.exists() and not force:
        print(f"[skip] {path.name} exists")
        return
    start, end = cfg["sample"]["har_burnin_start"], cfg["sample"]["end"]
    q = (f'select date, open, high, low, close, volume, "return" from optionm.secprd '
         f"where secid = {SPX_SECID} and date between '{start}' and '{end}'")
    df = db.raw_sql(q, date_cols=["date"]).sort_values("date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    log["index"] = dict(rows=len(df), first=str(df["date"].min().date()), last=str(df["date"].max().date()))
    print(f"[ok  ] {path.name}: {len(df):,} rows")


def main(cfg=None, force=False):
    from src.utils.wrds_conn import connect

    cfg = cfg or load_config()
    dates = request_dates(cfg)
    log_path = project_path(*RAW_DIR, "_pull_log.json")
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log.setdefault("options", {})
    log["buffer_days"] = BUFFER_DAYS
    log["columns"] = OPTION_COLUMNS

    db = connect()
    try:
        pull_options(db, dates, force, log)
        pull_zero_curve(db, dates, force, log)
        pull_index(db, cfg, force, log)
    finally:
        log["last_run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log_path.write_text(json.dumps(log, indent=2))
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)
