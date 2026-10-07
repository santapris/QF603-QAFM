"""Phase 1c: SPY 5-minute bars from daily TAQ (WRDS ``taqm_YYYY.ctm_YYYYMMDD``), reduced on the server.

For each trading day from ``sample.har_burnin_start`` to ``sample.end`` the server returns one row per
5-minute bucket b = floor((time - 09:30) / 5 min), b = 0..77, over regular hours [09:30, 16:00):
trade count, volume, VWAP, low, high, and the first and last trade (time, price) in the bucket.
Bucket 0's first trade is the first trade after 09:30; the last non-empty bucket's last trade is the
last trade before 16:00 (early-close days simply have fewer buckets).

Trade filters (D021):
* ``sym_root = 'SPY'`` and empty ``sym_suffix``;
* ``tr_corr = '00'`` (original, uncorrected, not cancelled);
* ``price > 0``, ``size > 0``;
* sale condition ``tr_scond`` made only of the characters '@', 'E', 'F', 'I' or blanks (regular trades,
  automatic executions, intermarket sweeps, odd lots). Excludes e.g. T (extended hours / late), Z and L
  (out of sequence), O / Q / 6 / M (opening / closing prints), 4 (derivatively priced), B, C, N, P, R,
  U, V, W, X.

Outputs: ``data/raw/taq/spy_5min_YYYY.parquet`` (one file per year; resumable, with a partial
checkpoint every CHECKPOINT_DAYS days) and ``data/raw/taq/_pull_log.json``.

Run:  python -m src.data.pull_taq [--force] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from src.utils.io import load_config, project_path

RAW_DIR = ("data", "raw", "taq")
CHECKPOINT_DAYS = 20
ALLOWED_SCOND = "@EFI "

QUERY = """
with t as (
  select date, time_m, tr_seqnum, price::float8 as price, size,
         floor(extract(epoch from (time_m - time '09:30:00')) / 300)::int as b
  from {table}
  where sym_root = 'SPY' and coalesce(sym_suffix, '') = '' and tr_corr = '00'
    and price > 0 and size > 0
    and time_m >= time '09:30:00' and time_m < time '16:00:00'
    and translate(coalesce(tr_scond, ''), '{allowed}', '') = ''
)
select date, b, count(*) as n, sum(size) as volume, sum(price * size) / sum(size) as vwap,
       min(price) as low, max(price) as high,
       min(time_m) as first_time, (array_agg(price order by time_m, tr_seqnum))[1] as first_price,
       max(time_m) as last_time, (array_agg(price order by time_m desc, tr_seqnum desc))[1] as last_price
from t group by date, b order by b
"""


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["b"] = df["b"].astype("int16")
    df["n"] = df["n"].astype("int32")
    df["volume"] = df["volume"].astype("int64")
    for c in ["first_time", "last_time"]:
        df[c] = pd.to_timedelta(df[c].astype(str))
    return df


def trading_dates(db, year: int, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    tables = db.list_tables(f"taqm_{year}")
    dates = sorted(pd.Timestamp(t[4:]) for t in tables if re.fullmatch(r"ctm_\d{8}", t))
    return [d for d in dates if start <= d <= end]


def pull_year(year: int, start: str, end: str, force: bool) -> dict:
    """Pull one year (runs in its own process with its own WRDS connection)."""
    from src.utils.wrds_conn import connect

    out_dir = project_path(*RAW_DIR)
    final = out_dir / f"spy_5min_{year}.parquet"
    partial = out_dir / f"spy_5min_{year}.partial.parquet"
    if final.exists() and not force:
        return dict(year=year, status="skipped (exists)")
    if force and partial.exists():
        partial.unlink()

    t0 = time.time()
    db = connect()
    try:
        dates = trading_dates(db, year, pd.Timestamp(start), pd.Timestamp(end))
        frames = [pd.read_parquet(partial)] if partial.exists() else []
        done = set(frames[0]["date"].dt.normalize()) if frames else set()
        empty = []
        todo = [d for d in dates if d not in done]
        for i, d in enumerate(todo, 1):
            df = db.raw_sql(QUERY.format(table=f"taqm_{year}.ctm_{d:%Y%m%d}", allowed=ALLOWED_SCOND))
            if df.empty:
                empty.append(str(d.date()))
            else:
                frames.append(_typed(df))
            if i % CHECKPOINT_DAYS == 0 and frames:
                pd.concat(frames, ignore_index=True).to_parquet(partial, index=False)
                print(f"  {year}: {len(done) + i}/{len(dates)} days", flush=True)
    finally:
        db.close()

    out = pd.concat(frames, ignore_index=True).sort_values(["date", "b"]).reset_index(drop=True)
    out.to_parquet(final, index=False)
    if partial.exists():
        partial.unlink()
    info = dict(year=year, status="ok", days_requested=len(dates), days_with_data=int(out["date"].nunique()),
                days_empty=empty, rows=len(out), trades=int(out["n"].sum()), seconds=round(time.time() - t0))
    print(f"[ok  ] {final.name}: {info['days_with_data']}/{len(dates)} days, {info['trades']:,} trades, "
          f"{info['seconds']}s", flush=True)
    return info


def main(cfg=None, force=False, workers: int = 3):
    cfg = cfg or load_config()
    start = pd.Timestamp(cfg["sample"]["har_burnin_start"])
    end = pd.Timestamp(cfg["sample"]["end"])
    log_path = project_path(*RAW_DIR, "_pull_log.json")
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log.setdefault("years", {})
    log.update(filters=dict(sym_root="SPY", sym_suffix="empty", tr_corr="00", price="> 0", size="> 0",
                            hours="[09:30, 16:00)", tr_scond_allowed_chars=ALLOWED_SCOND),
               bucket_minutes=5, start=str(start.date()), end=str(end.date()))

    years = range(start.year, end.year + 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(pull_year, y, str(start.date()), str(end.date()), force): y for y in years}
        for f in as_completed(futures):
            y = futures[f]
            try:
                info = f.result()
            except Exception as e:  # keep other years going; record the failure
                info = dict(year=y, status=f"FAILED: {type(e).__name__}: {e}")
                print(f"[fail] {y}: {info['status']}", flush=True)
            if not info["status"].startswith("skipped"):
                log["years"][str(y)] = info
            log["last_run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            log_path.write_text(json.dumps(log, indent=2))

    failed = [y for y, v in log["years"].items() if str(v.get("status", "")).startswith("FAILED")]
    if failed:
        raise SystemExit(f"TAQ pull failed for years {failed}; re-run to resume")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    main(force=args.force, workers=args.workers)
