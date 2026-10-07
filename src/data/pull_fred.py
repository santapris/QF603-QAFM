"""Phase 1e: FRED daily rates → raw CSVs and a month-end panel with publication lags.

Raw (untouched): ``data/raw/fred/{SERIES}.csv`` from the public fredgraph CSV endpoint, plus
``_pull_log.json``. ``VIXCLS`` is downloaded for the Phase 1.5 prototype only.

Processed: ``data/processed/fred_monthly.parquet`` indexed by month-end trading day (sample months), with
* each rate lagged ``publication_lag.fred_daily_bdays`` NYSE trading days, sampled as the last value on or
  before month-end, and set to NaN if that value is older than ``max_stale_days.fred_daily`` (D022);
* ``fund_spread = DCPF3M - DTB3`` (headline, D016) and ``fund_spread_nonfin = DCPN3M - DTB3`` (robustness);
* ``credit_spread = DBAA - DAAA``;
* ``rf``: T-bill holding-period return from month-end t to month-end t+1, set at t from DTB3 (D023).
Rates and spreads are stored in decimals (0.005 = 50 bp).

Run:  python -m src.data.pull_fred [--force]
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.dates import lag_business_days, month_end_trading_days, to_month_end
from src.utils.io import load_config, project_path, write_parquet

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SERIES = ["DCPF3M", "DCPN3M", "DTB3", "DBAA", "DAAA"]
PROTOTYPE_ONLY = ["VIXCLS"]
RAW_DIR = ("data", "raw", "fred")
TBILL_DAYS = 91


def download(force: bool) -> dict:
    log = {}
    for sid in SERIES + PROTOTYPE_ONLY:
        path = project_path(*RAW_DIR, f"{sid}.csv")
        if path.exists() and not force:
            print(f"[skip] {path.name} exists")
            continue
        with urllib.request.urlopen(FRED_URL.format(sid), timeout=60) as r:
            raw = r.read()
        pd.read_csv(io.BytesIO(raw), usecols=["observation_date", sid])      # validate format before saving
        path.write_bytes(raw)
        log[sid] = dict(bytes=len(raw), downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        print(f"[ok  ] {path.name}: {len(raw):,} bytes")
    return log


def load_raw(sid: str) -> pd.Series:
    df = pd.read_csv(project_path(*RAW_DIR, f"{sid}.csv"), parse_dates=["observation_date"])
    s = pd.to_numeric(df[sid], errors="coerce")
    s.index = pd.DatetimeIndex(df["observation_date"], name="date")
    return s.dropna().rename(sid)


def tbill_holding_return(discount_pct: pd.Series, days: pd.Series) -> pd.Series:
    """Holding-period return over ``days`` calendar days from a 3M T-bill discount rate (percent).

    Discount rate d → investment (bond-equivalent) yield y = 365 d / (360 - 91 d), then
    compounded over the holding period: (1 + y)^(days / 365) - 1.
    """
    d = discount_pct / 100
    y = 365 * d / (360 - TBILL_DAYS * d)
    return (1 + y) ** (days / 365) - 1


def build_monthly(cfg: dict) -> pd.DataFrame:
    lag = cfg["publication_lag"]["fred_daily_bdays"]
    max_stale = cfg["max_stale_days"]["fred_daily"]
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    me = month_end_trading_days(start - pd.DateOffset(months=1), end)

    cols = {}
    for sid in SERIES:
        cols[sid.lower()] = to_month_end(lag_business_days(load_raw(sid), lag), me, max_stale_days=max_stale)
    m = pd.DataFrame(cols)
    for c in m.columns:
        m[c] = m[c] / 100
    m["fund_spread"] = m["dcpf3m"] - m["dtb3"]
    m["fund_spread_nonfin"] = m["dcpn3m"] - m["dtb3"]
    m["credit_spread"] = m["dbaa"] - m["daaa"]

    next_me = pd.Series(me[1:].append(pd.DatetimeIndex([pd.NaT])), index=me)
    days = (next_me - pd.Series(me, index=me)).dt.days
    m["rf"] = tbill_holding_return(m["dtb3"] * 100, days)
    m.loc[days.isna().to_numpy(), "rf"] = np.nan          # holding period ends after the sample
    return m.loc[start:end]


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    log_path = project_path(*RAW_DIR, "_pull_log.json")
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log.update(download(force))
    log_path.write_text(json.dumps(log, indent=2))

    m = build_monthly(cfg)
    write_parquet(m, "data/processed/fred_monthly.parquet")
    missing = m.isna().sum()
    print(f"[ok  ] fred_monthly.parquet: {len(m)} month-ends {m.index.min().date()} → {m.index.max().date()}")
    print("NaN month-ends per column:", missing[missing > 0].to_dict() or "none")
    for c in missing[missing > 0].index:
        print(f"  {c}: {[str(d.date()) for d in m.index[m[c].isna()]][:40]}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)
