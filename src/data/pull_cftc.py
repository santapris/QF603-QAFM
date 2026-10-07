"""Phase 1f: CFTC Traders in Financial Futures (TFF) positioning in Cboe VIX futures.

Raw (untouched): ``data/raw/cftc/tff_futures_only_{code}.csv`` — every weekly report for the VIX futures
contract from the CFTC Public Reporting API (dataset ``gpe5-46if``, D015), all fields, plus ``_pull_log.json``.

Processed:
* ``data/processed/cftc_weekly.parquet`` — one row per report: as-of date, inferred release date (D025),
  open interest, dealer / leveraged-money / asset-manager nets and the positioning measures (D024).
* ``data/processed/cftc_monthly.parquet`` — month-end value = last report **released** on or before the
  month-end, NaN if that release is older than ``max_stale_days.cftc`` (D018).

Run:  python -m src.data.pull_cftc [--force]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from src.utils.dates import month_end_trading_days
from src.utils.io import load_config, project_path, write_parquet

API_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
RAW_DIR = ("data", "raw", "cftc")
FRIDAY = 4


def download(code: str, force: bool) -> dict:
    path = project_path(*RAW_DIR, f"tff_futures_only_{code}.csv")
    if path.exists() and not force:
        print(f"[skip] {path.name} exists")
        return {}
    params = {"$where": f"cftc_contract_market_code='{code}'", "$order": "report_date_as_yyyy_mm_dd",
              "$limit": "50000"}
    with urllib.request.urlopen(API_URL + "?" + urllib.parse.urlencode(params), timeout=120) as r:
        records = json.load(r)
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"[ok  ] {path.name}: {len(df):,} reports, {df.shape[1]} fields")
    return {path.name: dict(reports=len(df), downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            source=API_URL, where=params["$where"])}


def release_dates(as_of: pd.Series, overrides: list[dict]) -> pd.Series:
    """Infer the public release date of each report (D025)."""
    as_of = pd.to_datetime(as_of)
    holidays = USFederalHolidayCalendar().holidays(as_of.min() - pd.Timedelta(days=7),
                                                  as_of.max() + pd.Timedelta(days=30))
    bday = CustomBusinessDay(holidays=holidays)
    friday = as_of + pd.to_timedelta((FRIDAY - as_of.dt.weekday) % 7, unit="D")
    hit = pd.Series([bool(((holidays > a) & (holidays <= f)).any()) for a, f in zip(as_of, friday)],
                    index=as_of.index)
    next_bday = pd.Series([f + bday for f in friday], index=friday.index)
    release = friday.where(~hit, next_bday)
    for o in overrides:
        lo, hi, rel = pd.Timestamp(o["as_of_from"]), pd.Timestamp(o["as_of_to"]), pd.Timestamp(o["release"])
        in_window = (as_of >= lo) & (as_of <= hi)
        release = release.where(~in_window, release.clip(lower=rel))
    return release


def build_weekly(code: str, cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(project_path(*RAW_DIR, f"tff_futures_only_{code}.csv"))
    num = lambda c: pd.to_numeric(raw[c], errors="raise")  # noqa: E731
    w = pd.DataFrame({
        "as_of": pd.to_datetime(raw["report_date_as_yyyy_mm_dd"]).dt.normalize(),
        "open_interest": num("open_interest_all"),
        "dealer_net": num("dealer_positions_long_all") - num("dealer_positions_short_all"),
        "lev_net": num("lev_money_positions_long") - num("lev_money_positions_short"),
        "asset_mgr_net": num("asset_mgr_positions_long") - num("asset_mgr_positions_short"),
    }).sort_values("as_of").reset_index(drop=True)
    if w["as_of"].duplicated().any():
        raise ValueError("duplicate report dates in CFTC data")
    w["release"] = release_dates(w["as_of"], cfg["cftc"]["shutdown_release_overrides"])
    w["cftc_pos"] = w["dealer_net"] / w["open_interest"]
    w["cftc_pos_lev"] = w["lev_net"] / w["open_interest"]
    return w


def build_monthly(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    me = month_end_trading_days(start, end)
    w = weekly.sort_values("release")
    pos = w["release"].searchsorted(me, side="right") - 1
    rows = w.iloc[np.clip(pos, 0, None)].reset_index(drop=True)
    m = pd.DataFrame({"report_as_of": rows["as_of"].to_numpy(), "release": rows["release"].to_numpy(),
                      "cftc_pos": rows["cftc_pos"].to_numpy(), "cftc_pos_lev": rows["cftc_pos_lev"].to_numpy(),
                      "open_interest": rows["open_interest"].to_numpy()}, index=pd.DatetimeIndex(me, name="date"))
    m["release_age_days"] = (m.index - m["release"]).dt.days
    stale = (pos < 0) | (m["release_age_days"] > cfg["max_stale_days"]["cftc"]).to_numpy()
    m.loc[stale, ["cftc_pos", "cftc_pos_lev", "open_interest"]] = np.nan
    return m


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    code = str(cfg["cftc"]["contract_code"])
    log_path = project_path(*RAW_DIR, "_pull_log.json")
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log.update(download(code, force))
    log_path.write_text(json.dumps(log, indent=2))

    weekly = build_weekly(code, cfg)
    monthly = build_monthly(weekly, cfg)
    write_parquet(weekly, "data/processed/cftc_weekly.parquet")
    write_parquet(monthly, "data/processed/cftc_monthly.parquet")
    nan = monthly.index[monthly["cftc_pos"].isna()]
    print(f"[ok  ] cftc_weekly.parquet: {len(weekly)} reports {weekly['as_of'].min().date()} → {weekly['as_of'].max().date()}")
    print(f"[ok  ] cftc_monthly.parquet: {len(monthly)} month-ends; NaN at {[str(d.date()) for d in nan]}")
    print(f"       release age at month-end: median {monthly['release_age_days'].median():.0f}d, "
          f"max non-NaN {monthly.loc[monthly['cftc_pos'].notna(), 'release_age_days'].max()}d")
    return weekly, monthly


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)
