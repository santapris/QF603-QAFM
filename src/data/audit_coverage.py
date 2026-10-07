"""Phase 1a coverage audit: first/last dates and gaps for every source, before bulk downloads.

Reads only metadata and small aggregates (nothing is saved to data/raw). Writes
``outputs/tables/data_coverage.csv`` (one row per series) and ``outputs/tables/data_coverage_gaps.csv``
(every missing trading day / month-end / report gap found), and prints the 🚦 gate result:
any series ending before ``sample.end`` must be reported to the user.

Run:  python -m src.data.audit_coverage
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

import pandas as pd

from src.utils.dates import month_end_trading_days, trading_days
from src.utils.io import load_config, project_path

SPX_SECID = 108105
FRED_SERIES = ["DCPF3M", "DCPN3M", "DTB3", "DBAA", "DAAA", "VIXCLS"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CFTC_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"   # TFF, futures only
CFTC_VIX_CODE = "1170E1"                                                  # Cboe VIX futures
BLOOMBERG_TICKERS = ["VIX Index", "VIX3M Index", "VVIX Index", "SKEW Index", "SPXT Index",
                     "PUT Index", "BXM Index", "UX1 Index", "UX2 Index"]
BLOOMBERG_OPTIONAL = {"UX1 Index", "UX2 Index"}


def bloomberg_filename(ticker: str) -> str:
    """Expected export file name, e.g. 'VIX3M Index' -> 'VIX3M_Index.csv'."""
    return ticker.replace(" ", "_") + ".csv"


def _row(series, source, first=None, last=None, n_obs=None, n_missing=0, notes=""):
    return dict(series=series, source=source, first=first, last=last, n_obs=n_obs,
                n_missing_in_sample=n_missing, notes=notes)


def _gaps(series, kind, dates, detail=""):
    return [dict(series=series, kind=kind, date=pd.Timestamp(d).date(), detail=detail) for d in dates]


# ---------------------------------------------------------------- WRDS: OptionMetrics
def audit_optionmetrics(db, cfg):
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    tables = set(db.list_tables("optionm"))
    years = sorted(int(t[6:]) for t in tables if re.fullmatch(r"opprcd\d{4}", t))
    dates = []
    for y in range(start.year, max(years) + 1):
        if y not in years:
            continue
        q = f"select distinct date from optionm.opprcd{y} where secid = {SPX_SECID}"
        dates.append(pd.to_datetime(db.raw_sql(q)["date"]))
    have = pd.DatetimeIndex(pd.concat(dates).sort_values().unique())
    in_sample = trading_days(start.to_period("M").start_time, end)
    missing_days = in_sample.difference(have)
    missing_me = month_end_trading_days(start, end).difference(have)
    rows = [_row("SPX options (secid 108105)", "WRDS optionm.opprcdYYYY", have.min().date(), have.max().date(),
                 len(have), len(missing_days),
                 f"tables {min(years)}-{max(years)}; missing month-ends in sample: {len(missing_me)}")]
    gaps = _gaps("SPX options", "missing trading day", missing_days) + \
        _gaps("SPX options", "missing month-end", missing_me)

    z = db.raw_sql("select min(date) as first, max(date) as last, count(distinct date) as n from optionm.zerocd")
    zd = pd.to_datetime(db.raw_sql(f"select distinct date from optionm.zerocd where date >= '{start.date()}'")["date"])
    zmiss = month_end_trading_days(start, end).difference(pd.DatetimeIndex(zd))
    rows.append(_row("Zero curve", "WRDS optionm.zerocd", z.at[0, "first"], z.at[0, "last"], int(z.at[0, "n"]),
                     len(zmiss), f"missing month-ends in sample: {len(zmiss)}"))
    gaps += _gaps("Zero curve", "missing month-end", zmiss)

    s = db.raw_sql(f"select date, open, close from optionm.secprd where secid = {SPX_SECID} "
                   f"and date >= '{start.to_period('M').start_time.date()}'", date_cols=["date"])
    s = s.set_index("date").sort_index()
    smiss = in_sample.difference(s.index)
    n_open_missing = int(s["open"].isna().sum() + (s["open"] <= 0).sum())
    rows.append(_row("SPX index close/open", "WRDS optionm.secprd", s.index.min().date(), s.index.max().date(),
                     len(s), len(smiss), f"rows with missing/non-positive open: {n_open_missing}"))
    gaps += _gaps("SPX index", "missing trading day", smiss)
    return rows, gaps


# ---------------------------------------------------------------- WRDS: TAQ
def audit_taq(db, cfg):
    end = pd.Timestamp(cfg["sample"]["end"])
    libs = sorted(l for l in db.list_libraries() if re.fullmatch(r"taqm_\d{4}", l))
    dates = []
    for lib in libs:
        dates += [pd.Timestamp(t[4:]) for t in db.list_tables(lib) if re.fullmatch(r"ctm_\d{8}", t)]
    have = pd.DatetimeIndex(sorted(dates))
    first, last = have.min(), have.max()
    missing = trading_days(first, min(last, end)).difference(have)

    # SPY must be present on the first and last day of every year (cheap spot checks).
    spy_missing = []
    for d in pd.Series(have).groupby(have.year).agg(["min", "max"]).to_numpy().ravel():
        d = pd.Timestamp(d)
        q = (f"select count(*) as n from taqm_{d.year}.ctm_{d:%Y%m%d} "
             f"where sym_root = 'SPY' and coalesce(sym_suffix, '') = ''")
        if int(db.raw_sql(q).at[0, "n"]) == 0:
            spy_missing.append(d)
    rows = [_row("SPY trades (daily TAQ, ctm)", f"WRDS {libs[0]}..{libs[-1]}", first.date(), last.date(),
                 len(have), len(missing),
                 f"SPY absent on {len(spy_missing)} of the yearly first/last-day spot checks; "
                 f"proposed har_burnin_start = {first.date()}")]
    gaps = _gaps("TAQ ctm", "missing trading day", missing) + _gaps("TAQ SPY", "SPY absent (spot check)", spy_missing)
    return rows, gaps, first


# ---------------------------------------------------------------- FRED
def audit_fred(cfg):
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    rows, gaps = [], []
    me = month_end_trading_days(start, end)
    for sid in FRED_SERIES:
        df = pd.read_csv(FRED_URL.format(sid))
        s = pd.to_numeric(df[sid], errors="coerce")
        s.index = pd.to_datetime(df["observation_date"])
        valid = s.dropna()
        # month-ends whose latest valid value is more than 7 calendar days old
        pos = valid.index.searchsorted(me, side="right") - 1
        stale = [m for m, p in zip(me, pos) if p < 0 or (m - valid.index[p]).days > 7]
        window = s.loc[start.to_period("M").start_time:end]
        n_missing = int(window.isna().sum())
        rows.append(_row(sid, "FRED", valid.index.min().date(), valid.index.max().date(), len(valid), n_missing,
                         f"missing values in sample window (incl. holidays): {n_missing}; "
                         f"month-ends with value >7d stale: {len(stale)}"))
        gaps += _gaps(sid, "month-end stale >7d", stale)
    return rows, gaps


# ---------------------------------------------------------------- CFTC
def audit_cftc(cfg):
    params = {"$select": "report_date_as_yyyy_mm_dd,open_interest_all,dealer_positions_long_all,"
                         "dealer_positions_short_all,lev_money_positions_long,lev_money_positions_short",
              "$where": f"cftc_contract_market_code='{CFTC_VIX_CODE}'",
              "$order": "report_date_as_yyyy_mm_dd", "$limit": "5000"}
    url = CFTC_TFF_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        df = pd.DataFrame(json.load(r))
    d = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).sort_values()
    step = d.diff().dt.days
    long_gaps = d[step > 7]
    in_sample = d[(d >= pd.Timestamp(cfg["sample"]["start"]) - pd.Timedelta(days=31))
                  & (d <= pd.Timestamp(cfg["sample"]["end"]))]
    nulls = int(df.drop(columns="report_date_as_yyyy_mm_dd").isna().sum().sum())
    rows = [_row("VIX futures positioning (TFF)", f"CFTC TFF futures-only, code {CFTC_VIX_CODE}",
                 d.min().date(), d.max().date(), len(d), int((step.loc[in_sample.index] > 7).sum()),
                 f"reports in sample: {len(in_sample)}; gaps >7d: {len(long_gaps)}; null position fields: {nulls}; "
                 "dataset has as-of dates only — release dates must be inferred (Phase 1f)")]
    gaps = [dict(series="CFTC TFF", kind="report gap >7d", date=x.date(), detail=f"{int(g)} days since previous")
            for x, g in zip(long_gaps, step[step > 7])]
    return rows, gaps


# ---------------------------------------------------------------- manual sources
def audit_manual():
    rows = []
    bb = project_path("data", "raw", "bloomberg")
    for t in BLOOMBERG_TICKERS:
        f = bb / bloomberg_filename(t)
        if not f.exists():
            note = ("optional — " if t in BLOOMBERG_OPTIONAL else "") + f"NOT YET EXPORTED (expected {f.name})"
            rows.append(_row(t, "Bloomberg (🧑 export)", notes=note))
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        rows.append(_row(t, f"Bloomberg {f.name}", df["date"].min().date(), df["date"].max().date(), len(df)))
    hkm = [p for p in project_path("data", "raw", "hkm").iterdir() if p.name != ".gitkeep"]
    rows.append(_row("HKM capital ratio", "He-Kelly-Manela website (🧑 download)",
                     notes="NOT YET DOWNLOADED" if not hkm else f"files present: {[p.name for p in hkm]} — inspect"))
    return rows


def main(cfg=None, force=False):
    from src.utils.wrds_conn import connect

    cfg = cfg or load_config()
    rows, gaps = [], []
    db = connect()
    try:
        r, g = audit_optionmetrics(db, cfg); rows += r; gaps += g
        r, g, burnin = audit_taq(db, cfg); rows += r; gaps += g
    finally:
        db.close()
    r, g = audit_fred(cfg); rows += r; gaps += g
    r, g = audit_cftc(cfg); rows += r; gaps += g
    rows += audit_manual()

    out = pd.DataFrame(rows)
    out_dir = project_path("outputs", "tables")
    out.to_csv(out_dir / "data_coverage.csv", index=False)
    pd.DataFrame(gaps, columns=["series", "kind", "date", "detail"]).to_csv(
        out_dir / "data_coverage_gaps.csv", index=False)

    end = pd.Timestamp(cfg["sample"]["end"]).date()
    short = out[out["last"].notna() & (pd.to_datetime(out["last"]).dt.date < end - pd.Timedelta(days=7))]
    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        print(out.to_string(index=False))
    print(f"\nProposed sample.har_burnin_start = {burnin.date()}")
    if len(short):
        print("\n🚦 GATE: series ending before sample end — report to user:\n" + short[["series", "last"]].to_string(index=False))
    else:
        print("\n🚦 GATE: every audited series reaches the sample end.")
    return out


if __name__ == "__main__":
    main()
