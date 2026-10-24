"""Step 01-02: download spot prices and filter inputs, cache them raw, build the clean tables.

Sources (all free, no API key):
- FRED: WTI and Brent spot, NY Harbor conventional gasoline spot, OVX.
- EIA daily spot sheet: NY Harbor ULSD.
- EIA Weekly Petroleum Status Report history: crude ex-SPR, Cushing, gasoline, distillate, refinery runs.
- Caldara-Iacoviello daily geopolitical risk (GPR) index.
- CFTC disaggregated Commitments of Traders (futures only): managed-money positions.

Point-in-time rules. Every feature is stamped on the first day a trader could have used it:
- Weekly EIA stocks (week ending Friday) count as known from week-end + 6 calendar days (Thursday),
  which also covers holiday-delayed Wednesday releases.
- COT positions (as of Tuesday) are released Friday 3:30 pm ET, after the NYMEX settle, so they count
  as known from as-of + `cftc_release_lag_days` (6: the following Monday close).
- OVX closes at 4:15 pm ET, after the 2:30 pm settle, so it counts as known `ovx_lag_days` (1) later.
- GPR counts as known `gpr_lag_days` after its date.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import load_config, path
from .spreads import build_spreads

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
EIA_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/{key}{freq}.xls"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
CFTC_URL = "https://www.cftc.gov/files/dea/history/{name}"
CFTC_HIST_ZIP = "fut_disagg_txt_hist_2006_2016.zip"
CFTC_CODES = {"cl": "067651", "rb": "111659", "ho": "022651"}   # NYMEX WTI, RBOB, NY Harbor ULSD
UA = {"User-Agent": "Mozilla/5.0 (QF603 student research; contact via SMU)"}


# --------------------------------------------------------------------------- download
def _raw(sub: str) -> Path:
    d = path("raw_dir") / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path, refresh: bool = False, max_age_days: float | None = None,
              headers: dict | None = UA, tries: int = 3) -> Path:
    """Fetch `url` to `dest` unless a fresh enough copy exists. The file is kept exactly as received.

    FRED stalls on a custom User-Agent, so its calls pass headers=None.
    """
    if max_age_days is None:
        max_age_days = load_config()["data"]["cache_max_age_days"]
    stale = not dest.exists() or (time.time() - dest.stat().st_mtime) > max_age_days * 86400
    if refresh or stale:
        for attempt in range(tries):
            try:
                r = requests.get(url, headers=headers, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == tries - 1:
                    if dest.exists():                 # offline: fall back to the cached copy
                        return dest
                    raise
                time.sleep(5 * (attempt + 1))
        dest.write_bytes(r.content)
    return dest


def _parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """FRED marks missing values with '.' (older files) or an empty field (current files)."""
    df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True, na_values=[".", ""])
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.index.name = "date"
    return s.rename(series_id)


def fred(series_id: str, refresh: bool = False) -> pd.Series:
    """One FRED series, cached under data/raw/fred/<id>.csv."""
    f = _download(FRED_CSV_URL.format(sid=series_id), _raw("fred") / f"{series_id}.csv", refresh,
                  headers=None)
    text = f.read_text()
    if not text.startswith("observation_date") and "DATE" not in text[:20]:
        raise ValueError(f"Unexpected FRED response for {series_id}: {text[:200]!r}")
    return _parse_fred_csv(text, series_id)


def _parse_eia_xls(f: Path | io.BytesIO, key: str) -> pd.Series:
    """EIA history workbook: sheet 'Data 1', two title rows, then a Date column and one value column."""
    df = pd.read_excel(f, sheet_name="Data 1", skiprows=2)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]), name="date"), name=key)
    return s.dropna().sort_index()


def eia(key: str, freq: str = "d", refresh: bool = False) -> pd.Series:
    """One EIA series by source key; freq 'd' (daily spot) or 'w' (weekly)."""
    f = _download(EIA_XLS_URL.format(key=key, freq=freq), _raw("eia") / f"{key}{freq}.xls", refresh)
    return _parse_eia_xls(f, key)


def gpr_daily(refresh: bool = False) -> pd.Series:
    """Daily GPR index (GPRD), indexed by calendar date."""
    f = _download(GPR_URL, _raw("gpr") / "data_gpr_daily_recent.xls", refresh)
    df = pd.read_excel(f, sheet_name=0)
    s = pd.Series(df["GPRD"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df["date"]), name="date"))
    return s.dropna().sort_index().rename("gpr")


def _cftc_zips(refresh: bool = False) -> list[Path]:
    """History 2006-16 (frozen) plus one file per year since 2017; only the current year is refreshed."""
    files = [_download(CFTC_URL.format(name=CFTC_HIST_ZIP), _raw("cftc") / CFTC_HIST_ZIP,
                       refresh=False, max_age_days=np.inf)]
    this_year = pd.Timestamp.today().year
    for y in range(2017, this_year + 1):
        name = f"fut_disagg_txt_{y}.zip"
        files.append(_download(CFTC_URL.format(name=name), _raw("cftc") / name, refresh=refresh,
                               max_age_days=None if y == this_year else np.inf))
    return files


def _parse_cftc(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Managed-money net and open interest per market, indexed by the Tuesday as-of date."""
    keep = ["CFTC_Contract_Market_Code", "Report_Date_as_YYYY-MM-DD", "Open_Interest_All",
            "M_Money_Positions_Long_All", "M_Money_Positions_Short_All"]
    c = pd.concat([f[keep] for f in frames], ignore_index=True)
    c["code"] = c["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
    c["asof"] = pd.to_datetime(c["Report_Date_as_YYYY-MM-DD"])
    out = {}
    for leg, code in CFTC_CODES.items():
        d = c[c["code"] == code].drop_duplicates("asof").set_index("asof").sort_index()
        out[f"mm_net_{leg}"] = d["M_Money_Positions_Long_All"] - d["M_Money_Positions_Short_All"]
        out[f"oi_{leg}"] = d["Open_Interest_All"]
    df = pd.DataFrame(out)
    df.index.name = "asof"
    return df


def cftc_positions(refresh: bool = False) -> pd.DataFrame:
    frames = []
    for f in _cftc_zips(refresh):
        with zipfile.ZipFile(f) as z:
            for name in z.namelist():
                frames.append(pd.read_csv(z.open(name), low_memory=False,
                                          dtype={"CFTC_Contract_Market_Code": str}))
    return _parse_cftc(frames)


# --------------------------------------------------------------------------- point-in-time helpers
def asof_daily(s: pd.Series | pd.DataFrame, available: pd.DatetimeIndex, index: pd.DatetimeIndex):
    """Re-stamp observations on their availability dates, then carry the last known value onto `index`."""
    x = s.copy()
    x.index = pd.DatetimeIndex(available)
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x.reindex(x.index.union(index)).ffill().reindex(index)


def five_year_deviation(weekly: pd.Series, years: int = 5) -> pd.Series:
    """Percent deviation of each week from the mean of the same week in the previous `years` years.

    'Same week' is the week exactly 52*k weeks earlier, k = 1..years; only past data is used.
    """
    w = weekly.sort_index()
    lagged = [w.shift(52 * k, freq="7D").reindex(w.index, method="nearest", tolerance=pd.Timedelta("3D"))
              for k in range(1, years + 1)]
    avg = pd.concat(lagged, axis=1).mean(axis=1, skipna=False)
    return (100 * (w / avg - 1)).rename(f"{w.name}_dev_{years}y")


# --------------------------------------------------------------------------- cleaning
def flag_spikes(x: pd.Series, sigma: float, reversal_frac: float, vol_window: int) -> pd.Series:
    """True on days whose change is > `sigma` x trailing sd AND mostly reverses the next day.

    The trailing sd uses data up to t-1 only. Flags are for manual review; nothing is dropped.
    """
    dx = x.diff()
    sd = dx.rolling(vol_window, min_periods=vol_window // 2).std().shift(1)
    big = dx.abs() > sigma * sd
    nxt = dx.shift(-1)
    reverses = (np.sign(nxt) == -np.sign(dx)) & (nxt.abs() >= reversal_frac * dx.abs())
    return (big & reverses).fillna(False)


def spot_prices(refresh: bool = False) -> pd.DataFrame:
    """Daily spot legs in native units: cl, brent ($/bbl); rb, ho ($/gal). Only days with all four."""
    cfg = load_config()["data"]
    f = cfg["fred_series"]
    px = pd.concat([
        fred(f["wti"], refresh).rename("cl"),
        fred(f["brent"], refresh).rename("brent"),
        fred(f["rb"], refresh).rename("rb"),
        eia(cfg["eia_spot"]["ho"], "d", refresh).rename("ho"),
    ], axis=1)
    return px.dropna().loc[cfg["sample_start"]: cfg["sample_end"]]


def build_clean(refresh: bool = False, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spot legs and spreads, one row per day with all four legs present.

    Returns (prices, spreads). `spreads` holds the three traded spreads ($/bbl), the legs in $/bbl,
    spike-review flags per spread and `flag_apr2020` (the week WTI settled negative).
    """
    cfg = load_config()["data"]
    px = spot_prices(refresh)
    sp = build_spreads(px)
    for name in ["crack321", "brent_wti", "ho_rb"]:
        sp[f"spike_{name}"] = flag_spikes(sp[name], cfg["spike_sigma"], cfg["spike_reversal_frac"],
                                          cfg["spike_vol_window"])
    sp["flag_apr2020"] = False
    for d in cfg["flag_dates"]:
        wk = pd.Timestamp(d).to_period("W-FRI")
        sp.loc[sp.index.to_period("W-FRI") == wk, "flag_apr2020"] = True
    if save:
        px.to_csv(path("clean_dir") / "spot_prices.csv")
        sp.to_csv(path("clean_dir") / "spreads.csv")
    return px, sp


def build_features(index: pd.DatetimeIndex, refresh: bool = False, save: bool = True) -> pd.DataFrame:
    """Raw daily filter inputs, each stamped on its availability date and carried onto `index`.

    Percentiles, z-scores and GARCH-based features are built in the model and regime steps,
    because they need rolling windows.
    """
    cfg = load_config()["data"]
    feats = pd.DataFrame(index=index)

    ovx = fred(cfg["fred_series"]["ovx"], refresh).dropna()
    feats["ovx"] = asof_daily(ovx, ovx.index + pd.Timedelta(days=cfg["ovx_lag_days"]), index)

    g = gpr_daily(refresh)
    feats["gpr"] = asof_daily(g, g.index + pd.Timedelta(days=cfg["gpr_lag_days"]), index)

    weekly = pd.concat({k: eia(v, "w", refresh) for k, v in cfg["eia_weekly"].items()}, axis=1)
    weekly["total_stocks"] = weekly[["crude_ex_spr", "gasoline", "distillate"]].sum(axis=1, min_count=3)
    weekly["inventory_dev_5y"] = five_year_deviation(weekly["total_stocks"].dropna()).reindex(weekly.index)
    for k in ["crude_ex_spr", "cushing", "gasoline", "distillate"]:
        weekly[f"{k}_dev_5y"] = five_year_deviation(weekly[k].dropna()).reindex(weekly.index)
    avail = weekly.index + pd.Timedelta(days=cfg["weekly_stock_lag_days"])
    feats = feats.join(asof_daily(weekly, avail, index))

    cot = cftc_positions(refresh)
    feats = feats.join(asof_daily(cot, cot.index + pd.Timedelta(days=cfg["cftc_release_lag_days"]), index))

    if save:
        feats.to_csv(path("clean_dir") / "features.csv")
        weekly.to_csv(path("clean_dir") / "weekly_stocks.csv")
    return feats


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = path("clean_dir")
    read = lambda n: pd.read_csv(d / n, index_col=0, parse_dates=True)
    return read("spot_prices.csv"), read("spreads.csv")


def load_features() -> pd.DataFrame:
    return pd.read_csv(path("clean_dir") / "features.csv", index_col=0, parse_dates=True)


def data_report(px: pd.DataFrame, sp: pd.DataFrame, feats: pd.DataFrame) -> dict:
    """Summary numbers for the data section of the report."""
    last = sp.index[-1]
    rep = {
        "source": "spot",
        "first_date": sp.index[0].date().isoformat(),
        "last_date": last.date().isoformat(),
        "n_days": len(sp),
        "max_gap_calendar_days": int(sp.index.to_series().diff().dt.days.max()),
        "n_spike_flags": {c: int(sp[c].sum()) for c in sp.columns if c.startswith("spike_")},
        "n_days_flag_apr2020": int(sp["flag_apr2020"].sum()),
        "wti_min": float(px["cl"].min()),
        "latest": {c: round(float(sp[c].iloc[-1]), 2) for c in ["crack321", "brent_wti", "ho_rb"]},
    }
    for c in ["crack321", "brent_wti", "ho_rb"]:
        rep[f"{c}_median"] = round(float(sp[c].median()), 2)
        rep[f"{c}_range"] = [round(float(sp[c].min()), 2), round(float(sp[c].max()), 2)]
    rep["feature_first_valid"] = {c: (feats[c].first_valid_index().date().isoformat()
                                      if feats[c].notna().any() else None) for c in feats.columns}
    rep["feature_missing_after_2010"] = {c: int(feats.loc["2010":, c].isna().sum()) for c in feats.columns}
    return rep
