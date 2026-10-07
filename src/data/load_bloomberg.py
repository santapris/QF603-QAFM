"""Phase 1d: load and validate the user's Bloomberg exports (🧑 manual step).

Expected files (D014): ``data/raw/bloomberg/<TICKER with spaces → _>.csv`` with columns ``date,px_last``
(daily PX_LAST), e.g. ``VIX3M_Index.csv``. Required: VIX, VIX3M, VVIX, SKEW, SPXT, PUT, BXM.
Optional: UX1, UX2 (VIX futures).

Validation: required columns present, dates parse, no duplicate dates, numeric positive prices, coverage
through ``sample.end``. Output: ``data/processed/bloomberg_daily.parquet`` (one column per series).

Run:  python -m src.data.load_bloomberg
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_config, project_path, write_parquet

REQUIRED = {"vix": "VIX Index", "vix3m": "VIX3M Index", "vvix": "VVIX Index", "skew": "SKEW Index",
            "spxt": "SPXT Index", "put": "PUT Index", "bxm": "BXM Index"}
OPTIONAL = {"ux1": "UX1 Index", "ux2": "UX2 Index"}


def filename(ticker: str) -> str:
    return ticker.replace(" ", "_") + ".csv"


def read_one(path: Path, name: str) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.strip().lower() for c in df.columns}
    if not {"date", "px_last"} <= cols:
        raise ValueError(f"{path.name}: expected columns date,px_last — found {list(df.columns)}")
    df.columns = [c.strip().lower() for c in df.columns]
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{path.name}: {int(dates.isna().sum())} unparseable dates, e.g. {df.loc[dates.isna(), 'date'].head(3).tolist()}")
    px = pd.to_numeric(df["px_last"], errors="coerce")
    bad = px.isna() | (px <= 0)
    if bad.any():
        raise ValueError(f"{path.name}: {int(bad.sum())} missing/non-positive prices, e.g. rows {df.index[bad][:3].tolist()}")
    s = pd.Series(px.to_numpy(), index=pd.DatetimeIndex(dates, name="date"), name=name).sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"{path.name}: duplicate dates {s.index[s.index.duplicated()][:3].date.tolist()}")
    return s


def load(cfg, directory: Path | None = None, require_all: bool = True) -> tuple[pd.DataFrame, list[str]]:
    directory = directory or project_path("data", "raw", "bloomberg")
    series, missing = {}, []
    for name, ticker in {**REQUIRED, **OPTIONAL}.items():
        path = directory / filename(ticker)
        if not path.exists():
            if name in REQUIRED:
                missing.append(f"{ticker} ({path.name})")
            continue
        series[name] = read_one(path, name)
    if missing and require_all:
        raise FileNotFoundError("Bloomberg exports missing (🧑 PLAN 1d): " + ", ".join(missing))
    end = pd.Timestamp(cfg["sample"]["end"])
    short = [n for n, s in series.items() if s.index.max() < end - pd.Timedelta(days=5)]
    if short and require_all:
        raise ValueError(f"Bloomberg series ending before sample.end {end.date()}: "
                         + ", ".join(f"{n} ({series[n].index.max().date()})" for n in short))
    return pd.DataFrame(series), missing


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    df, _ = load(cfg)
    write_parquet(df, "data/processed/bloomberg_daily.parquet")
    print(f"[ok  ] bloomberg_daily.parquet: {list(df.columns)}, {df.index.min().date()} → {df.index.max().date()}")
    return df


if __name__ == "__main__":
    main()
