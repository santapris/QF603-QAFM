"""Phase 1g: He–Kelly–Manela intermediary capital ratio (🧑 user downloads the monthly file).

Reads the monthly CSV in ``data/raw/hkm/`` (the authors' factor file has a ``yyyymm`` column and an
``intermediary_capital_ratio`` column; a ``date`` column is also accepted). Applies the publication lag
[F4]: the value for month m becomes usable at the month-end trading day of month m + lag.

Output ``data/processed/hkm_monthly.parquet`` indexed by availability month-end:
``hkm`` (lag = ``publication_lag.hkm_months``, headline) and ``hkm_lag1`` (robustness).

Run:  python -m src.data.load_hkm
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.dates import lag_months
from src.utils.io import load_config, project_path, write_parquet


def find_file(directory: Path) -> Path:
    csvs = sorted(p for p in directory.glob("*.csv"))
    if not csvs:
        others = [p.name for p in directory.iterdir() if p.name != ".gitkeep"]
        raise FileNotFoundError(f"no HKM CSV in {directory} (🧑 PLAN 1g); found: {others or 'nothing'}")
    monthly = [p for p in csvs if "month" in p.name.lower()]
    return (monthly or csvs)[0]


def read_capital_ratio(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ratio_cols = [c for c in df.columns if "capital_ratio" in c]
    if not ratio_cols:
        raise ValueError(f"{path.name}: no capital-ratio column; columns are {list(df.columns)}")
    if "yyyymm" in df.columns:
        dates = pd.to_datetime(df["yyyymm"].astype(int).astype(str), format="%Y%m")
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"{path.name}: need a yyyymm or date column; columns are {list(df.columns)}")
    s = pd.Series(pd.to_numeric(df[ratio_cols[0]], errors="raise").to_numpy(),
                  index=dates.dt.to_period("M").dt.to_timestamp("M"), name="hkm").sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"{path.name}: duplicate months")
    return s


def build(cfg, directory: Path | None = None) -> pd.DataFrame:
    path = find_file(directory or project_path("data", "raw", "hkm"))
    raw = read_capital_ratio(path)
    lag = cfg["publication_lag"]["hkm_months"]
    return pd.concat({"hkm": lag_months(raw, lag), "hkm_lag1": lag_months(raw, 1)}, axis=1)


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    df = build(cfg)
    write_parquet(df, "data/processed/hkm_monthly.parquet")
    print(f"[ok  ] hkm_monthly.parquet: {df['hkm'].notna().sum()} months, available "
          f"{df['hkm'].first_valid_index().date()} → {df['hkm'].last_valid_index().date()}")
    return df


if __name__ == "__main__":
    main()
