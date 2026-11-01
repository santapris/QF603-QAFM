"""DESCRIPTIVE: Ljung-Box on the AR(1) innovations of every stored fit and on the pair residuals.

Usage:  .venv/bin/python scripts/run_residual_checks.py [--source futures|spot] [--simulate]
Requires run_model.py (same source). Writes output/tables/04_ljung_box_ar1.csv and
data/derived/rolling_ljung_box.csv; on futures also output/exploratory/pairs/tables/ljung_box*.csv;
with --simulate also output/tables/04_ljung_box_size_power.csv.

Added 2026-10-31, after every registered and exploratory result was seen. Not a gate or a trading rule, so
it changes no verdict. It reads the fits and writes only the files above. Each residual is rebuilt from a
stored fit and must reproduce that fit's AR(1), so these are the residuals the strategy actually used.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from oilrv import diagnostics as dg
from oilrv import pairs as P
from oilrv.config import ROOT, load_config, path, set_source
from oilrv.data import load_source
from oilrv.model import legs_frame
from oilrv.plotting import save_table

ALPHA = 0.05
LAGS = (10, 20)
AR1_DF = 1                                 # the AR coefficient


def spread_rows(src, cfg) -> pd.DataFrame:
    """One row per stored monthly fit of the registered spreads."""
    fits = pd.read_csv(path("derived_dir") / "rolling_fits.csv", index_col=[0, 1], parse_dates=["date"])
    window = cfg["model"]["window"]
    rows = []
    for key in fits.index.get_level_values(0).unique():
        name = key.split("|")[1]
        legs, div = legs_frame(src.legs[name]), cfg["spreads"][name]["divisor"]
        for date, row in fits.loc[key].iterrows():
            p = row.to_dict()
            if not np.isfinite(p["sd"]):       # johansen month with no usable vector
                continue
            r = dg.fitted_residual(dg.window_before(legs, int(p["pos"]), window), p, div)
            if not np.isclose(dg.ar1(r)[0], p["ar1_b"], atol=1e-9):
                raise ValueError(f"{key} {date.date()}: rebuilt residual does not match the stored fit")
            rows.append({"key": key, "date": date, "ar1_b": p["ar1_b"], "half_life": p["half_life"],
                         **{f"ar1_{k}": v for k, v in dg.ljung_box(dg.ar1_residuals(r), LAGS, AR1_DF).items()}})
    return pd.DataFrame(rows)


def pair_rows(cfg) -> pd.DataFrame:
    """One row per monthly pair fit: AR(1) innovations, and the fitted residual itself."""
    data = P.build_data(cfg)
    window = cfg["model"]["window"]
    rows = []
    for name, spec in data["specs"].items():
        legs = data["legs"][name]
        cache = P.PairFitCache(legs, spec, "adjusted_with_trend", None, P.pair_cfg(name, cfg))
        for s in dg.month_starts(legs.index):
            if s < window:
                continue
            p = cache.get(s, window)
            r = dg.fitted_residual(dg.window_before(legs, s, window), p)
            if not np.isclose(dg.ar1(r)[1], p["half_life_raw"], rtol=1e-9):
                raise ValueError(f"{name} {legs.index[s].date()}: rebuilt residual does not match the fit")
            rows.append({"key": name, "date": legs.index[s], "eg_p": p["eg_p"], "half_life": p["half_life_raw"],
                         **{f"ar1_{k}": v for k, v in dg.ljung_box(dg.ar1_residuals(r), LAGS, AR1_DF).items()},
                         **{f"resid_{k}": v for k, v in dg.ljung_box(r, LAGS).items()}})
    return pd.DataFrame(rows)


def summarise(rows: pd.DataFrame, periods: list, eg_p_max: float | None = None) -> pd.DataFrame:
    """Share of windows rejecting at ALPHA and the median p, per key and sub-period."""
    tests = [c for c in rows.columns if "_p_" in c and c != "eg_p"]
    parts = [("all", rows)]
    parts += [(f"{a[:4]}–{(b or str(rows['date'].max().date()))[:4]}",
               rows[(rows["date"] >= a) & (rows["date"] <= (b or rows["date"].max()))]) for a, b in periods]
    if eg_p_max is not None:
        parts.append(("gate open", rows[rows["eg_p"] < eg_p_max]))
    out = {}
    for label, d in parts:
        for key, g in d.groupby("key", sort=False):
            out[(key, label)] = {"months": len(g), **{f"reject|{c}": (g[c] < ALPHA).mean() for c in tests},
                                 **{f"median|{c}": g[c].median() for c in tests}}
    return pd.DataFrame(out).T.rename_axis(["key", "period"])


def size_and_power(n: int = 756, reps: int = 1000, seed: int = 20261031) -> pd.DataFrame:
    """Rejection rates at ALPHA on simulated series shaped like the fits: x_t = phi1 x_{t-1} + phi2 x_{t-2} + e_t,
    with phi1 + phi2 = 0.985 and GARCH(0.10, 0.87) Student-t errors. phi2 = 0 is a true AR(1), so its rate is the
    false-positive rate; phi2 < 0 is dynamics an AR(1) misses, so its rate is the power."""
    rng = np.random.default_rng(seed)
    a, b, burn = 0.10, 0.87, 200

    def noise(nu):
        z = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)
        e, s2 = np.empty(n + burn), 1.0
        for t in range(n + burn):
            e[t] = np.sqrt(s2) * z[t]
            s2 = (1 - a - b) + a * e[t] ** 2 + b * s2
        return e[burn:]

    rows = []
    for nu in (12, 4, 2.5):
        for phi2 in (0.0, -0.05, -0.10, -0.20):
            res = []
            for _ in range(reps):
                e, x = noise(nu), np.zeros(n)
                for t in range(2, n):
                    x[t] = (0.985 - phi2) * x[t - 1] + phi2 * x[t - 2] + e[t]
                res.append(dg.ljung_box(dg.ar1_residuals(x), LAGS, AR1_DF))
            rows.append({"t_df": nu, "phi2": phi2, **(pd.DataFrame(res) < ALPHA).mean().to_dict()})
    return pd.DataFrame(rows).set_index(["t_df", "phi2"])


def main(source: str | None = None, simulate: bool = False) -> dict:
    cfg = load_config()
    source = set_source(source)
    periods = cfg["evaluation"]["sub_periods"]
    rows = spread_rows(load_source(), cfg)
    rows.to_csv(path("derived_dir") / "rolling_ljung_box.csv", index=False)
    table = summarise(rows, periods)
    save_table(table, "04_ljung_box_ar1.csv")
    out = {"source": source, "status": "descriptive; not a gate", "spreads": table.loc[(slice(None), "all"), :].round(3).reset_index(level=1, drop=True).to_dict("index")}
    if source == "futures":
        tab = ROOT / "output" / "exploratory" / "pairs" / "tables"
        pr = pair_rows(cfg)
        pr.to_csv(tab / "ljung_box_monthly.csv", index=False)
        pt = summarise(pr, periods, cfg["exploratory_pairs"]["eg_p_max"])
        pt.to_csv(tab / "ljung_box.csv")
        out["pairs"] = pt.loc[(slice(None), "all"), :].round(3).reset_index(level=1, drop=True).to_dict("index")
    if simulate:
        save_table(size_and_power(), "04_ljung_box_size_power.csv")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["futures", "spot"], default=None)
    ap.add_argument("--simulate", action="store_true", help="also write the size and power simulation (~1 min)")
    args = ap.parse_args()
    print(json.dumps(main(args.source, args.simulate), indent=2, default=float))
