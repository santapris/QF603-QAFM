"""Recursive HAR-RV forecasts of one-month realised variance (Corsi 2009; PLAN 2c).

Inputs are daily realised variances ``rv`` (daily, *not* annualised) on the trading-day index.
All quantities are annualised with 252 trading days:

* target at day s:   y_s  = (252/h) · Σ_{j=1..h} RV_{s+j}                 (h = rv_trading_days = 21)
* regressors at s:   RV_d = 252·RV_s,  RV_w = 252·mean(RV_{s-4..s}),  RV_m = 252·mean(RV_{s-21..s})

At each forecast origin t (a month-end) the model is fitted by OLS on rows s with s + h ≤ t, i.e. only
targets fully observed by t, then E_t[RV] is formed from the regressors at t. Nothing after t is used.

Variants
* ``level``: y on [1, RV_d, RV_w, RV_m].
* ``log``:   log y on [1, log RV_d, log RV_w, log RV_m]; forecast exp(ŷ + σ̂²/2), σ̂² = training residual
             variance. Daily RV is floored at ``har.rv_floor`` before logs.
* ``iv``:    level HAR plus implied variance IV_s (Bekaert & Hoerova 2014), IV = daily VIX²/10000.

Evaluation (pre-holdout only): Mincer-Zarnowitz regression of realised on forecast, QLIKE, MSE,
Diebold-Mariano on QLIKE vs the level variant, and the share of positive ex-ante VRP each variant implies.

Outputs: ``data/processed/har_forecasts.parquet`` (all variants, all month-ends; forecasts use only past
data), ``outputs/tables/har_evaluation.csv``.

Run:  python -m src.measures.har
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.dates import month_end_trading_days
from src.utils.holdout import restrict
from src.utils.io import load_config, project_path, write_parquet
from src.utils.newey_west import nw_cov, ols_nw

ANNUAL = 252
VARIANTS = ("level", "log", "iv")


def har_design(rv: pd.Series) -> pd.DataFrame:
    """Annualised HAR regressors dated s (use RV up to and including s)."""
    return pd.DataFrame({"rv_d": ANNUAL * rv,
                         "rv_w": ANNUAL * rv.rolling(5).mean(),
                         "rv_m": ANNUAL * rv.rolling(22).mean()}, index=rv.index)


def har_target(rv: pd.Series, horizon: int) -> pd.Series:
    """Annualised realised variance over the ``horizon`` trading days after s (NaN near the end)."""
    return (ANNUAL / horizon) * rv.rolling(horizon).sum().shift(-horizon)


def _design(rv: pd.Series, horizon: int, variant: str, iv_daily: pd.Series | None, rv_floor: float):
    if variant not in VARIANTS:
        raise ValueError(f"unknown HAR variant {variant!r}; choose from {VARIANTS}")
    if variant == "log":
        X = np.log(har_design(rv.clip(lower=rv_floor)))
        y = np.log(har_target(rv.clip(lower=rv_floor), horizon))
    else:
        X = har_design(rv)
        y = har_target(rv, horizon)
    if variant == "iv":
        if iv_daily is None:
            raise ValueError("variant 'iv' needs iv_daily (annualised implied variance by trading day)")
        X["iv"] = iv_daily.reindex(rv.index)
    return X, y


def recursive_har(rv: pd.Series, origins, horizon: int = 21, variant: str = "level",
                  min_obs: int = 250, iv_daily: pd.Series | None = None, rv_floor: float = 1e-8) -> pd.DataFrame:
    """Out-of-sample HAR forecast of annualised RV over the next ``horizon`` days, at each origin.

    ``origins`` must be trading days in ``rv.index``. Returns one row per origin with the forecast
    (annualised variance), the number of training rows and the coefficients.
    """
    rv = rv.sort_index().astype(float)
    X, y = _design(rv, horizon, variant, iv_daily, rv_floor)
    names = ["const"] + list(X.columns)
    Xv = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    yv = y.to_numpy(float)
    valid = np.isfinite(Xv).all(axis=1)

    rows = []
    for t in pd.DatetimeIndex(origins):
        p = rv.index.get_loc(t)
        train = valid.copy()
        train[p - horizon + 1:] = False               # s + horizon <= p
        train &= np.isfinite(yv)
        n = int(train.sum())
        if n < min_obs or not valid[p]:
            rows.append(dict(date=t, har_fcst=np.nan, n_train=n))
            continue
        beta, *_ = np.linalg.lstsq(Xv[train], yv[train], rcond=None)
        fit = float(Xv[p] @ beta)
        if variant == "log":
            resid = yv[train] - Xv[train] @ beta
            fit = float(np.exp(fit + resid.var(ddof=len(beta)) / 2))
        rows.append(dict(date=t, har_fcst=fit, n_train=n, **{f"b_{k}": b for k, b in zip(names, beta)}))
    return pd.DataFrame(rows).set_index("date")


# ------------------------------------------------------------------ evaluation
def qlike(realised: pd.Series, forecast: pd.Series) -> pd.Series:
    """Patton (2011) QLIKE loss RV/F − log(RV/F) − 1 (0 for a perfect forecast; needs F > 0)."""
    ratio = realised / forecast
    return ratio - np.log(ratio) - 1


def evaluate(forecasts: pd.DataFrame, realised: pd.Series, iv: pd.Series | None = None,
             base: str = "level") -> pd.DataFrame:
    """Forecast evaluation table, one row per variant (column of ``forecasts``)."""
    rows = []
    loss = {}
    for v in forecasts.columns:
        df = pd.concat({"rv": realised, "f": forecasts[v]}, axis=1).dropna()
        pos = df[df["f"] > 0]
        mz = ols_nw(df["rv"], df[["f"]], lags=1)
        # joint Wald test a = 0, b = 1 with the NW covariance
        X = np.column_stack([np.ones(len(df)), df["f"].to_numpy()])
        cov = nw_cov(X, mz.resid.to_numpy(), 1)
        d = mz.params.to_numpy() - np.array([0.0, 1.0])
        try:
            wald = float(d @ np.linalg.solve(cov, d))
        except np.linalg.LinAlgError:
            wald = np.nan
        loss[v] = qlike(pos["rv"], pos["f"])
        row = dict(variant=v, nobs=len(df), start=df.index.min().date(), end=df.index.max().date(),
                   mz_a=mz.params["const"], mz_b=mz.params["f"], mz_t_b_eq_1=(mz.params["f"] - 1) / mz.se["f"],
                   mz_wald_p=float(np.exp(-wald / 2)), mz_r2=mz.rsquared,
                   mse=float(((df["rv"] - df["f"]) ** 2).mean()), qlike=float(loss[v].mean()),
                   n_nonpositive=int((df["f"] <= 0).sum()), mean_bias=float((df["rv"] - df["f"]).mean()))
        if iv is not None:
            vrp = (iv.reindex(df.index) - df["f"]).dropna()
            row["share_vrp_exante_pos"] = float((vrp > 0).mean())
            row["min_vrp_exante"] = float(vrp.min())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("variant")
    if base in loss:
        for v in out.index:
            dl = (loss[v] - loss[base]).dropna()
            if v == base or dl.abs().sum() == 0:
                out.loc[v, "dm_t_qlike_vs_" + base] = np.nan
                continue
            res = ols_nw(dl, pd.DataFrame(index=dl.index), lags=1, add_const=True)
            out.loc[v, "dm_t_qlike_vs_" + base] = res.tstat["const"]
    return out


# ------------------------------------------------------------------ pipeline
def load_iv_daily(cfg) -> pd.Series:
    """Daily implied variance (VIX²/10000) for the IV-augmented variant, from the configured source."""
    src = cfg["har"]["iv_daily_source"]
    if src == "VIXCLS":
        from src.data.pull_fred import load_raw
        vix = load_raw("VIXCLS")
    else:
        vix = pd.read_csv(project_path("data", "raw", "bloomberg", f"{src.replace(' ', '_')}.csv"),
                          parse_dates=["date"]).set_index("date")["px_last"]
    return (vix.astype(float) ** 2 / 10000).rename("iv_daily")


def build(rv: pd.Series, cfg) -> pd.DataFrame:
    h = cfg["vrp_tenor"]["rv_trading_days"]
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    origins = month_end_trading_days(start - pd.DateOffset(months=1), end)
    iv_daily = load_iv_daily(cfg)
    out = {}
    for v in cfg["har"]["variants"]:
        f = recursive_har(rv, origins, horizon=h, variant=v, min_obs=cfg["har"]["min_train_days"],
                          iv_daily=iv_daily if v == "iv" else None, rv_floor=cfg["har"]["rv_floor"])
        out[f"har_{v}"] = f["har_fcst"]
        out[f"n_train_{v}"] = f["n_train"]
    fc = pd.DataFrame(out)
    r = har_target(rv.sort_index().astype(float), h).reindex(origins)
    fc["rv_fwd"] = r
    fc["obs_date_rv_fwd"] = pd.DatetimeIndex([rv.index[rv.index.get_loc(t) + h] if rv.index.get_loc(t) + h < len(rv)
                                              else pd.NaT for t in origins])
    return fc


def evaluation_table(fc: pd.DataFrame, cfg, iv: pd.Series | None = None) -> pd.DataFrame:
    """Evaluate on origins whose realised target is observed by oos.validate_end (never the holdout)."""
    df = fc.loc[pd.Timestamp(cfg["sample"]["start"]):].dropna(subset=["obs_date_rv_fwd"])
    df = restrict(df, "vrp_expost", 0, cfg=cfg, obs_dates=df["obs_date_rv_fwd"])
    cols = [c for c in df.columns if c.startswith("har_")]
    periods = {"2008-2020 (pre-holdout)": df,
               f"train (≤ {cfg['oos']['train_end']})": df.loc[:cfg["oos"]["train_end"]],
               "validate": df.loc[pd.Timestamp(cfg["oos"]["train_end"]) + pd.Timedelta(days=1):]}
    tables = []
    for name, d in periods.items():
        t = evaluate(d[cols].rename(columns=lambda c: c[4:]), d["rv_fwd"], iv=iv)
        tables.append(t.assign(period=name))
    return pd.concat(tables).reset_index().set_index(["period", "variant"])


def main(cfg=None, force=False, rv: pd.Series | None = None, save: bool = True):
    cfg = cfg or load_config()
    if rv is None:
        rv = pd.read_parquet(project_path("data", "processed", "rv_daily.parquet"))["rv"]
    fc = build(rv, cfg)
    iv = pd.read_parquet(project_path("data", "processed", "iv_monthly.parquet"))["iv"]
    table = evaluation_table(fc, cfg, iv=iv)
    if save:
        write_parquet(fc, "data/processed/har_forecasts.parquet")
        table.to_csv(project_path("outputs", "tables", "har_evaluation.csv"))
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(table[["nobs", "mz_a", "mz_b", "mz_wald_p", "mz_r2", "mse", "qlike", "dm_t_qlike_vs_level",
                     "n_nonpositive", "share_vrp_exante_pos", "min_vrp_exante"]].to_string())
    return fc, table


if __name__ == "__main__":
    main()
