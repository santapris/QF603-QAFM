"""Stage 5 — does VRP predict S&P 500 excess returns? (PLAN Phase 4, Q5; Bollerslev, Tauchen & Zhou 2009)

* ``insample``: exret_{t→t+k} = a + b·VRP_t + e for k = 1, 3, 6 (cumulative log excess returns, panel columns
  ``exret_k``), with Newey-West and Hodrick (1992) 1B standard errors side by side. The slope is reported per
  unit of VRP and per training-window standard deviation.
* ``oos``: expanding-window forecasts over the validation window; benchmark = historical mean of the same
  k-month excess return on the training rows. Campbell–Thompson (2008) R²_OS with no restriction, with the
  slope restriction (b̂ < 0 → use the historical mean) and with slope + non-negative premium (forecast
  floored at 0); Clark–West test vs the mean.
Predictors: ``stage5.predictors`` (trailing VRP = BTZ convention; ex-ante VRP = extension).
Target observation date = month-end t + k (``target_observation_lag_months.excess_return`` = 0).

Run:  python -m src.models.stage5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.holdout import guard, restrict, training_rows
from src.utils.io import load_config, project_path
from src.utils.newey_west import hodrick_1b, nw_lags, ols_nw

TARGET = "excess_return"


def _frame(panel: pd.DataFrame, x: str, k: int, cfg: dict, final: bool) -> pd.DataFrame:
    df = pd.DataFrame({"y": panel[f"exret_{k}"], "x": panel[x]}, index=panel.index)
    df = restrict(df, TARGET, k, final=final, cfg=cfg).dropna()
    return guard(df, TARGET, k, final=final, cfg=cfg)


def contiguous_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Rows after the last gap of more than one month (Hodrick 1B needs consecutive months)."""
    gaps = df.index.to_series().diff().dt.days > 35
    return df.loc[gaps[gaps].index[-1]:] if gaps.any() else df


def insample(panel: pd.DataFrame, predictors: list[str], horizons, cfg: dict, final: bool = False) -> pd.DataFrame:
    rows = []
    for x in predictors:
        sd = panel[x].loc[:cfg["oos"]["train_end"]].std()
        # one-month returns for Hodrick 1B: a gap-free run of rows whose 1-month return is observed
        r1 = contiguous_tail(_frame(panel, x, 1, cfg, final))
        for k in horizons:
            df = _frame(panel, x, k, cfg, final)
            nw = ols_nw(df["y"], df[["x"]], nw_lags(k, len(df), cfg=cfg))
            hb = hodrick_1b(r1["y"], r1[["x"]], k)
            rows.append(dict(predictor=x, k=k, b=nw.params["x"], b_per_sd=nw.params["x"] * sd,
                             t_nw=nw.tstat["x"], p_nw=nw.pvalue["x"], t_hodrick=hb.tstat["x"], p_hodrick=hb.pvalue["x"],
                             b_hodrick_sample=hb.params["x"], r2=nw.rsquared, nobs=nw.nobs, nobs_hodrick=hb.nobs,
                             start=df.index.min().date(), end=df.index.max().date()))
    return pd.DataFrame(rows)


def oos(panel: pd.DataFrame, predictors: list[str], horizons, cfg: dict, final: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo = pd.Timestamp(cfg["oos"]["validate_end" if final else "train_end"])
    min_train = cfg["stage5"]["min_train_months"]
    fc_rows, met_rows = [], []
    for x in predictors:
        for k in horizons:
            d = pd.DataFrame({"y": panel[f"exret_{k}"], "x": panel[x]}, index=panel.index)
            origins = restrict(d.loc[d.index > lo], TARGET, k, final=final, cfg=cfg).dropna().index
            for origin in origins:
                tr = training_rows(d, origin, TARGET, k, cfg=cfg).dropna()
                if len(tr) < min_train:
                    continue
                A = np.column_stack([np.ones(len(tr)), tr["x"].to_numpy(float)])
                (a, b), *_ = np.linalg.lstsq(A, tr["y"].to_numpy(float), rcond=None)
                mean = float(tr["y"].mean())
                f = a + b * d.loc[origin, "x"]
                f_slope = f if b >= 0 else mean
                fc_rows.append(dict(predictor=x, k=k, date=origin, y=d.loc[origin, "y"], mean=mean, unrestricted=f,
                                    slope_restricted=f_slope, slope_and_premium=max(f_slope, 0.0), b_hat=b))
            fc = pd.DataFrame([r for r in fc_rows if r["predictor"] == x and r["k"] == k])
            if fc.empty:
                continue
            e_m = fc["y"] - fc["mean"]
            for variant in ["unrestricted", "slope_restricted", "slope_and_premium"]:
                e = fc["y"] - fc[variant]
                cw = e_m ** 2 - (e ** 2 - (fc["mean"] - fc[variant]) ** 2)
                cw = pd.Series(cw.to_numpy(), index=fc["date"])
                t = ols_nw(cw, pd.DataFrame(index=cw.index), lags=max(k, 1)).tstat["const"] if cw.std() > 0 else np.nan
                met_rows.append(dict(predictor=x, k=k, variant=variant, n=len(fc),
                                     r2_os=float(1 - (e ** 2).sum() / (e_m ** 2).sum()),
                                     cw_t=t, cw_p=float(stats.norm.sf(t)) if np.isfinite(t) else np.nan,
                                     share_slope_negative=float((fc["b_hat"] < 0).mean())))
    return pd.DataFrame(met_rows), pd.DataFrame(fc_rows)


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    panel = pd.read_parquet(project_path("data", "processed", "panel_monthly.parquet"))
    preds, horizons = cfg["stage5"]["predictors"], cfg["horizons"]
    out = project_path("outputs", "tables")
    ins = insample(panel, preds, horizons, cfg)
    ins.to_csv(out / "stage5_insample.csv", index=False)
    met, fc = oos(panel, preds, horizons, cfg)
    met.to_csv(out / "stage5_oos_validation.csv", index=False)
    fc.to_csv(out / "stage5_oos_forecasts_validation.csv", index=False)
    print("[ok  ] Stage 5 written (stage5_*).\nIn-sample:\n", ins[["predictor", "k", "b_per_sd", "t_nw", "t_hodrick", "r2", "nobs"]].to_string(index=False))
    print("OOS (validation):\n", met.to_string(index=False))
    return ins, met


if __name__ == "__main__":
    main()
