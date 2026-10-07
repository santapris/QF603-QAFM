"""Stage 1 — baseline regressions (PLAN Phase 3, Q1–Q2). Pre-holdout sample unless ``final=True``.

* ``persistence``: VRP_{t+k} = a + ρ_k·VRP_t; compare ρ_k with ρ_1^k (flag if |ρ_k − ρ_1^k| > 2·se(ρ_k)).
* ``ar2``: VRP_{t+1} = a + φ1·VRP_t + φ2·VRP_{t−1}, with the implied k-step coefficient on VRP_t.
* ``local_projections``: VRP_{t+h} = a + β_h·rv_surprise_t + γ_h·VRP_t + e, h in ``stage1.lp_horizons``
  (slow updating: gradual rather than immediate response to realised-variance surprises; D033).
* ``single_predictors``: VRP_{t+k} = a + ρ·VRP_t + β·z(X_t) + e, z standardised on the training window
  only; NW t-stats, moving-block bootstrap p-values [F11], incremental R² over AR(1) on the same rows,
  Holm-adjusted p-values across the predictor × horizon family [F5].
* ``adf_table``: ADF test and AR(1) coefficient per predictor; persistent ones flagged (Stambaugh bias).

Every regression sample goes through ``holdout.restrict`` / ``holdout.guard`` on the target's observation
date [F2].

Run:  python -m src.models.stage1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller

from src.utils.bootstrap import mbb_pvalues
from src.utils.holdout import guard, restrict
from src.utils.io import load_config, project_path
from src.utils.newey_west import nw_lags, ols_nw


def zscore_train(x: pd.Series, train_end) -> pd.Series:
    """Standardise with the mean / sd of the training window only (no look-ahead)."""
    tr = x.loc[:pd.Timestamp(train_end)].dropna()
    return (x - tr.mean()) / tr.std()


def control_series(panel: pd.DataFrame, target: str) -> pd.Series:
    """Lagged control for a regression on ``target``: the target itself, except that the ex-post VRP (a
    payoff known only after t) is controlled by the ex-ante VRP. Ex-post VRP is never a regressor."""
    c = "vrp_exante" if target == "vrp_expost" else target
    if c == "vrp_expost":
        raise ValueError("vrp_expost is never a predictor")
    return panel[c]


def lags_for(k: int, T: int, target: str, cfg: dict) -> int:
    """NW lags from the config rule; ex-post targets span one more month, so use the ex-post rule [F6]."""
    return nw_lags(k, T, expost=target == "vrp_expost", cfg=cfg)


def _sample(panel: pd.DataFrame, target: str, k: int, cols: dict, cfg: dict, final: bool) -> pd.DataFrame:
    df = pd.DataFrame({"y": panel[target].shift(-k), **cols}, index=panel.index)
    obs = None
    if f"obs_date_{target}" in panel:
        obs = panel[f"obs_date_{target}"].shift(-k)
        df = df.loc[obs.notna()]
        obs = obs.loc[df.index]
    df = restrict(df, target, k, final=final, cfg=cfg, obs_dates=obs)
    df = df.dropna()
    guard(df, target, k, final=final, cfg=cfg, obs_dates=None if obs is None else obs.loc[df.index])
    return df


def persistence(panel: pd.DataFrame, target: str, horizons, cfg: dict | None = None,
                final: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    rows = []
    for k in horizons:
        df = _sample(panel, target, k, {"lag": control_series(panel, target)}, cfg, final)
        res = ols_nw(df["y"], df[["lag"]], lags_for(k, len(df), target, cfg))
        rows.append(dict(k=k, rho=res.params["lag"], se=res.se["lag"], t_nw=res.tstat["lag"], r2=res.rsquared,
                         nobs=res.nobs, lags=res.lags, start=df.index.min().date(), end=df.index.max().date()))
    out = pd.DataFrame(rows).set_index("k")
    if 1 in out.index:
        out["rho1_pow_k"] = out.loc[1, "rho"] ** out.index.to_numpy()
        out["differs_from_ar1"] = (out["rho"] - out["rho1_pow_k"]).abs() > 2 * out["se"]
    return out


def ar2(panel: pd.DataFrame, target: str, horizons, cfg: dict | None = None, final: bool = False) -> pd.DataFrame:
    """AR(2) at k = 1 and the implied k-step coefficient on VRP_t (companion-matrix power)."""
    cfg = cfg or load_config()
    df = _sample(panel, target, 1, {"lag1": control_series(panel, target), "lag2": control_series(panel, target).shift(1)}, cfg, final)
    res = ols_nw(df["y"], df[["lag1", "lag2"]], lags_for(1, len(df), target, cfg))
    phi1, phi2 = res.params["lag1"], res.params["lag2"]
    comp = np.array([[phi1, phi2], [1.0, 0.0]])
    rows = [dict(k=k, implied_coef_on_vrp_t=np.linalg.matrix_power(comp, k)[0, 0]) for k in horizons]
    out = pd.DataFrame(rows).set_index("k")
    out["phi1"], out["phi2"], out["t_phi2"], out["nobs"] = phi1, phi2, res.tstat["lag2"], res.nobs
    return out


def local_projections(panel: pd.DataFrame, target: str, shock: str, horizons, cfg: dict | None = None,
                      final: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    rows = []
    for h in horizons:
        df = _sample(panel, target, h, {"shock": panel[shock], "lag": control_series(panel, target)}, cfg, final)
        cols = ["shock"] if h == 0 else ["shock", "lag"]         # at h = 0 the control would be the target itself
        res = ols_nw(df["y"], df[cols], lags_for(max(h, 1), len(df), target, cfg))
        rows.append(dict(h=h, beta=res.params["shock"], se=res.se["shock"], t_nw=res.tstat["shock"],
                         lo95=res.params["shock"] - 1.96 * res.se["shock"],
                         hi95=res.params["shock"] + 1.96 * res.se["shock"], nobs=res.nobs, lags=res.lags))
    return pd.DataFrame(rows).set_index("h")


def single_predictors(panel: pd.DataFrame, target: str, predictors: list[str], horizons,
                      cfg: dict | None = None, final: bool = False, bootstrap: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    train_end = cfg["oos"]["train_end"]
    rows = []
    for x in predictors:
        z = zscore_train(panel[x], train_end)
        for k in horizons:
            df = _sample(panel, target, k, {"lag": control_series(panel, target), "z": z}, cfg, final)
            lags = lags_for(k, len(df), target, cfg)
            full = ols_nw(df["y"], df[["lag", "z"]], lags)
            ar1 = ols_nw(df["y"], df[["lag"]], lags)
            row = dict(predictor=x, k=k, beta=full.params["z"], t_nw=full.tstat["z"], p_nw=full.pvalue["z"],
                       rho=full.params["lag"], r2=full.rsquared, r2_ar1=ar1.rsquared,
                       incr_r2=full.rsquared - ar1.rsquared, nobs=full.nobs, lags=lags,
                       start=df.index.min().date(), end=df.index.max().date())
            if bootstrap:
                row["p_mbb"] = mbb_pvalues(df["y"], df[["lag", "z"]], lags, cfg=cfg)["z"]
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_nw"], method="holm")[1]
    if bootstrap:
        out["p_mbb_holm"] = multipletests(out["p_mbb"], method="holm")[1]
    return out


def adf_table(panel: pd.DataFrame, columns: list[str], cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    pre = panel.loc[:cfg["oos"]["validate_end"]]
    rows = []
    for c in columns:
        x = pre[c].dropna()
        stat, p, lags, nobs, *_ = adfuller(x, autolag="AIC")
        ar1 = x.autocorr(1)
        rows.append(dict(series=c, adf_stat=stat, adf_p=p, adf_lags=lags, nobs=nobs, ar1=ar1,
                         persistent=bool(p > 0.10 or ar1 > cfg["stage1"]["persistent_ar1"])))
    return pd.DataFrame(rows).set_index("series")


def lp_figure(lp: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lp.index, lp["beta"], marker="o")
    ax.fill_between(lp.index, lp["lo95"], lp["hi95"], alpha=0.2, label="95% NW band")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("horizon h (months)"); ax.set_ylabel("response of VRP_{t+h} to rv_surprise_t")
    ax.set_title("Local projections: VRP response to a realised-variance surprise (pre-holdout)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    if cfg["har"]["headline"] is None:
        raise ValueError("har.headline not chosen (🧑 PLAN 2c) — Stage 1 runs only after the headline HAR decision")
    panel = pd.read_parquet(project_path("data", "processed", "panel_monthly.parquet"))
    target, horizons, preds = "vrp_exante", cfg["horizons"], cfg["predictors"]
    tables = {
        "persistence": persistence(panel, target, horizons, cfg),
        "ar2": ar2(panel, target, horizons, cfg),
        "lp_rv_surprise": local_projections(panel, target, "rv_surprise", cfg["stage1"]["lp_horizons"], cfg),
        "single": single_predictors(panel, target, preds, horizons, cfg, bootstrap=True),
        "adf": adf_table(panel, [target] + preds, cfg),
    }
    out = project_path("outputs", "tables")
    for name, t in tables.items():
        t.to_csv(out / f"stage1_{name}.csv", index=name != "single")
    lp_figure(tables["lp_rv_surprise"], project_path("outputs", "figures", "lp_rv_surprise.png"))
    print(f"[ok  ] Stage 1 tables written: {', '.join('stage1_' + n for n in tables)}")
    return tables


if __name__ == "__main__":
    main()
