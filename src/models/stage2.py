"""Stage 2 — multivariate (PLAN Phase 3, Q2). In-sample / descriptive, pre-holdout unless ``final=True``.

* ``collinearity``: correlation matrix and VIFs of lagged VRP + predictors.
* ``full_ols``: VRP_{t+k} = a + ρ·VRP_t + Σ β_j·z(X_j,t) + e (reference; NW + block-bootstrap p-values).
* ``pca``: PCA on the predictors standardised with training-window moments, fitted on the pre-holdout
  complete cases; loadings (sign-normalised: largest |loading| positive), explained variance, and the
  top-loading variables per PC to support economic labels (🧑 user confirms labels); regressions of
  VRP_{t+k} on VRP_t + first m PCs, m = 1..``pca_components_reported``.
* ``ridge_cv``: ridge on [VRP_t, predictors] with λ chosen by expanding-window time-series CV — at each CV
  origin t the scaler and ridge are fitted only on rows whose target is observed by t
  (``holdout.training_rows``), then VRP_{t+k} is predicted from row t. Never random k-fold.
* ``ridge_path``: coefficients across the λ grid on the full pre-holdout sample.

Rows with any missing predictor are dropped (listwise); ``nobs`` is reported everywhere.

Run:  python -m src.models.stage2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.models.stage1 import _sample, control_series, lags_for, zscore_train
from src.utils.bootstrap import mbb_pvalues
from src.utils.holdout import training_rows
from src.utils.io import load_config, project_path
from src.utils.newey_west import nw_lags, ols_nw


def _design(panel: pd.DataFrame, target: str, k: int, predictors: list[str], cfg: dict, final: bool,
            standardise: bool = True) -> pd.DataFrame:
    """y = target_{t+k}, lag = target_t, predictors (z-scored on the training window), complete cases only."""
    cols = {"lag": control_series(panel, target)}
    for x in predictors:
        cols[x] = zscore_train(panel[x], cfg["oos"]["train_end"]) if standardise else panel[x]
    return _sample(panel, target, k, cols, cfg, final)


def collinearity(panel: pd.DataFrame, target: str, predictors: list[str], cfg: dict,
                 final: bool = False) -> pd.DataFrame:
    df = _design(panel, target, 1, predictors, cfg, final).drop(columns="y").rename(columns={"lag": target})
    corr = df.corr()
    X = np.column_stack([np.ones(len(df)), df.to_numpy()])
    corr["vif"] = [variance_inflation_factor(X, i + 1) for i in range(df.shape[1])]
    corr["nobs"] = len(df)
    return corr


def full_ols(panel: pd.DataFrame, target: str, predictors: list[str], horizons, cfg: dict,
             final: bool = False, bootstrap: bool = True) -> pd.DataFrame:
    rows = []
    for k in horizons:
        df = _design(panel, target, k, predictors, cfg, final)
        lags = lags_for(k, len(df), target, cfg)
        res = ols_nw(df["y"], df.drop(columns="y"), lags)
        pb = mbb_pvalues(df["y"], df.drop(columns="y"), lags, cfg=cfg) if bootstrap else None
        ar1 = ols_nw(df["y"], df[["lag"]], lags)
        for name in res.params.index:
            rows.append(dict(k=k, term=name, coef=res.params[name], t_nw=res.tstat[name], p_nw=res.pvalue[name],
                             p_mbb=np.nan if pb is None else pb[name], r2=res.rsquared,
                             incr_r2_over_ar1=res.rsquared - ar1.rsquared, nobs=res.nobs, lags=lags))
    return pd.DataFrame(rows)


def pca(panel: pd.DataFrame, target: str, predictors: list[str], horizons, cfg: dict,
        final: bool = False) -> dict[str, pd.DataFrame]:
    m_max = cfg["stage2"]["pca_components_reported"]
    base = _design(panel, target, 0, predictors, cfg, final)          # complete predictor rows, pre-holdout
    Z = base[predictors]
    model = PCA().fit(Z.to_numpy())
    load = pd.DataFrame(model.components_.T, index=predictors,
                        columns=[f"PC{i + 1}" for i in range(len(predictors))])
    load = load * np.sign(load.loc[load.abs().idxmax(), :].to_numpy().diagonal())   # largest |loading| > 0
    var = pd.DataFrame({"explained": model.explained_variance_ratio_,
                        "cumulative": np.cumsum(model.explained_variance_ratio_)}, index=load.columns)
    var["top_loadings"] = [", ".join(f"{i} ({load.loc[i, pc]:+.2f})" for i in load[pc].abs().nlargest(3).index)
                           for pc in load.columns]

    scores_all = pd.DataFrame(
        (zscore_frame(panel, predictors, cfg).to_numpy() - model.mean_) @ load.to_numpy(),
        index=panel.index, columns=load.columns)
    rows = []
    for k in horizons:
        for m in range(1, m_max + 1):
            pcs = {f"PC{i}": scores_all[f"PC{i}"] for i in range(1, m + 1)}
            df = _sample(panel, target, k, {"lag": control_series(panel, target), **pcs}, cfg, final)
            res = ols_nw(df["y"], df.drop(columns="y"), lags_for(k, len(df), target, cfg))
            for name in pcs:
                rows.append(dict(k=k, m=m, term=name, coef=res.params[name], t_nw=res.tstat[name],
                                 r2=res.rsquared, nobs=res.nobs))
    return {"loadings": load, "variance": var, "regressions": pd.DataFrame(rows)}


def zscore_frame(panel: pd.DataFrame, predictors: list[str], cfg: dict) -> pd.DataFrame:
    return pd.DataFrame({x: zscore_train(panel[x], cfg["oos"]["train_end"]) for x in predictors})


def alpha_grid(cfg: dict) -> np.ndarray:
    lo, hi, n = cfg["stage2"]["ridge_log10_alpha"]
    return np.logspace(lo, hi, int(n))


def expanding_folds(data: pd.DataFrame, target: str, k: int, min_train: int, cfg: dict, obs_dates=None):
    """Yield (origin, train_index) for expanding-window CV: train rows have targets observed by origin."""
    for origin in data.index:
        tr = training_rows(data, origin, target, k, cfg=cfg,
                           obs_dates=None if obs_dates is None else obs_dates.loc[data.index])
        tr = tr.loc[tr.index < origin]
        if len(tr) >= min_train:
            yield origin, tr.index


def ridge_cv(panel: pd.DataFrame, target: str, predictors: list[str], horizons, cfg: dict,
             final: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window CV MSE per (k, λ) and the chosen λ per k (raw predictors, scaled inside each fold)."""
    alphas = alpha_grid(cfg)
    cv_rows, best = [], []
    for k in horizons:
        df = _design(panel, target, k, predictors, cfg, final, standardise=False)
        X, y = df.drop(columns="y"), df["y"]
        sq = {a: [] for a in alphas}
        for origin, tr_idx in expanding_folds(df, target, k, cfg["stage2"]["cv_min_train_months"], cfg):
            for a in alphas:
                model = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X.loc[tr_idx], y.loc[tr_idx])
                sq[a].append((y.loc[origin] - model.predict(X.loc[[origin]])[0]) ** 2)
        for a in alphas:
            cv_rows.append(dict(k=k, alpha=a, cv_mse=float(np.mean(sq[a])), n_folds=len(sq[a])))
        tab = pd.DataFrame([r for r in cv_rows if r["k"] == k])
        b = tab.loc[tab["cv_mse"].idxmin()]
        best.append(dict(k=k, alpha=b["alpha"], cv_mse=b["cv_mse"], n_folds=int(b["n_folds"]), nobs=len(df)))
    return pd.DataFrame(cv_rows), pd.DataFrame(best).set_index("k")


def ridge_path(panel: pd.DataFrame, target: str, predictors: list[str], k: int, cfg: dict,
               final: bool = False) -> pd.DataFrame:
    df = _design(panel, target, k, predictors, cfg, final, standardise=False)
    X, y = df.drop(columns="y"), df["y"]
    rows = []
    for a in alpha_grid(cfg):
        model = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X, y)
        rows.append(dict(alpha=a, **dict(zip(X.columns, model[-1].coef_))))
    return pd.DataFrame(rows).set_index("alpha")


def figures(pca_out: dict, paths: dict[int, pd.DataFrame], best: pd.DataFrame, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = pca_out["variance"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(v) + 1), v["explained"]); ax.plot(range(1, len(v) + 1), v["cumulative"], "k-o")
    ax.set_xlabel("principal component"); ax.set_ylabel("share of variance"); ax.set_title("Scree plot (pre-holdout)")
    fig.tight_layout(); fig.savefig(out_dir / "stage2_scree.png", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 4), sharey=True)
    for ax, (k, p) in zip(np.atleast_1d(axes), paths.items()):
        for c in p.columns:
            ax.plot(p.index, p[c], label=c)
        ax.axvline(best.loc[k, "alpha"], color="k", ls="--", lw=0.8)
        ax.set_xscale("log"); ax.set_title(f"Ridge path, k = {k}"); ax.set_xlabel("λ")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "stage2_ridge_path.png", dpi=120); plt.close(fig)


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    if cfg["har"]["headline"] is None:
        raise ValueError("har.headline not chosen (🧑 PLAN 2c) — Stage 2 runs only after the headline HAR decision")
    panel = pd.read_parquet(project_path("data", "processed", "panel_monthly.parquet"))
    target, horizons, preds = "vrp_exante", cfg["horizons"], cfg["predictors"]
    out = project_path("outputs", "tables")
    collinearity(panel, target, preds, cfg).to_csv(out / "stage2_collinearity.csv")
    full_ols(panel, target, preds, horizons, cfg).to_csv(out / "stage2_ols.csv", index=False)
    p = pca(panel, target, preds, horizons, cfg)
    p["loadings"].to_csv(out / "stage2_pca_loadings.csv")
    p["variance"].to_csv(out / "stage2_pca_variance.csv")
    p["regressions"].to_csv(out / "stage2_pca_regressions.csv", index=False)
    cv, best = ridge_cv(panel, target, preds, horizons, cfg)
    cv.to_csv(out / "stage2_ridge_cv.csv", index=False)
    best.to_csv(out / "stage2_ridge_best.csv")
    paths = {k: ridge_path(panel, target, preds, k, cfg) for k in horizons}
    pd.concat(paths, names=["k"]).to_csv(out / "stage2_ridge_path.csv")
    figures(p, paths, best, project_path("outputs", "figures"))
    print("[ok  ] Stage 2 tables written (stage2_*). 🧑 Proposed PC labels need confirmation:")
    print(p["variance"][["explained", "cumulative", "top_loadings"]].head(cfg["stage2"]["pca_components_reported"]).to_string())
    return p, best


if __name__ == "__main__":
    main()
