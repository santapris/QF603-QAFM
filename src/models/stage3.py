"""Stage 3 — state dependence and stability (PLAN Phase 3, Q3). Pre-holdout unless ``final=True``.

* ``interactions``: VRP_{t+k} = a + ρ·VRP_t + β·z_t + γ·z_t·D_t + δ·D_t + e with D_t = 1{VIX_t > p80}, the
  80th percentile computed on the training window only; and the continuous version with VIX_t (z-scored
  on the training window) in place of D_t. Reports the calm and stressed slopes (β, β + γ).
* ``rolling_betas``: 60-month rolling single-predictor regressions (k = 1) with ±1.96 NW-s.e. bands.
* ``sup_wald``: Andrews (1993) sup-Wald test for a break in all coefficients at an unknown date,
  15% trimming, White-robust Wald at each candidate date; headline p-value from Hansen's (2000)
  fixed-regressor wild bootstrap, plus the simulated asymptotic p-value (D035).
* ``ex_crisis``: Stage 1 persistence / single predictors and Stage 2 full OLS without 2008–2009.
* ``regime_note``: which predictors change slope in the high-VIX regime (input for Phase 6 sizing caps).

Run:  python -m src.models.stage3
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.models.stage1 import _sample, control_series, lags_for, persistence, single_predictors, zscore_train
from src.models.stage2 import full_ols
from src.utils.io import load_config, project_path
from src.utils.newey_west import _fit, nw_cov, nw_lags, ols_nw


# ------------------------------------------------------------------ regime interactions
def high_vol_dummy(vix: pd.Series, cfg: dict) -> tuple[pd.Series, float]:
    thr = float(np.nanpercentile(vix.loc[:cfg["oos"]["train_end"]], cfg["regimes"]["high_vol_vix_percentile"]))
    return (vix > thr).astype(float).where(vix.notna()), thr


def interactions(panel: pd.DataFrame, target: str, predictors: list[str], horizons, cfg: dict,
                 final: bool = False) -> pd.DataFrame:
    D, thr = high_vol_dummy(panel["vix"], cfg)
    vz = zscore_train(panel["vix"], cfg["oos"]["train_end"])
    rows = []
    for x in predictors:
        z = zscore_train(panel[x], cfg["oos"]["train_end"])
        for k in horizons:
            for kind, s in [("dummy_p80", D), ("continuous_vix", vz)]:
                cols = {"lag": control_series(panel, target), "z": z, "zs": z * s, "s": s}
                if x == "vix" and kind == "continuous_vix":
                    cols.pop("s")                               # z·s is VIX², s would duplicate z
                df = _sample(panel, target, k, cols, cfg, final)
                res = ols_nw(df["y"], df.drop(columns="y"), lags_for(k, len(df), target, cfg))
                row = dict(predictor=x, k=k, kind=kind, beta_calm=res.params["z"], t_calm=res.tstat["z"],
                           gamma=res.params["zs"], t_gamma=res.tstat["zs"], p_gamma=res.pvalue["zs"],
                           nobs=res.nobs, threshold_vix=thr if kind == "dummy_p80" else np.nan)
                if kind == "dummy_p80":
                    cov = nw_cov(np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()]),
                                 res.resid.to_numpy(), res.lags)
                    names = ["const"] + list(df.drop(columns="y").columns)
                    i, j = names.index("z"), names.index("zs")
                    se = np.sqrt(cov[i, i] + cov[j, j] + 2 * cov[i, j])
                    row.update(beta_stress=res.params["z"] + res.params["zs"],
                               t_stress=(res.params["z"] + res.params["zs"]) / se,
                               share_high_vol=float(df["s"].mean()))
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rolling betas
def rolling_betas(panel: pd.DataFrame, target: str, predictors: list[str], cfg: dict,
                  final: bool = False, k: int = 1) -> pd.DataFrame:
    window = cfg["rolling_window_months"]
    rows = []
    for x in [None] + list(predictors):
        cols = {"lag": control_series(panel, target)}
        if x is not None:
            cols["z"] = zscore_train(panel[x], cfg["oos"]["train_end"])
        df = _sample(panel, target, k, cols, cfg, final)
        term = "lag" if x is None else "z"
        for end in range(window, len(df) + 1):
            w = df.iloc[end - window:end]
            res = ols_nw(w["y"], w.drop(columns="y"), lags_for(k, window, target, cfg))
            rows.append(dict(spec="AR(1)" if x is None else x, term="vrp_lag" if x is None else x,
                             window_end=w.index[-1], beta=res.params[term], se=res.se[term]))
    out = pd.DataFrame(rows)
    out["lo95"], out["hi95"] = out["beta"] - 1.96 * out["se"], out["beta"] + 1.96 * out["se"]
    return out


# ------------------------------------------------------------------ Andrews sup-Wald
@lru_cache(maxsize=None)
def supwald_null(p: int, trim: float, reps: int, grid: int, seed: int) -> np.ndarray:
    """Simulated draws of sup_τ ||B_p(τ) − τ B_p(1)||² / (τ(1−τ)) over τ ∈ [trim, 1 − trim]."""
    rng = np.random.default_rng(seed)
    tau = np.arange(1, grid + 1) / grid
    keep = (tau >= trim) & (tau <= 1 - trim)
    out = np.empty(reps)
    chunk = 500
    for i in range(0, reps, chunk):
        m = min(chunk, reps - i)
        W = np.cumsum(rng.standard_normal((m, grid, p)) / np.sqrt(grid), axis=1)
        bridge = W - tau[None, :, None] * W[:, -1:, :]
        stat = (bridge ** 2).sum(axis=2)[:, keep] / (tau[keep] * (1 - tau[keep]))[None, :]
        out[i:i + m] = stat.max(axis=1)
    return np.sort(out)


def _wald_stats(Z: np.ndarray, Y: np.ndarray, p: int) -> np.ndarray:
    """White-robust Wald statistics for the post-break block of Z (last p columns), one per column of Y."""
    A = np.linalg.inv(Z.T @ Z)
    B = A @ Z.T @ Y                                         # (2p, R)
    U = Y - Z @ B
    meat = np.einsum("ti,tj,tr->rij", Z, Z, U ** 2)          # (R, 2p, 2p)
    V = A[None] @ meat @ A[None]
    d = B[p:].T                                             # (R, p)
    Vd = V[:, p:, p:]
    return np.einsum("ri,ri->r", d, np.linalg.solve(Vd, d[..., None])[..., 0])


def sup_wald(y: pd.Series, X: pd.DataFrame, cfg: dict, reps: int | None = None) -> dict:
    """Andrews (1993) sup-Wald for a break in all coefficients (incl. intercept) at an unknown date.

    White-robust Wald at each candidate date in [trim, 1 − trim] (k = 1: no overlap). p-values:
    ``p_value`` from Hansen's (2000) fixed-regressor wild bootstrap (Rademacher; regressors held fixed,
    residuals from the no-break fit) — the headline, since the asymptotic test over-rejects with ~150
    observations (D035); ``p_asymptotic`` from the simulated limiting distribution for reference.
    """
    trim = cfg["stage3"]["break_trim"]
    reps = reps or cfg["stage3"]["supwald_boot_reps"]
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    yv = y.to_numpy(float)
    T, p = Xc.shape
    breaks = list(range(int(np.floor(trim * T)), int(np.ceil((1 - trim) * T)) + 1))
    beta0, u0 = _fit(yv, Xc)
    rng = np.random.default_rng(cfg["seed"])
    eta = rng.choice([-1.0, 1.0], size=(T, reps))
    Ystar = (Xc @ beta0)[:, None] + u0[:, None] * eta
    Y = np.column_stack([yv, Ystar])                       # column 0 = data
    sup = np.full(reps + 1, -np.inf)
    arg = 0
    for b in breaks:
        post = (np.arange(T) >= b).astype(float)[:, None]
        w = _wald_stats(np.column_stack([Xc, Xc * post]), Y, p)
        if w[0] > sup[0]:
            arg = b
        sup = np.maximum(sup, w)
    null = supwald_null(p, trim, cfg["stage3"]["supwald_sim_reps"], cfg["stage3"]["supwald_sim_grid"], cfg["seed"])
    return dict(sup_wald=float(sup[0]), break_date=y.index[arg].date(), p_value=float((sup[1:] >= sup[0]).mean()),
                p_asymptotic=float((null >= sup[0]).mean()), crit_5pct_asymptotic=float(np.quantile(null, 0.95)),
                crit_5pct_bootstrap=float(np.quantile(sup[1:], 0.95)), n_params=p, nobs=T, boot_reps=reps)


def break_tests(panel: pd.DataFrame, target: str, predictors: list[str], cfg: dict, final: bool = False):
    main = cfg["stage3"]["main_predictors"]
    specs = {"AR(1)": []} if main is None else {}
    specs["AR(1) + " + ("all predictors" if main is None else "+".join(main))] = main or predictors
    rows = []
    for name, preds in specs.items():
        cols = {"lag": control_series(panel, target), **{x: zscore_train(panel[x], cfg["oos"]["train_end"]) for x in preds}}
        df = _sample(panel, target, 1, cols, cfg, final)
        rows.append(dict(spec=name, **sup_wald(df["y"], df.drop(columns="y"), cfg)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ robustness & notes
def ex_crisis(panel: pd.DataFrame, target: str, predictors: list[str], horizons, cfg: dict) -> dict[str, pd.DataFrame]:
    lo, hi = (pd.Timestamp(d) for d in cfg["stage3"]["exclude_crisis"])
    sub = panel.loc[panel.index > hi] if lo <= panel.index.min() else panel.loc[(panel.index < lo) | (panel.index > hi)]
    return {"persistence": persistence(sub, target, horizons, cfg),
            "single": single_predictors(sub, target, predictors, horizons, cfg),
            "ols": full_ols(sub, target, predictors, horizons, cfg, bootstrap=False)}


def regime_note(inter: pd.DataFrame) -> pd.DataFrame:
    d = inter[(inter["kind"] == "dummy_p80") & (inter["k"] == 1)]
    return d.assign(unstable_in_high_vix=d["p_gamma"] < 0.05)[
        ["predictor", "beta_calm", "beta_stress", "t_gamma", "p_gamma", "unstable_in_high_vix", "share_high_vol"]]


def rolling_figures(rb: pd.DataFrame, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = list(rb["spec"].unique())
    n = len(specs)
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(14, 3.2 * int(np.ceil(n / 3))), squeeze=False)
    for ax, s in zip(axes.ravel(), specs):
        d = rb[rb["spec"] == s]
        ax.plot(d["window_end"], d["beta"]); ax.fill_between(d["window_end"], d["lo95"], d["hi95"], alpha=0.2)
        ax.axhline(0, color="k", lw=0.5); ax.set_title(d["term"].iloc[0], fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Rolling 60-month betas, k = 1 (window end; pre-holdout)")
    fig.tight_layout(); fig.savefig(out_dir / "rolling_betas_all.png", dpi=110); plt.close(fig)


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    if cfg["har"]["headline"] is None:
        raise ValueError("har.headline not chosen (🧑 PLAN 2c) — Stage 3 runs only after the headline HAR decision")
    panel = pd.read_parquet(project_path("data", "processed", "panel_monthly.parquet"))
    target, horizons, preds = "vrp_exante", cfg["horizons"], cfg["predictors"]
    out = project_path("outputs", "tables")
    inter = interactions(panel, target, preds, horizons, cfg)
    inter.to_csv(out / "stage3_interactions.csv", index=False)
    rb = rolling_betas(panel, target, preds, cfg)
    rb.to_csv(out / "stage3_rolling_betas.csv", index=False)
    rolling_figures(rb, project_path("outputs", "figures"))
    bt = break_tests(panel, target, preds, cfg)
    bt.to_csv(out / "stage3_break_test.csv", index=False)
    for name, t in ex_crisis(panel, target, preds, horizons, cfg).items():
        t.to_csv(out / f"stage3_ex2008_09_{name}.csv", index=name == "persistence")
    note = regime_note(inter)
    note.to_csv(out / "stage3_regime_note.csv", index=False)
    print("[ok  ] Stage 3 tables written (stage3_*). Break tests:\n", bt.to_string(index=False))
    print("High-VIX slope changes (k = 1) — input to Phase 6 multiplier cap:\n", note.to_string(index=False))
    return inter, rb, bt, note


if __name__ == "__main__":
    main()
