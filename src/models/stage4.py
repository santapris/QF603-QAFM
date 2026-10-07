"""Stage 4 — robustness and out-of-sample tests (PLAN Phase 4, Q4).

1. ``expost_rerun``: Stage 1–3 specifications with the ex-post VRP (variance-swap payoff) as the dependent
   variable; control = ex-ante VRP (ex-post is never a regressor), NW lags ≥ k + 1 [F6].
2. ``bridge_insample`` [F7]: vrp_expost_t = a + b·vrp_exante_t + c'X_t + e. Tests b = 1 (is ex-ante VRP an
   unbiased forecast of the short-variance payoff?) and c = 0 (do predictors forecast the payoff?).
3. ``oos_forecasts``: expanding-window forecasts at monthly origins in the validation window. At each origin
   every model is re-fitted on rows whose target is observed by the origin (``holdout.training_rows``, exact
   observation dates for ex-post targets); scaling, PCA and ridge are fitted inside the window. The HAR
   inside VRP is already recursive (Phase 2c), so ex-ante VRP at t uses data up to t only.
   Models: historical mean, AR(1) (target on its control), AR(1) + each predictor, AR(1) + all (OLS),
   AR(1) + m PCs, ridge(λ); for the bridge also the naive forecast E_t[payoff] = vrp_exante_t.
4. ``oos_metrics``: OOS R² vs the benchmark, Clark–West (nested, one-sided), Diebold–Mariano (two-sided),
   HAC lags = k (≥ 1). PCA and ridge families are marked ``tuned`` — their m / λ are picked on the same
   validation window, so only the holdout is a clean test [F10].
5. ``selection``: best λ and m per horizon on validation; proposed specification set for 🧑 approval.

The final holdout run (``final=True``) is done once in Phase 7 together with Stage 5 and Phase 6 [F14].

Run:  python -m src.models.stage4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.stage1 import _sample, control_series, lags_for, persistence, single_predictors, zscore_train
from src.models.stage2 import alpha_grid, full_ols
from src.models.stage3 import interactions
from src.utils.holdout import restrict, training_rows
from src.utils.io import load_config, project_path, write_parquet
from src.utils.newey_west import nw_cov, ols_nw


# ------------------------------------------------------------------ 1–2: in-sample robustness
def expost_rerun(panel: pd.DataFrame, predictors: list[str], horizons, cfg: dict, final: bool = False) -> dict:
    t = "vrp_expost"
    return {"persistence": persistence(panel, t, horizons, cfg, final),
            "single": single_predictors(panel, t, predictors, horizons, cfg, final),
            "ols": full_ols(panel, t, predictors, horizons, cfg, final, bootstrap=False),
            "interactions": interactions(panel, t, predictors, horizons, cfg, final)}


def bridge_insample(panel: pd.DataFrame, predictors: list[str], cfg: dict, final: bool = False) -> pd.DataFrame:
    target = cfg["stage4"]["bridge_target"]
    specs = {"vrp_exante only": []} | {f"+ {x}": [x] for x in predictors} | {"+ all predictors": predictors}
    rows = []
    for name, preds in specs.items():
        cols = {"vrp_exante": panel["vrp_exante"],
                **{x: zscore_train(panel[x], cfg["oos"]["train_end"]) for x in preds}}
        df = _sample(panel, target, 0, cols, cfg, final)
        lags = lags_for(0, len(df), target, cfg)
        res = ols_nw(df["y"], df.drop(columns="y"), lags)
        X = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
        cov = nw_cov(X, res.resid.to_numpy(), lags)
        # joint Wald: b = 1 and all c = 0
        r = res.params.to_numpy()[1:] - np.r_[1.0, np.zeros(len(preds))]
        V = cov[1:, 1:]
        wald = float(r @ np.linalg.solve(V, r))
        rows.append(dict(spec=name, a=res.params["const"], b=res.params["vrp_exante"], se_b=res.se["vrp_exante"],
                         t_b_eq_1=(res.params["vrp_exante"] - 1) / res.se["vrp_exante"],
                         wald_b1_c0=wald, p_wald=float(stats.chi2.sf(wald, len(r))),
                         **{f"t_{x}": res.tstat[x] for x in preds}, r2=res.rsquared, nobs=res.nobs, lags=lags))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3: OOS engine
def model_specs(predictors: list[str], cfg: dict, bridge: bool = False) -> list[dict]:
    specs = [dict(name="mean", kind="mean")]
    if bridge:
        specs.append(dict(name="naive", kind="naive"))
    specs.append(dict(name="ar1", kind="ols", preds=[]))
    specs += [dict(name=f"ar1+{x}", kind="ols", preds=[x]) for x in predictors]
    specs.append(dict(name="ar1+all", kind="ols", preds=list(predictors)))
    specs += [dict(name=f"pca{m}", kind="pca", preds=list(predictors), m=m, tuned=True) for m in cfg["stage4"]["pca_m"]
              if m <= len(predictors)]
    specs += [dict(name=f"ridge{a:.4g}", kind="ridge", preds=list(predictors), alpha=a, tuned=True)
              for a in alpha_grid(cfg)]
    return specs


def _data(panel: pd.DataFrame, target: str, k: int, predictors: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    d = pd.DataFrame({"y": panel[target].shift(-k), "ctrl": control_series(panel, target)}, index=panel.index)
    for x in predictors:
        d[x] = panel[x]
    obs = panel[f"obs_date_{target}"].shift(-k) if f"obs_date_{target}" in panel else None
    return d, obs


def _fit_predict(spec: dict, tr: pd.DataFrame, x0: pd.DataFrame) -> float:
    kind = spec["kind"]
    if kind == "mean":
        return float(tr["y"].mean())
    if kind == "naive":
        return float(x0["ctrl"].iloc[0])
    cols = ["ctrl"] + spec.get("preds", [])
    X, y = tr[cols], tr["y"]
    if kind == "ols":
        A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
        beta, *_ = np.linalg.lstsq(A, y.to_numpy(float), rcond=None)
        return float(np.r_[1.0, x0[cols].to_numpy(float)[0]] @ beta)
    if kind == "pca":
        preds = spec["preds"]
        sc = StandardScaler().fit(tr[preds])
        pca = PCA(n_components=spec["m"]).fit(sc.transform(tr[preds]))
        f_tr = pca.transform(sc.transform(tr[preds]))
        f_0 = pca.transform(sc.transform(x0[preds]))
        A = np.column_stack([np.ones(len(tr)), tr["ctrl"].to_numpy(float), f_tr])
        beta, *_ = np.linalg.lstsq(A, y.to_numpy(float), rcond=None)
        return float(np.r_[1.0, x0["ctrl"].iloc[0], f_0[0]] @ beta)
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"])).fit(X, y)
        return float(model.predict(x0[cols])[0])
    raise ValueError(f"unknown model kind {kind!r}")


def validation_origins(panel: pd.DataFrame, target: str, k: int, cfg: dict, final: bool = False) -> pd.DatetimeIndex:
    """Origins after train_end (validation) or after validate_end (holdout) whose target is observable."""
    d, obs = _data(panel, target, k, [])
    lo = pd.Timestamp(cfg["oos"]["validate_end" if final else "train_end"])
    d = d.loc[d.index > lo]
    if obs is not None:
        obs = obs.loc[d.index]
        d, obs = d.loc[obs.notna()], obs.loc[obs.notna()]
    d = restrict(d, target, k, final=final, cfg=cfg, obs_dates=obs)
    return pd.DatetimeIndex(d.index[d["y"].notna()])


def oos_forecasts(panel: pd.DataFrame, target: str, k: int, specs: list[dict], origins, cfg: dict) -> pd.DataFrame:
    preds = sorted({x for s in specs for x in s.get("preds", [])})
    d, obs = _data(panel, target, k, preds)
    min_train = cfg["stage4"]["min_train_months"]
    rows = []
    for origin in pd.DatetimeIndex(origins):
        train = training_rows(d, origin, target, k, cfg=cfg, obs_dates=obs)
        row = dict(date=origin, y=d.loc[origin, "y"], ctrl=d.loc[origin, "ctrl"])
        for spec in specs:
            cols = ["y", "ctrl"] + spec.get("preds", [])
            tr = train[cols].dropna()
            x0 = d.loc[[origin], cols]
            if len(tr) < min_train or x0.drop(columns="y").isna().any(axis=None):
                row[spec["name"]] = np.nan
                continue
            row[spec["name"]] = _fit_predict(spec, tr, x0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# ------------------------------------------------------------------ 4: metrics
def _hac_t(x: pd.Series, lags: int) -> float:
    res = ols_nw(x, pd.DataFrame(index=x.index), lags=lags, add_const=True)
    return float(res.tstat["const"])


def oos_metrics(fc: pd.DataFrame, bench: str, k: int, specs: list[dict]) -> pd.DataFrame:
    lags = max(k, 1)
    tuned = {s["name"]: s.get("tuned", False) for s in specs}
    rows = []
    for m in [s["name"] for s in specs if s["name"] != bench]:
        d = fc[["y", bench, m]].dropna()
        if len(d) < 12:
            continue
        eb, em = d["y"] - d[bench], d["y"] - d[m]
        cw = eb ** 2 - (em ** 2 - (d[bench] - d[m]) ** 2)
        dm = eb ** 2 - em ** 2
        t_cw, t_dm = _hac_t(cw, lags), _hac_t(dm, lags)
        rows.append(dict(model=m, bench=bench, k=k, n=len(d), mse=float((em ** 2).mean()),
                         mse_bench=float((eb ** 2).mean()), oos_r2=float(1 - (em ** 2).sum() / (eb ** 2).sum()),
                         cw_t=t_cw, cw_p=float(stats.norm.sf(t_cw)), dm_t=t_dm, dm_p=float(2 * stats.norm.sf(abs(t_dm))),
                         tuned=tuned.get(m, False)))
    return pd.DataFrame(rows)


def selection(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per horizon: best ridge λ, best PCA m, and all models ranked by validation MSE (🧑 approves the set)."""
    out = []
    for (k, bench), g in metrics.groupby(["k", "bench"]):
        ridge = g[g["model"].str.startswith("ridge")].nsmallest(1, "mse")
        pca = g[g["model"].str.startswith("pca")].nsmallest(1, "mse")
        best = g.nsmallest(1, "mse")
        out.append(dict(k=k, bench=bench, best_ridge=ridge["model"].iloc[0] if len(ridge) else None,
                        best_ridge_oos_r2=ridge["oos_r2"].iloc[0] if len(ridge) else np.nan,
                        best_pca=pca["model"].iloc[0] if len(pca) else None,
                        best_pca_oos_r2=pca["oos_r2"].iloc[0] if len(pca) else np.nan,
                        best_overall=best["model"].iloc[0], best_overall_oos_r2=best["oos_r2"].iloc[0],
                        n_models_beating_bench=int((g["oos_r2"] > 0).sum()), n_models=len(g)))
    return pd.DataFrame(out)


def gw_figure(fcs: dict, benches: dict, highlight: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(fcs), figsize=(5 * len(fcs), 4), squeeze=False)
    for ax, (label, fc) in zip(axes[0], fcs.items()):
        b = benches[label]
        for m in highlight[label]:
            d = fc[["y", b, m]].dropna()
            ax.plot(d.index, ((d["y"] - d[b]) ** 2 - (d["y"] - d[m]) ** 2).cumsum(), label=m)
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{label}: cum. SSE({b}) − SSE(model)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Goyal–Welch cumulative SSE difference, validation 2016–2020 (> 0 rising = model beats benchmark)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ------------------------------------------------------------------ pipeline
def _keep(specs: list[dict], models: dict | None, k: int, benches: tuple[str, ...]) -> list[dict]:
    """Restrict to benchmarks + the frozen models for horizon k (holdout run evaluates only frozen specs)."""
    if models is None:
        return specs
    keep = set(models.get(k, models.get(str(k), []))) | set(benches)
    return [s for s in specs if s["name"] in keep]


def run_oos(panel: pd.DataFrame, predictors: list[str], cfg: dict, final: bool = False, models: dict | None = None):
    """Validation (or, with final=True, holdout) forecasts and metrics for ex-ante VRP and the payoff bridge.

    ``models`` (frozen in Phase 7): {k: [model names]} — only these plus the benchmarks are evaluated."""
    fcs, metrics, benches = {}, [], {}
    for k in cfg["horizons"]:
        specs = _keep(model_specs(predictors, cfg), models, k, ("mean", "ar1"))
        fc = oos_forecasts(panel, "vrp_exante", k, specs, validation_origins(panel, "vrp_exante", k, cfg, final), cfg)
        fcs[f"VRP k={k}"], benches[f"VRP k={k}"] = fc, "ar1"
        metrics += [oos_metrics(fc, "ar1", k, specs).assign(target="vrp_exante"),
                    oos_metrics(fc, "mean", k, specs).assign(target="vrp_exante")]
    tgt = cfg["stage4"]["bridge_target"]
    specs = _keep(model_specs(predictors, cfg, bridge=True), models, 0, ("mean", "naive"))
    fc = oos_forecasts(panel, tgt, 0, specs, validation_origins(panel, tgt, 0, cfg, final), cfg)
    fcs["payoff k=0"], benches["payoff k=0"] = fc, "naive"
    metrics += [oos_metrics(fc, "naive", 0, specs).assign(target=tgt),
                oos_metrics(fc, "mean", 0, specs).assign(target=tgt)]
    return fcs, pd.concat(metrics, ignore_index=True), benches


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    panel = pd.read_parquet(project_path("data", "processed", "panel_monthly.parquet"))
    preds, horizons = cfg["predictors"], cfg["horizons"]
    out = project_path("outputs", "tables")
    for name, t in expost_rerun(panel, preds, horizons, cfg).items():
        t.to_csv(out / f"stage4_expost_{name}.csv", index=name == "persistence")
    bridge_insample(panel, preds, cfg).to_csv(out / "stage4_bridge_insample.csv", index=False)

    fcs, metrics, benches = run_oos(panel, preds, cfg)
    metrics.to_csv(out / "stage4_oos_metrics_validation.csv", index=False)
    sel = selection(metrics)
    sel.to_csv(out / "stage4_selection.csv", index=False)
    for label, fc in fcs.items():
        fc.to_csv(out / f"stage4_oos_forecasts_{label.replace(' ', '_').replace('=', '')}.csv")
    write_parquet(fcs["payoff k=0"], "data/processed/payoff_forecasts_validation.parquet")
    highlight = {}
    for lab in fcs:
        s = sel[sel["bench"] == benches[lab]]
        k = int(lab.split("=")[1])
        row = s[s["k"] == k].iloc[0]
        highlight[lab] = [m for m in dict.fromkeys(["ar1+all", row["best_ridge"], row["best_pca"]]) if m]
    gw_figure(fcs, benches, highlight, project_path("outputs", "figures", "oos_cum_sse.png"))
    print("[ok  ] Stage 4 written (stage4_*, oos_cum_sse.png, payoff_forecasts_validation.parquet).")
    print("🧑 Proposed selection (validation, TUNED — only the holdout is clean):\n", sel.to_string(index=False))
    return fcs, metrics, sel


if __name__ == "__main__":
    main()
