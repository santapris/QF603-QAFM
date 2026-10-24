"""Part 5: regime filter -> data/derived/regime_*.csv, output/tables/05_*, output/figures/05_*.

Usage:  .venv/bin/python scripts/run_regime.py
Requires scripts/build_data.py and scripts/run_model.py. All results are on spot data (a spot proxy).

The gated configuration uses the pre-registered core features, the logit, and the 20-day label.
Probit, the robustness features and the 10/40-day labels are reported as robustness only.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilrv import regime as R
from oilrv.config import load_config, path
from oilrv.data import load_clean, load_features
from oilrv.plotting import CRITICAL, REF_LINE, SERIES, TAB_DIR, save, save_table, use_style

LABELS = {"crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}


def load_model() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    d = path("derived_dir")
    fits = pd.read_csv(d / "rolling_fits.csv", index_col=[0, 1], parse_dates=["date"])
    paths = pd.read_csv(d / "model_path.csv", index_col=[0, 1], parse_dates=["date", "fit_date"])
    return {k: (fits.loc[k], paths.loc[k]) for k in fits.index.get_level_values(0).unique()}


def build(cfg, sp, feats, cal, model, horizon=None, extra=None, kind="logit"):
    return R.build_regime(sp, feats, cal, model, cfg, horizon, extra, kind)


def evaluations(ep, enter_p):
    rows = {"pooled (fixed)": R.evaluate_filter(ep[ep["key"].str.startswith("fixed|")], enter_p)}
    for key in sorted(ep["key"].unique()):
        rows[key] = R.evaluate_filter(ep[ep["key"] == key], enter_p)
    rows["pooled, gate open only"] = R.evaluate_filter(ep[ep["key"].str.startswith("fixed|") & ep["gate"]], enter_p)
    return pd.DataFrame(rows).T


def stress_table(ep, cfg, enter_p):
    rows = {}
    for w, (a, b) in cfg["evaluation"]["stress_windows"].items():
        for key in sorted(ep["key"].unique()):
            e = ep[(ep["key"] == key)].loc[a:b]
            if not len(e):
                continue
            has_p = e["p_hat"].notna()
            take = has_p & (e["p_hat"] >= enter_p)
            block = has_p & (e["p_hat"] < enter_p)
            rows[(w, key)] = {"episodes": len(e), "success_all": e["reverted"].mean(),
                              "with_p": int(has_p.sum()), "passed": int(take.sum()), "blocked": int(block.sum()),
                              "success_passed": e.loc[take, "reverted"].mean() if take.any() else np.nan,
                              "success_blocked": e.loc[block, "reverted"].mean() if block.any() else np.nan,
                              "mean_p": e["p_hat"].mean()}
    return pd.DataFrame(rows).T.rename_axis(["window", "key"])


def main() -> dict:
    use_style()
    cfg = load_config()
    enter_p, exit_p = cfg["strategy"]["regime_enter_p"], cfg["strategy"]["regime_exit_p"]
    _, sp = load_clean()
    feats = load_features()
    cal = pd.read_csv(path("clean_dir") / "event_features.csv", index_col=0, parse_dates=True)
    model = load_model()

    ep, daily, p_daily, coefs = build(cfg, sp, feats, cal, model)
    ep.to_csv(path("derived_dir") / "regime_episodes.csv")
    pd.concat({k: daily[k].join(p_daily[k]) for k in daily}, names=["key", "date"]).to_csv(
        path("derived_dir") / "regime_probability.csv")
    coefs.to_csv(path("derived_dir") / "regime_coefficients.csv")

    out = {"source": "spot", "features": R.model_features(cfg)}
    summ = ep.groupby("key").agg(episodes=("reverted", "size"), share_reverted=("reverted", "mean"),
                                 share_gate_open=("gate", "mean"), median_days=("days", "median"))
    oc = pd.crosstab(ep["key"].to_numpy(), ep["outcome"].to_numpy(), normalize="index").rename_axis("key")
    summ = summ.join(oc)
    save_table(summ, "05_episodes_summary.csv")
    out["episodes"] = summ.astype(float).round(3).to_dict("index")

    ev = evaluations(ep, enter_p)
    save_table(ev, "05_filter_eval.csv")
    out["filter_eval"] = ev.astype(float).round(3).to_dict("index")
    cal_tab = R.calibration_table(ep[ep["key"].str.startswith("fixed|")])
    save_table(cal_tab, "05_calibration.csv")
    st = stress_table(ep, cfg, enter_p)
    save_table(st, "05_stress_windows.csv")
    out["stress_windows"] = {f"{w}|{k}": v for (w, k), v in st.astype(float).round(3).to_dict("index").items()}
    if len(coefs):
        save_table(coefs, "05_coefficients_by_month.csv")
        out["last_coefficients"] = coefs.iloc[-1].round(4).to_dict()
        out["first_month_with_filter"] = str(coefs.index[0].date())
        out["months_with_filter"] = len(coefs)

    # Robustness (reported, never gated): probit, robustness features, label horizons
    rob = {}
    variants = {"probit": dict(kind="probit"),
                "core + robustness features": dict(extra=[f for f in cfg["regime"]["robustness_features"]
                                                          if daily[next(iter(daily))][f].notna().any()])}
    for h in cfg["robustness"]["label_horizon_days"]:
        if h != cfg["regime"]["label_horizon_days"]:
            variants[f"label horizon {h}d"] = dict(horizon=h)
    for lab, kw in variants.items():
        e2, *_ = build(cfg, sp, feats, cal, model, **kw)
        r = R.evaluate_filter(e2[e2["key"].str.startswith("fixed|")], enter_p)
        rob[lab] = r
    rob["logit core (gated config)"] = ev.loc["pooled (fixed)"].to_dict()
    rob_tab = pd.DataFrame(rob).T
    save_table(rob_tab, "05_robustness.csv")
    out["robustness"] = rob_tab.astype(float).round(3).to_dict("index")

    # Descriptive full-sample logit (look-ahead by construction; never used for trading)
    train = ep[ep["key"].str.startswith("fixed|")]
    full = R.fit_model(train, R.model_features(cfg))
    if full is not None:
        desc = pd.DataFrame({"coef": full.params, "se": full.bse, "z": full.tvalues, "p": full.pvalues})
        save_table(desc, "05_logit_full_sample_descriptive.csv")
        out["full_sample_logit_descriptive"] = desc.round(4).to_dict("index")

    # Figures
    last = sp.index[-1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for a, s in zip(ax, LABELS):
        k = f"fixed|{s}"
        p = p_daily[k]["p_hat"]
        a.plot(p.index, p, color=SERIES[0], lw=0.8, label="P(revert), point in time")
        a.axhline(enter_p, color=SERIES[2], lw=0.8, ls="--", label=f"Enter ≥ {enter_p}")
        a.axhline(exit_p, color=CRITICAL, lw=0.8, ls="--", label=f"Exit < {exit_p}")
        e = ep[ep["key"] == k]
        a.scatter(e.index[e["reverted"] == 1], np.full((e["reverted"] == 1).sum(), 1.03), s=8, color=SERIES[2],
                  marker="|", label="Episode: reverted")
        a.scatter(e.index[e["reverted"] == 0], np.full((e["reverted"] == 0).sum(), -0.03), s=8, color=CRITICAL,
                  marker="|", label="Episode: failed")
        for a0, b0 in cfg["evaluation"]["stress_windows"].values():
            a.axvspan(pd.Timestamp(a0), pd.Timestamp(b0) if b0 else last, color=REF_LINE, alpha=0.10, lw=0)
        a.set_ylim(-0.08, 1.08)
        a.set_title(f"{LABELS[s]}: pooled logit P(revert) (no filter before {out.get('first_month_with_filter', 'n/a')})")
        a.legend(loc="lower left", ncols=5, fontsize=7)
    fig.tight_layout()
    save(fig, "05_regime_probability.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot([0, 1], [0, 1], color=REF_LINE, lw=0.8)
    ax[0].scatter(cal_tab["mean_p"], cal_tab["share_reverted"], s=cal_tab["n"] * 3, color=SERIES[0], alpha=0.8)
    for _, r in cal_tab.iterrows():
        ax[0].annotate(f"n={int(r['n'])}", (r["mean_p"], r["share_reverted"]), fontsize=7, xytext=(4, 4),
                       textcoords="offset points")
    ax[0].set_xlabel("Mean predicted P (bin)")
    ax[0].set_ylabel("Share reverted")
    ax[0].set_title("Calibration, out of sample (pooled fixed episodes)")
    if len(coefs):
        for i, c in enumerate([c for c in cfg["regime"]["core_features"] if c in coefs]):
            ax[1].plot(coefs.index, coefs[c], color=SERIES[i], lw=1, label=c)
        ax[1].axhline(0, color=REF_LINE, lw=0.7)
        ax[1].set_title("Logit coefficients, refit monthly")
        ax[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "05_calibration_and_coefficients.png")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "regime_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
