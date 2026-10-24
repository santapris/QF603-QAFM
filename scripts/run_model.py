"""Part 4: model formation -> data/derived/rolling_fits.csv, model_path.csv, output/tables/04_*, figures.

Usage:  .venv/bin/python scripts/run_model.py
Requires scripts/build_data.py. All results are on spot data (a spot proxy).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilrv import model as M
from oilrv.config import load_config, path
from oilrv.data import load_clean
from oilrv.plotting import CRITICAL, REF_LINE, SERIES, TAB_DIR, save, save_table, use_style

LABELS = {"crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}


def _shade_stress(ax, cfg, last):
    for a, b in cfg["evaluation"]["stress_windows"].values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) if b else last, color=REF_LINE, alpha=0.10, lw=0)


def period_medians(fits: pd.DataFrame, periods: list, cols: list[str]) -> pd.DataFrame:
    rows = {}
    for a, b in periods:
        d = fits.loc[a:b, cols]
        if len(d):
            rows[f"{a[:4]}–{(b or str(fits.index[-1].date()))[:4]}"] = d.replace(np.inf, np.nan).median()
    return pd.DataFrame(rows).T


def main() -> dict:
    use_style()
    cfg = load_config()
    _, sp = load_clean()
    res = M.build_model(sp, cfg)
    last = sp.index[-1]

    fits_all = pd.concat({f"{v}|{s}": f for (v, s), (f, _) in res.items()}, names=["key", "date"])
    paths_all = pd.concat({f"{v}|{s}": p for (v, s), (_, p) in res.items()}, names=["key", "date"])
    fits_all.to_csv(path("derived_dir") / "rolling_fits.csv")
    paths_all.to_csv(path("derived_dir") / "model_path.csv")

    out = {"source": "spot"}
    cols = ["mean", "sd", "half_life", "adf_p", "dS_sd", "garch_a", "garch_b", "garch_nu", "vecm_spread_speed"]
    med = pd.concat({k: period_medians(f, cfg["evaluation"]["sub_periods"], cols)
                     for k, (f, _) in [(f"{v}|{s}", r) for (v, s), r in res.items()]}, names=["key", "period"])
    save_table(med, "04_param_medians_by_period.csv")

    gates, vt, latest = {}, {}, {}
    for (v, s), (f, p) in res.items():
        k = f"{v}|{s}"
        gates[k] = {"months": len(f), "gate_open_rate": float(f["gate"].mean()),
                    "garch_ok_rate": float(f["garch_ok"].mean()) if "garch_ok" in f else np.nan,
                    "share_days_abs_z_ge_2": float((p["z"].abs() >= 2).mean()),
                    "share_days_abs_z_ge_4": float((p["z"].abs() >= 4).mean())}
        if v == "johansen":
            gates[k]["weights_ok_rate"] = float(f["weights_ok"].mean())
        ok = p["sigma_next"].notna()
        vt[k] = M.vol_forecast_test(p[ok])
        lp = p.iloc[-1]
        latest[k] = {"date": str(p.index[-1].date()), "S": float(lp["S"]), "z": float(lp["z"]),
                     "sigma_next": float(lp["sigma_next"]), "dS_sd": float(lp["dS_sd"]),
                     "half_life": float(lp["half_life"]), "adf_p": float(lp["adf_p"]), "gate": bool(lp["gate"]),
                     "sigma_pct_in_history": float((p["sigma_next"] <= lp["sigma_next"]).mean())}
    save_table(pd.DataFrame(gates).T, "04_gate_and_signal_rates.csv")
    save_table(pd.DataFrame(vt).T, "04_vol_forecast_test.csv")
    save_table(pd.DataFrame(latest).T, "04_latest_state.csv")
    out.update({"gates": gates, "vol_forecast_test": vt, "latest": latest})

    # VECM loadings with the traded vector (rolling) and the unrestricted VECM (full sample)
    vec = {}
    for s in LABELS:
        f = res[("fixed", s)][0]
        legs = [c[len("vecm_alpha_"):] for c in f.columns if c.startswith("vecm_alpha_")]
        vec[s] = {**{f"alpha_{k}": float(f[f"vecm_alpha_{k}"].median()) for k in legs},
                  **{f"share_t_lt_-2_{k}": float((f[f"vecm_t_{k}"] < -2).mean()) for k in legs},
                  **{f"share_t_gt_2_{k}": float((f[f"vecm_t_{k}"] > 2).mean()) for k in legs},
                  "spread_speed": float(f["vecm_spread_speed"].median())}
    out["vecm_known_vector"] = vec
    free = M.fit_vecm_free(sp)
    save_table(pd.concat({"alpha": free["alpha"], "alpha_t": free["alpha_t"], "beta": free["beta"]}),
               "04_vecm_free_full_sample.csv")
    out["vecm_free_full_sample"] = {k: v.round(4).to_dict() for k, v in free.items()}

    # Figures
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for a, s in zip(ax, LABELS):
        p = res[("fixed", s)][1]
        a.plot(p.index, p["z"].clip(-8, 8), color=SERIES[0], lw=0.7, label="z (clipped at ±8)")
        for kz, st in [(2, "--"), (4, "-")]:
            a.axhline(kz, color=CRITICAL if kz == 4 else REF_LINE, lw=0.7, ls=st)
            a.axhline(-kz, color=CRITICAL if kz == 4 else REF_LINE, lw=0.7, ls=st)
        closed = ~p["gate"]
        a.fill_between(p.index, -8, 8, where=closed, color=SERIES[1], alpha=0.12, lw=0, label="Gate closed (half-life out of band)")
        a.set_ylim(-8.5, 8.5)
        a.set_title(f"{LABELS[s]}: point-in-time z (deseasonalised, frozen monthly fits)")
        a.legend(loc="upper left", ncols=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "04_zscore_and_gates.png")

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for a, s in zip(ax, LABELS):
        p = res[("fixed", s)][1]
        a.plot(p.index, p["sigma_next"], color=SERIES[0], lw=0.7, label="GARCH-t σ̂ (next day)")
        a.plot(p.index, p["dS_sd"], color=SERIES[1], lw=1.1, label="Window sd of ΔS")
        a.set_yscale("log")
        a.set_ylabel("$/bbl per day")
        a.set_title(f"{LABELS[s]}: one-day volatility forecast")
        a.legend(loc="upper left", ncols=2, fontsize=7.5)
        _shade_stress(a, cfg, last)
    fig.tight_layout()
    save(fig, "04_garch.png")

    f = res[("fixed", "crack321")][0]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for i, k in enumerate(["cl", "rb", "ho"]):
        ax[0].plot(f.index, f[f"vecm_t_{k}"], color=SERIES[i], lw=1, label=k.upper())
    for v in (-2, 2):
        ax[0].axhline(v, color=REF_LINE, lw=0.7, ls="--")
    ax[0].set_ylabel("t-stat of loading on crack")
    ax[0].set_title("Crack VECM (vector fixed at 3:2:1): which leg corrects? RB/HO correct when t < −2; CL corrects when t > +2")
    ax[0].legend(loc="upper left", ncols=3)
    fj = res[("johansen", "crack321")][0]
    ax[1].plot(fj.index, fj["w_rb"], color=SERIES[1], lw=1, marker=".", ms=3, label="RB weight")
    ax[1].plot(fj.index, fj["w_ho"], color=SERIES[2], lw=1, marker=".", ms=3, label="HO weight")
    ax[1].axhline(2, color=SERIES[1], lw=0.7, ls="--")
    ax[1].axhline(1, color=SERIES[2], lw=0.7, ls="--")
    ax[1].set_title(f"Johansen variant: first vector scaled to −3 CL (dashed: 3:2:1). Usable in {fj['weights_ok'].mean():.0%} of months")
    ax[1].legend(loc="upper left", ncols=2)
    for a in ax:
        _shade_stress(a, cfg, last)
    fig.tight_layout()
    save(fig, "04_vecm_and_johansen_weights.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for a, s in zip(ax, ["crack321", "ho_rb"]):
        f = res[("fixed", s)][0]
        seas = f[[f"seas_{m}" for m in range(1, 13)]]
        im = a.imshow(seas.T.to_numpy(), aspect="auto", cmap="RdBu_r",
                      vmin=-np.nanmax(np.abs(seas.to_numpy())), vmax=np.nanmax(np.abs(seas.to_numpy())),
                      extent=[0, len(seas), 12.5, 0.5])
        yrs = [i for i, d in enumerate(seas.index) if d.month == 1 and d.year % 2 == 0]
        a.set_xticks(yrs, [seas.index[i].year for i in yrs])
        a.set_yticks(range(1, 13), list("JFMAMJJASOND"))
        a.grid(False)
        a.set_title(f"{LABELS[s]}: in-window month effects ($/bbl)")
        fig.colorbar(im, ax=a, fraction=0.03)
    fig.tight_layout()
    save(fig, "04_seasonal_effects.png")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "model_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
