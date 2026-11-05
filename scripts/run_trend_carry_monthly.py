"""EXPLORATORY: monthly trend and carry on the Bloomberg futures -> output/exploratory/trend_carry_monthly/.

Usage:  .venv/bin/python scripts/run_trend_carry_monthly.py
Rules: PREREGISTRATION_TREND_CARRY_MONTHLY.md (registered 2026-11-05, after the daily trend and carry result, before this
code computed any monthly signal or P&L). Run once. Nothing here changes the registered verdict.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from oilrv import trend_carry as TC
from oilrv.config import ROOT, load_config, set_source
from oilrv.metrics import subperiods, window_pnl
from oilrv.plotting import REF_LINE, SERIES, use_style

OUT = ROOT / "output" / "exploratory" / "trend_carry_monthly"
REFERENCE_SHARPE, REFERENCE_CHANGES, SHARPE_TOLERANCE = 0.447, 1150, 0.01   # section 6 of the pre-registration


def costs(run: dict) -> pd.DataFrame:
    """Gross, cost and roll per sleeve (a sleeve is the average of its instruments)."""
    rows = {}
    for sleeve in ["trend", "carry"]:
        ds = [d for k, d in run["instruments"].items() if k.startswith(sleeve + "|")]
        g, c, r = (sum(float(d[col].sum()) for d in ds) / len(ds) for col in ["gross", "cost", "roll"])
        rows[sleeve] = {"gross": g, "cost": c, "roll": r, "cost_share_of_gross": (c + r) / g if g else np.nan}
    return pd.DataFrame(rows).T


def main() -> dict:
    use_style()
    cfg = load_config()
    set_source("futures")
    tc, m = cfg["exploratory_trend_carry"], cfg["exploratory_trend_carry_monthly"]
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    inst = TC.build_instruments(cfg=cfg)
    sig = TC.sigmas(inst, cfg)
    K = m["decision_months"]
    runs = {"daily": TC.run_sleeves(inst, cfg, sig=sig),
            "monthly": TC.run_sleeves(inst, cfg, sig=sig, decision_months=K, hold_sigma=m["hold_sigma"]),
            "monthly_signal_daily_size": TC.run_sleeves(inst, cfg, sig=sig, decision_months=K, hold_sigma=False)}

    ref = runs["daily"]
    if abs(TC.sharpe(ref["combined"]) - REFERENCE_SHARPE) > SHARPE_TOLERANCE or ref["changes"]["combined"] != REFERENCE_CHANGES:
        raise RuntimeError(f"daily reference does not reproduce the registered result: "
                           f"{TC.sharpe(ref['combined']):.4f}, {ref['changes']['combined']}")

    # G4 grid at the base setting: lookbacks (carry at base) and slopes (trend at base)
    L0, S0, base = tc["trend"]["lookback_days"], tc["carry"]["slope"], runs["monthly"]
    grid = {}
    for L in tc["trend"]["lookback_grid"]:
        r = base if L == L0 else TC.run_sleeves(inst, cfg, lookback=L, sig=sig, decision_months=K, hold_sigma=m["hold_sigma"])
        grid[("trend lookback", L)] = {"trend": TC.sharpe(r["trend"]), "combined": TC.sharpe(r["combined"])}
    for s in tc["carry"]["slope_grid"]:
        r = base if s == S0 else TC.run_sleeves(inst, cfg, slope=s, sig=sig, decision_months=K, hold_sigma=m["hold_sigma"])
        grid[("carry slope", s)] = {"carry": TC.sharpe(r["carry"]), "combined": TC.sharpe(r["combined"])}
    grid_t = pd.DataFrame(grid).T.rename_axis(["grid", "value"])
    grid_t.to_csv(OUT / "tables" / "grid.csv")
    g_comb = {f"{a}={b}": v for (a, b), v in grid_t["combined"].items()}
    g_trend = {f"lookback={b}": v for (a, b), v in grid_t["trend"].dropna().items()}
    g_carry = {f"slope={b}": v for (a, b), v in grid_t["carry"].dropna().items()}
    verdicts = {"combined": TC.verdict(base["combined"], base["changes"]["combined"], g_comb, cfg),
                "trend": TC.verdict(base["trend"], base["changes"]["trend"], g_trend, cfg),
                "carry": TC.verdict(base["carry"], base["changes"]["carry"], g_carry, cfg)}
    pd.DataFrame({k: {kk: (json.dumps(v, default=float) if isinstance(v, dict) else v) for kk, v in d.items()}
                  for k, d in verdicts.items()}).T.to_csv(OUT / "tables" / "gate_verdicts.csv")

    # side by side: every run and book
    rows = {}
    for name, r in runs.items():
        for k in ["trend", "carry", "combined"]:
            p = r[k]
            rows[(name, k)] = {**TC.summary(p), "position_changes": r["changes"][k],
                               "share_2026": float(p.loc["2026"].sum() / p.sum())}
    comp = pd.DataFrame(rows).T.rename_axis(["run", "book"])
    comp.to_csv(OUT / "tables" / "comparison.csv")
    cost_t = pd.concat({n: costs(r) for n, r in runs.items()}, names=["run", "sleeve"])
    cost_t.to_csv(OUT / "tables" / "costs.csv")

    books = pd.DataFrame({f"{n}|{k}": r[k] for n, r in runs.items() for k in ["trend", "carry", "combined"]})
    books.to_csv(OUT / "tables" / "daily_books.csv")
    inst_t = pd.DataFrame({(n, key): {**TC.summary(d["pnl"]), "gross_total": float(d["gross"].sum()),
                                      "cost_total": float(d["cost"].sum()), "roll_total": float(d["roll"].sum()),
                                      "position_changes": TC.position_changes(d["target"])}
                           for n, r in runs.items() for key, d in r["instruments"].items()}).T
    inst_t.to_csv(OUT / "tables" / "instruments.csv")
    sub = pd.concat({n: subperiods(r["combined"].to_frame("pnl"), "pnl", cfg["evaluation"]["sub_periods"])
                     for n, r in runs.items()}, names=["run", "period"])
    sub.to_csv(OUT / "tables" / "subperiods.csv")
    stress = pd.DataFrame({w: {n: window_pnl(r["combined"].to_frame("pnl"), "pnl", a, b) for n, r in runs.items()}
                           for w, (a, b) in cfg["evaluation"]["stress_windows"].items()}).T
    stress.to_csv(OUT / "tables" / "stress_windows.csv")
    annual = pd.DataFrame({n: r["combined"].groupby(r["combined"].index.year).sum() for n, r in runs.items()})
    annual.to_csv(OUT / "tables" / "annual.csv")
    corr = pd.DataFrame({n: r["combined"] for n, r in runs.items()}).corr().round(3)
    corr.to_csv(OUT / "tables" / "correlation.csv")

    # descriptive: oil exposure of each book, as run_support_tests.py does for the daily book
    dx = pd.concat([inst["cl"]["x"].diff(), inst["brent"]["x"].diff()], axis=1, keys=["cl", "brent"])
    beta = []
    for name, r in runs.items():
        for k in ["trend", "carry", "combined"]:
            y = r[k]
            fit = sm.OLS(y, sm.add_constant(dx.loc[y.index]), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 10})
            beta.append({"run": name, "book": k, "raw_sharpe": TC.sharpe(y), "alpha_t": fit.tvalues["const"],
                         "sharpe_from_alpha": fit.params["const"] / y.std() * np.sqrt(252)})
    pd.DataFrame(beta).to_csv(OUT / "tables" / "oil_beta.csv", index=False)
    now = pd.DataFrame({key: {"date": d.index[-1].date(), "direction": "long" if d["target"].iloc[-1] > 0 else "short"}
                        for key, d in base["instruments"].items()}).T
    now.to_csv(OUT / "tables" / "current_book.csv")

    t = float(comp.loc[("monthly", "combined"), "sharpe_t"])
    v = verdicts["combined"]
    out = {"status": "EXPLORATORY, seventh look (PREREGISTRATION_TREND_CARRY_MONTHLY.md); does not change the registered verdict",
           "reference_reproduced": True, "gates_passed": v["PASS"], "t": t,
           "called_evidence": bool(v["PASS"] and t > m["pass_needs_t"]),
           "verdicts": verdicts, "comparison": {f"{a}|{b}": d for (a, b), d in comp.astype(float).round(4).to_dict("index").items()},
           "costs": {f"{a}|{b}": d for (a, b), d in cost_t.astype(float).round(3).to_dict("index").items()},
           "stress_windows": stress.round(0).to_dict("index")}
    (OUT / "tables" / "summary.json").write_text(json.dumps(out, indent=2, default=float))

    last = base["combined"].index[-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (n, r) in enumerate(runs.items()):
        ax.plot(r["combined"].index, r["combined"].cumsum() / 1e3, color=SERIES[i], lw=1.1,
                label=f"{n} (Sharpe {TC.sharpe(r['combined']):.2f})")
    for a0, b0 in cfg["evaluation"]["stress_windows"].values():
        ax.axvspan(pd.Timestamp(a0), pd.Timestamp(b0) if b0 else last, color=REF_LINE, alpha=0.10, lw=0)
    ax.axhline(0, color=REF_LINE, lw=0.7)
    ax.set_ylabel("cumulative net P&L, \\$k")
    ax.set_title("Trend and carry 50/50 book: registered daily against the monthly variants (exploratory)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cumulative_pnl.png")
    plt.close(fig)
    return out


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ["status", "reference_reproduced", "gates_passed", "t", "called_evidence"]}, indent=2))
