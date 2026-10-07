"""Phase 6 driver: run S0 / S1 / S2 on the validation window (or the holdout with ``final=True``, Phase 7 only).

* S0 always-short (m = 1); S1 VRP-timed (μ̂ = vrp_exante); S2 model-timed (μ̂ = Stage 4 bridge forecast,
  ``strategy.s2_model``). All share instruments, sizing and costs, so only the timing differs.
* Base and stress cost scenarios; break-even cost multiples; metrics (summary, MPPM, CE gains, Ledoit–Wolf vs
  S0, deflated Sharpe); attribution on SPX and PUT excess returns; stress table.
* 🚦 gates: S0 monthly excess returns correlate positively with PUT-index excess returns; S0 cycle returns
  correlate positively with the paper variance-swap payoff; S2 is dropped unless it beats S1 net of costs.
Outputs: ``outputs/tables/strategy_*.csv`` and ``outputs/figures/strategy_*.png``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import metrics
from src.strategy.backtest import backtest, load_inputs
from src.strategy.costs import CostModel
from src.strategy.signals import expected_cost_pts, expected_pnl_pts, multipliers, sigma_hat
from src.utils.io import load_config, project_path


def window(cfg, final: bool):
    return ((cfg["oos"]["validate_end"], cfg["oos"]["holdout_end"]) if final
            else (cfg["oos"]["train_end"], cfg["oos"]["validate_end"]))


def signal_frames(panel, contracts, s0_full, costs, cfg, final: bool) -> dict[str, pd.DataFrame]:
    """m_t for S1 and S2 at every signal date (expanding estimates; only past information)."""
    s = cfg["strategy"]
    hedge_hist = -s0_full["cycles"].set_index("settle")["hedge_cost_pts"]
    exp_cost = expected_cost_pts(contracts, costs, hedge_hist)
    origins = contracts["signal_date"]
    out = {}
    mus = {"S1": panel["vrp_exante"]}
    if s["s2_model"]:
        name = "payoff_forecasts_holdout.parquet" if final else "payoff_forecasts_validation.parquet"
        fc = pd.read_parquet(project_path("data", "processed", name))
        mus["S2"] = fc[s["s2_model"]]
    for label, mu in mus.items():
        mu_o = mu.reindex(origins)
        sig = sigma_hat(panel, mu, origins, cfg)
        out[label] = multipliers(mu_o, sig, expected_pnl_pts(contracts, mu_o), exp_cost, cfg,
                                 vix=panel["vix"], cap_high_vol=bool(s["cap_high_vol"]))
    return out


def evaluate(runs: dict, bench: str, panel, cfg) -> pd.DataFrame:
    rows = []
    b = runs[bench]
    for label, r in runs.items():
        row = dict(strategy=label, **metrics.summary(r["monthly"], r["monthly_excess"]),
                   mppm=metrics.mppm(r["monthly"], r["rf_monthly"]),
                   months_invested=int((r["cycles"]["m"] > 0).sum()) if len(r["cycles"]) else 0,
                   cost_drag_ann=12 * r.get("cost_drag", np.nan))
        for g in cfg["strategy"]["crra_gammas"]:
            row[f"ce_gain_vs_{bench}_g{g}"] = metrics.ce_gain(r["monthly"], b["monthly"], g)
        if label != bench:
            lw = metrics.sharpe_diff_test(r["monthly_excess"], b["monthly_excess"])
            row.update(lw_t_vs_bench=lw["t"], lw_p_vs_bench=lw["p"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


def main(cfg=None, force=False, final: bool = False):
    if final:
        raise RuntimeError("final=True is reserved for the single Phase 7 holdout run (python -m src.final.holdout_run)")
    return run_strategy(cfg or load_config(), final=False)


def run_strategy(cfg, final: bool = False, out_dir=None):
    """Validation run, or (final=True, called only by src.final.holdout_run) the holdout run."""
    panel = pd.read_parquet(project_path("data", "processed", "panel_monthly.parquet"))
    contracts, daily, spx, rf = load_inputs(cfg)
    lo, hi = window(cfg, final)
    start = contracts.loc[contracts["signal_date"] > pd.Timestamp(lo), "signal_date"].min()
    base, stress = CostModel.from_config(cfg, "base"), CostModel.from_config(cfg, "stress")
    ones = pd.Series(1.0, index=contracts["signal_date"])

    s0_full = backtest(ones, contracts, daily, spx, rf, cfg, base, end=hi)      # 2008 → window end (history)
    sig = signal_frames(panel, contracts, s0_full, base, cfg, final)
    mults = {"S0": ones} | {k: v["m"] for k, v in sig.items()}

    runs, cost_rows = {}, []
    for label, m in mults.items():
        for scen, cm in [("base", base), ("stress", stress), ("gross", CostModel.zero())]:
            r = backtest(m, contracts, daily, spx, rf, cfg, cm, start=start, end=hi)
            if scen == "base":
                runs[label] = r
            cost_rows.append(dict(strategy=label, scenario=scen, **metrics.summary(r["monthly"], r["monthly_excess"])))
    gross = {row["strategy"]: row["mean_excess_ann"] for row in cost_rows if row["scenario"] == "gross"}
    for label in runs:
        runs[label]["cost_drag"] = (gross[label] - metrics.summary(runs[label]["monthly"], runs[label]["monthly_excess"])["mean_excess_ann"]) / 12

    table = evaluate(runs, "S0", panel, cfg)
    be = []
    for k in cfg["strategy"]["breakeven_cost_multiples"]:
        cm = base.scaled(k)
        res = {lab: backtest(m, contracts, daily, spx, rf, cfg, cm, start=start, end=hi) for lab, m in mults.items()}
        for lab, r in res.items():
            be.append(dict(cost_multiple=k, strategy=lab, **metrics.summary(r["monthly"], r["monthly_excess"]),
                           ce_gain_vs_S0_g3=metrics.ce_gain(r["monthly"], res["S0"]["monthly"], 3)))
    be = pd.DataFrame(be)

    # attribution and gates
    s0 = runs["S0"]["monthly_excess"]
    fac = pd.DataFrame({"spx_excess": panel["exret_1"], "put_excess": panel["put_ret_fwd1"] - panel["rf"]})
    fac.index = (fac.index.to_period("M") + 1)                  # month-end t → return over month t+1
    attr = {lab: metrics.attribution(r["monthly_excess"], fac.reindex(r["monthly_excess"].index).dropna())
            for lab, r in runs.items()}
    corr_put = float(pd.concat([s0, fac["put_excess"]], axis=1).dropna().corr().iloc[0, 1])
    cyc = runs["S0"]["cycles"].set_index("signal_date")
    corr_vs = float(pd.concat([cyc["cycle_ret"], panel["vrp_expost"]], axis=1).dropna().corr().iloc[0, 1])
    gates = [("S0 correlates positively with PUT index", corr_put > 0, f"corr {corr_put:.2f}"),
             ("S0 cycle returns track the variance-swap payoff", corr_vs > 0, f"corr {corr_vs:.2f}")]
    if "S2" in runs:
        better = table.loc["S2", "sharpe"] > table.loc["S1", "sharpe"] and \
            table.loc["S2", "ce_gain_vs_S0_g3"] > table.loc["S1", "ce_gain_vs_S0_g3"]
        gates.append(("S2 beats S1 net of costs (else drop S2 before the holdout)", bool(better),
                      f"Sharpe {table.loc['S2', 'sharpe']:.2f} vs {table.loc['S1', 'sharpe']:.2f}"))

    stress_rows = []
    months = [pd.Period(str(m), "M") for m in cfg["strategy"]["stress_months"]]
    for lab, r in {"S0 (2008→)": s0_full} | runs.items():
        for mth in months:
            if mth in r["monthly"].index:
                stress_rows.append(dict(strategy=lab, month=str(mth), ret=r["monthly"].loc[mth]))

    out = out_dir or project_path("outputs", "tables")
    tag = "holdout" if final else "validation"
    table.to_csv(out / f"strategy_{tag}.csv")
    pd.DataFrame(cost_rows).to_csv(out / "strategy_costs.csv", index=False)
    be.to_csv(out / "strategy_breakeven.csv", index=False)
    pd.concat(attr, names=["strategy"]).to_csv(out / "strategy_attribution.csv")
    pd.DataFrame(stress_rows).to_csv(out / "strategy_stress.csv", index=False)
    pd.DataFrame({lab: r["monthly"] for lab, r in runs.items()}).to_csv(out / f"strategy_monthly_{tag}.csv")
    for lab, s in sig.items():
        s.to_csv(out / f"strategy_signals_{lab}.csv")
    fig_dir = out if (final and out_dir is not None) else project_path("outputs", "figures")
    figures(runs, be, fig_dir, suffix="_holdout" if final else "")
    print(table.round(3).to_string())
    print("🚦 Strategy gates:")
    for n, ok, d in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}")
    return table, gates


def figures(runs, be, out_dir, suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for lab, r in runs.items():
        w = (1 + r["monthly"]).cumprod()
        ax.plot(w.index.to_timestamp(), w, label=lab)
    ax.set_title(f"Cumulative value, {'holdout' if suffix else 'validation'} window (net of base costs)"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"strategy_cum_pnl{suffix}.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for lab, r in runs.items():
        w = (1 + r["monthly"]).cumprod()
        ax.plot(w.index.to_timestamp(), w / w.cummax() - 1, label=lab)
    ax.set_title("Drawdown"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"strategy_drawdown{suffix}.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, g in be.groupby("strategy"):
        ax.plot(g["cost_multiple"], g["sharpe"], marker="o", label=lab)
    ax.set_xlabel("cost multiple of base"); ax.set_ylabel("Sharpe (ann.)"); ax.set_title("Break-even costs"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"strategy_cost_breakeven{suffix}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
