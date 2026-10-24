"""Part 6: walk-forward backtest, robustness, integer book and the pre-registered gate verdicts.

Usage:  .venv/bin/python scripts/run_backtest.py
Requires build_data.py, run_model.py and run_regime.py. About 3-5 minutes.

Every result here is a SPOT PROXY (PREREGISTRATION.md section 5): the binding verdict needs the
Bloomberg futures run. Gated configuration = `fixed` variant, ladder step `logit_filter`, research
P&L stream, base parameters, $0.05/bbl per side.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilrv import backtest as B
from oilrv import metrics as MT
from oilrv import regime as R
from oilrv.config import load_config, path
from oilrv.data import load_clean, load_features
from oilrv.model import build_model, legs_frame
from oilrv.plotting import CRITICAL, REF_LINE, SERIES, TAB_DIR, save, save_table, use_style

LABELS = {"crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}
COL = "pnl_usd"


class World:
    """Model fits, caches and daily P(revert) for one data set and one estimation window."""

    def __init__(self, sp, feats, cal, cfg, window, model=None, p_daily=None, horizon=None):
        self.sp, self.cfg, self.window = sp, cfg, window
        self.legs = legs_frame(sp)
        self.blackout = cal["event_blackout"].reindex(sp.index).fillna(False)
        if model is None:
            built = build_model(sp, cfg, window)
            model = {f"{v}|{s}": fp for (v, s), fp in built.items()}
        self.model = model
        if p_daily is None:
            _, _, p_daily, _ = R.build_regime(sp, feats, cal, model, cfg, horizon)
        self.p = {k: v["p_hat"] for k, v in p_daily.items()}
        self.caches = {}
        for key, (fits, _) in model.items():
            v, s = key.split("|")
            c = B.FitCache(self.legs, s, cfg["variants"][v]["weights"] if v == "fixed" else "johansen", cfg)
            c.seed(fits, window)
            self.caches[key] = c

    def run(self, key, rules: B.Rules, mode="research"):
        name = key.split("|")[1]
        rules.window = self.window
        rules.variant = key.split("|")[0]
        return B.run_backtest(name, self.legs, self.caches[key], self.p[key], self.blackout, rules, self.cfg, mode)

    def sim(self, key, rules: B.Rules):
        name = key.split("|")[1]
        rules.window, rules.variant = self.window, key.split("|")[0]
        return B.SpreadSim(name, self.legs, self.caches[key], self.p[key], self.blackout, rules, self.cfg)


def load_base_world(sp, feats, cal, cfg):
    d = path("derived_dir")
    fits = pd.read_csv(d / "rolling_fits.csv", index_col=[0, 1], parse_dates=["date"])
    paths = pd.read_csv(d / "model_path.csv", index_col=[0, 1], parse_dates=["date", "fit_date"])
    model = {k: (fits.loc[k], paths.loc[k]) for k in fits.index.get_level_values(0).unique()}
    pr = pd.read_csv(d / "regime_probability.csv", index_col=[0, 1], parse_dates=["date"])
    p_daily = {k: pr.loc[k][["p_hat"]] for k in pr.index.get_level_values(0).unique()}
    return World(sp, feats, cal, cfg, cfg["model"]["window"], model, p_daily)


def main() -> dict:
    use_style()
    cfg = load_config()
    _, sp = load_clean()
    feats = load_features()
    cal = pd.read_csv(path("clean_dir") / "event_features.csv", index_col=0, parse_dates=True)
    W0 = load_base_world(sp, feats, cal, cfg)
    keys = list(W0.model)
    fixed = [k for k in keys if k.startswith("fixed|")]
    periods = cfg["evaluation"]["sub_periods"]
    out = {"source": "spot", "binding": False,
           "note": "Spot proxy. The binding verdict needs the Bloomberg futures run (PREREGISTRATION.md section 5)."}

    # ---------------------------------------------------------------- ladder (all keys)
    runs, rows = {}, {}
    for key in keys:
        for step in cfg["ladder"]:
            d, t = W0.run(key, B.Rules.ladder(step, cfg))
            runs[(key, step)] = (d, t)
            s = MT.summarize(d, t, COL)
            s_unit = MT.summarize(d, t, "pnl_bbl")
            rows[(key, step)] = {**s, "sharpe_unit": s_unit["sharpe"], "total_bbl_per_unit": s_unit["total"]}
    ladder = pd.DataFrame(rows).T.rename_axis(["key", "step"])
    save_table(ladder, "06_ladder_summary.csv")
    out["ladder"] = {f"{k}|{s}": v for (k, s), v in ladder[["sharpe", "sharpe_t", "total", "max_dd", "n_trades",
                                                              "win_rate", "time_in_market"]].astype(float).round(3).to_dict("index").items()}
    gated = {k: runs[(k, "logit_filter")] for k in keys}
    unfiltered = {k: runs[(k, "blackout")] for k in keys}

    for k, (d, t) in gated.items():
        d.to_csv(path("derived_dir") / f"backtest_daily_{k.replace('|', '_')}.csv")
        t.to_csv(TAB_DIR / f"06_trades_{k.replace('|', '_')}.csv", index=False)
    ex = pd.DataFrame({f"{k}|{s}": MT.exit_reasons(runs[(k, s)][1]) for k in keys for s in ["blackout", "logit_filter"]}).fillna(0).astype(int)
    save_table(ex, "06_exit_reasons.csv")

    # ---------------------------------------------------------------- sub-periods and stress windows
    sub = pd.concat({f"{k}|{lab}": MT.subperiods(runs[(k, lab)][0], COL, periods)
                     for k in keys for lab in ["blackout", "logit_filter"]}, names=["run", "period"])
    save_table(sub, "06_subperiods.csv")
    st_rows = {}
    for w, (a, b) in cfg["evaluation"]["stress_windows"].items():
        for k in keys:
            st_rows[(w, k)] = {"filtered": MT.window_pnl(gated[k][0], COL, a, b),
                               "unfiltered": MT.window_pnl(unfiltered[k][0], COL, a, b),
                               "filtered_trades": int(gated[k][1]["fill_date"].between(a, b or "2100").sum()) if len(gated[k][1]) else 0,
                               "unfiltered_trades": int(unfiltered[k][1]["fill_date"].between(a, b or "2100").sum()) if len(unfiltered[k][1]) else 0}
    stress = pd.DataFrame(st_rows).T.rename_axis(["window", "key"])
    save_table(stress, "06_stress_windows.csv")

    # ---------------------------------------------------------------- entry x exit grid and cost sweep (gated config)
    rb = cfg["robustness"]
    grid_rows, cost_rows = {}, {}
    for k in keys:
        for ez in rb["entry_z"]:
            for xz in rb["exit_z"]:
                if ez == cfg["strategy"]["entry_z"] and xz == cfg["strategy"]["exit_z"]:
                    d, t = gated[k]
                else:
                    d, t = W0.run(k, B.Rules(entry_z=ez, exit_z=xz))
                s = MT.summarize(d, t, COL)
                grid_rows[(k, ez, xz)] = {"sharpe": s["sharpe"], "total": s["total"], "n_trades": s["n_trades"]}
        for c in rb["cost_usd_per_bbl"]:
            d, t = W0.run(k, B.Rules(cost_bbl=c))
            s = MT.summarize(d, t, COL)
            cost_rows[(k, c)] = {"sharpe": s["sharpe"], "total": s["total"], "n_trades": s["n_trades"]}
    grid = pd.DataFrame(grid_rows).T.rename_axis(["key", "entry_z", "exit_z"])
    save_table(grid, "06_entry_exit_grid.csv")
    costs = pd.DataFrame(cost_rows).T.rename_axis(["key", "cost_usd_per_bbl"])
    save_table(costs, "06_cost_sweep.csv")

    # ---------------------------------------------------------------- gate verdicts (fixed variant only)
    verdicts = {}
    for k in fixed:
        d, t = gated[k]
        summ = MT.summarize(d, t, COL)
        subp = MT.subperiods(d, COL, periods)
        g4 = {ez: float(grid.loc[(k, ez, cfg["strategy"]["exit_z"]), "sharpe"]) for ez in rb["entry_z"]}
        g5 = {w: {"filtered": float(stress.loc[(w, k), "filtered"]), "unfiltered": float(stress.loc[(w, k), "unfiltered"])}
              for w in cfg["gates"]["stress_filter"]["windows"]}
        verdicts[k] = MT.gate_verdict(summ, subp, g4, g5, cfg)
    vt = pd.DataFrame({k: {kk: (json.dumps(v, default=float) if isinstance(v, dict) else v) for kk, v in d.items()}
                       for k, d in verdicts.items()}).T
    save_table(vt, "06_gate_verdicts.csv")
    out["gate_verdicts"] = verdicts

    # ---------------------------------------------------------------- robustness: windows, label horizons, April 2020
    rob = {}
    for wlen in rb["window"]:
        if wlen == cfg["model"]["window"]:
            continue
        Ww = World(sp, feats, cal, cfg, wlen)
        for k in keys:
            d, t = Ww.run(k, B.Rules())
            rob[(f"window {wlen}", k)] = MT.summarize(d, t, COL)
    for h in rb["label_horizon_days"]:
        if h == cfg["regime"]["label_horizon_days"]:
            continue
        Wh = World(sp, feats, cal, cfg, cfg["model"]["window"], W0.model, None, horizon=h)
        for k in keys:
            d, t = Wh.run(k, B.Rules())
            rob[(f"label {h}d", k)] = MT.summarize(d, t, COL)
    if rb.get("drop_flagged_dates"):
        spx = sp[~sp["flag_apr2020"].astype(bool)]
        Wx = World(spx, feats, cal, cfg, cfg["model"]["window"])
        for k in keys:
            for step in ["blackout", "logit_filter"]:
                d, t = Wx.run(k, B.Rules.ladder(step, cfg))
                rob[(f"drop Apr 2020 week, {step}", k)] = MT.summarize(d, t, COL)
    for k in keys:
        rob[("base (gated config)", k)] = MT.summarize(*gated[k], COL)
        rob[("base, no logit (blackout step)", k)] = MT.summarize(*unfiltered[k], COL)
    robt = pd.DataFrame(rob).T.rename_axis(["run", "key"])
    save_table(robt, "06_robustness.csv")

    # ---------------------------------------------------------------- integer book ($2m, drawdown rules)
    book_res = {}
    for lab, step in [("gated", "logit_filter"), ("no logit", "blackout")]:
        sims = {k: W0.sim(k, B.Rules.ladder(step, cfg)) for k in fixed}
        book, res = B.run_book(sims, cfg)
        book.to_csv(path("derived_dir") / f"integer_book_{lab.replace(' ', '_')}.csv")
        tr = pd.concat([res[k][1].assign(key=k) for k in fixed if len(res[k][1])], ignore_index=True) if any(len(res[k][1]) for k in fixed) else pd.DataFrame()
        book_res[lab] = {"final_equity": float(book["equity"].iloc[-1]), "total": float(book["pnl_usd"].sum()),
                         "sharpe": MT.sharpe(book["pnl_usd"]), "max_dd_usd": MT.max_drawdown(book["pnl_usd"]),
                         "max_drawdown_pct": float(book["drawdown"].max()), "halted": bool(book["halted"].any()),
                         "n_trades": int(len(tr)), "blocked_by_size": int(sum(res[k][0].attrs["blocked_by_size"] for k in fixed)),
                         "days_half_size": int((book["size_mult"] == 0.5).sum())}
        if len(tr):
            tr.to_csv(TAB_DIR / f"06_integer_book_trades_{lab.replace(' ', '_')}.csv", index=False)
    save_table(pd.DataFrame(book_res).T, "06_integer_book.csv")
    out["integer_book"] = book_res

    # ---------------------------------------------------------------- figures
    last = sp.index[-1]
    fig, ax = plt.subplots(len(keys), 1, figsize=(11, 3.1 * len(keys)), sharex=True)
    for a, k in zip(ax, keys):
        for i, step in enumerate(cfg["ladder"]):
            d = runs[(k, step)][0]
            a.plot(d.index, d[COL].cumsum() / 1e3, color=SERIES[i], lw=1, label=step)
        a.axhline(0, color=REF_LINE, lw=0.7)
        for a0, b0 in cfg["evaluation"]["stress_windows"].values():
            a.axvspan(pd.Timestamp(a0), pd.Timestamp(b0) if b0 else last, color=REF_LINE, alpha=0.10, lw=0)
        v = verdicts.get(k)
        tag = (" — PASS" if v["PASS"] else " — FAIL") if v else " (robustness, not gated)"
        a.set_title(f"{k}: cumulative net P&L, research stream (\\$k, R = \\$6k/day){tag}")
        a.set_ylabel("\\$k")
        a.legend(loc="upper left", ncols=4, fontsize=7.5)
    fig.tight_layout()
    save(fig, "06_cumulative_pnl.png")

    fig, ax = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.2))
    for a, k in zip(ax, keys):
        m = grid.loc[k, "sharpe"].unstack().astype(float)
        im = a.imshow(m.to_numpy(), cmap="RdBu", vmin=-1.5, vmax=1.5)
        a.set_xticks(range(m.shape[1]), [f"{x:g}" for x in m.columns])
        a.set_yticks(range(m.shape[0]), [f"{x:g}" for x in m.index])
        a.set_xlabel("exit |z|")
        a.set_ylabel("entry |z|")
        a.grid(False)
        for (r_, c_), v in np.ndenumerate(m.to_numpy()):
            a.text(c_, r_, "–" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
        a.set_title(k, fontsize=9)
    fig.suptitle("Net Sharpe, gated configuration, entry × exit grid", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "06_entry_exit_grid.png")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for lab, ls in [("gated", "-"), ("no logit", "--")]:
        b = pd.read_csv(path("derived_dir") / f"integer_book_{lab.replace(' ', '_')}.csv", index_col=0, parse_dates=True)
        ax.plot(b.index, b["equity"] / 1e6, lw=1, ls=ls, color=SERIES[0] if lab == "gated" else SERIES[1],
                label=f"Integer book, {lab}")
    ax.axhline(cfg["sizing"]["book_usd"] / 1e6, color=REF_LINE, lw=0.7)
    ax.set_ylabel("Equity (\\$m)")
    ax.set_title("Integer book: \\$2m, R = \\$6k/day, L = \\$40k/trade, ≤ 3 units per spread, drawdown rules")
    ax.legend(loc="upper left")
    save(fig, "06_integer_book.png")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "backtest_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
