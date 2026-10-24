"""Part 7: current-regime readout -> output/tables/07_*, readout_summary.json, output/figures/07_*.

Usage:  .venv/bin/python scripts/run_readout.py [--today YYYY-MM-DD]
Requires build_data.py, run_model.py and run_regime.py. Descriptive only; spot proxy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as RB  # noqa: E402

from oilrv import backtest as B  # noqa: E402
from oilrv import readout as RO  # noqa: E402
from oilrv import sizing as SZ  # noqa: E402
from oilrv.calendars import load_events  # noqa: E402
from oilrv.config import load_config, path  # noqa: E402
from oilrv.data import load_clean, load_features  # noqa: E402
from oilrv.plotting import CRITICAL, REF_LINE, SERIES, TAB_DIR, save, save_table, use_style  # noqa: E402

LABELS = {"crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}
PLAN_DATES = [
    {"date": "2026-11-01", "item": "US clocks change: NYMEX settle moves to 3:30 am SGT, US open to 10:30 pm SGT",
     "source": "oil-rv-plan.md (Step 10)"},
    {"date": "2026-11-10", "item": "Next EIA Short-Term Energy Outlook",
     "source": "https://www.eia.gov/outlooks/steo/data.php (via oil-rv-plan.md)"},
    {"date": "2026-11-30", "item": "End of the Atlantic hurricane season", "source": "NOAA convention (oil-rv-plan.md)"},
    {"date": "2027-02-26", "item": "G7 100 mb stock release expected to run through February 2027 (diesel front-loaded)",
     "source": "https://www.hellenicshippingnews.com/?p=1150554 (via oil-rv-plan.md)"},
]


def main(today: str | None = None) -> dict:
    use_style()
    cfg = load_config()
    today = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    _, sp = load_clean()
    feats = load_features()
    cal = pd.read_csv(path("clean_dir") / "event_features.csv", index_col=0, parse_dates=True)
    events = load_events()
    W0 = RB.load_base_world(sp, feats, cal, cfg)
    asof = sp.index[-1]
    window = cfg["model"]["window"]

    # volatility percentiles: each spread's own GARCH path, plus the diesel crack (HO - CL)
    vol_pct = {k: RO.expanding_pct(W0.model[k][1]["sigma_next"]) for k in W0.model}
    diesel = (sp["ho_bbl"] - sp["cl_bbl"]).rename("diesel_crack")
    diesel_sig = RO.garch_path(diesel, window)
    diesel_pct = RO.expanding_pct(diesel_sig)

    closures = events.loc[(events["type"] == "HORMUZ") & events["name"].str.startswith("closure"), "date"]
    last_closure = closures[closures <= today].max()
    days_closure_today = float((today - last_closure).days)

    state, trig = {}, {}
    for key in W0.model:
        name = key.split("|")[1]
        sim = W0.sim(key, B.Rules())
        for i in range(sim.first, len(sim.idx)):
            sim.step(i)
        d, _ = sim.results()
        last = d.iloc[-1]
        fit = sim.fit
        M = sim.M
        # integer size if a signal were taken today: at max(|z|, entry_z); none possible at or past the stop
        az = abs(last["z"]) if np.isfinite(last["z"]) else np.nan
        z_size = max(az, cfg["strategy"]["entry_z"]) if np.isfinite(az) else np.nan
        if np.isfinite(z_size) and z_size < cfg["strategy"]["stop_z"] and RB.B._ok(fit):
            isz = SZ.integer_units(last["sigma_next"], z_size, fit["sd"], M, cfg=cfg)
        else:
            isz = {"vol_limit": SZ.vol_limit(cfg["sizing"]["risk_per_day_usd"], last["sigma_next"], M),
                   "stop_limit": np.nan, "units": 0}
        vp = float(vol_pct[key].iloc[-1])
        rel_pct = float(diesel_pct.iloc[-1]) if name == "ho_rb" else vp
        state[key] = {
            "as_of": str(asof.date()), "S": float(sim.S(fit, len(sim.idx) - 1)) if RB.B._ok(fit) else np.nan,
            "z": float(last["z"]), "sigma_next": float(last["sigma_next"]), "sigma_pct_history": vp,
            "half_life": float(fit.get("half_life", np.nan)), "adf_p": float(fit.get("adf_p", np.nan)),
            "gate_open": bool(last["gate_open"]), "p_revert": float(last["p_hat"]), "armed": bool(sim.armed),
            "in_trade": bool(sim.side != 0), "entry_pending": bool(sim.pending is not None and sim.pending[0] == "enter"),
            "blackout_today": bool(W0.blackout.iloc[-1]),
            "integer_vol_limit": float(isz["vol_limit"]), "integer_stop_limit_at_max_z_entry": float(isz["stop_limit"]),
            "integer_units_at_max_z_entry": int(isz["units"]), "research_units": float(SZ.research_units(last["sigma_next"], M, cfg)),
        }
        if key.startswith("fixed|"):
            trig[key] = RO.triggers(name, last["p_hat"], rel_pct, days_closure_today, last["z"], "spot", cfg)
            trig[key]["relevant_vol"] = "diesel crack (HO − CL)" if name == "ho_rb" else name
            trig[key]["relevant_vol_pct"] = rel_pct
    st = pd.DataFrame(state).T
    save_table(st, "07_state.csv")
    tt = pd.DataFrame({k: {**v["conditions"], "ALL MET": v["all_met"], "vol series": v["relevant_vol"],
                           "vol percentile": round(v["relevant_vol_pct"], 3)} for k, v in trig.items()}).T
    save_table(tt, "07_triggers.csv")
    wl = RO.watch_list(events, today, PLAN_DATES)
    save_table(wl.set_index("date"), "07_watch_list.csv")

    out = {"source": "spot", "data_as_of": str(asof.date()), "today": str(today.date()),
           "last_hormuz_closure": str(last_closure.date()), "days_since_hormuz_closure_today": days_closure_today,
           "diesel_crack": {"level": float(diesel.iloc[-1]), "sigma_next": float(diesel_sig.iloc[-1]),
                            "sigma_pct_history": float(diesel_pct.iloc[-1])},
           "state": state, "triggers": trig,
           "spot_proxy_gate_verdicts": {k: v["PASS"] for k, v in
                                        json.loads((TAB_DIR / "backtest_summary.json").read_text())["gate_verdicts"].items()}}

    # figure: the last ~16 months
    start = pd.Timestamp("2025-07-01")
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for a, key in zip(ax[:3], [k for k in W0.model if k.startswith("fixed|")]):
        p = W0.model[key][1].loc[start:]
        a.plot(p.index, p["z"].clip(-8, 8), color=SERIES[0], lw=0.9, label="z (deseasonalised)")
        for v, c in [(2, REF_LINE), (4, CRITICAL)]:
            a.axhline(v, color=c, lw=0.7, ls="--")
            a.axhline(-v, color=c, lw=0.7, ls="--")
        a2 = a.twinx()
        a2.plot(p.index, W0.p[key].loc[start:], color=SERIES[2], lw=0.9, label="P(revert)")
        rel = diesel_pct if key.endswith("ho_rb") else vol_pct[key]
        a2.plot(rel.loc[start:].index, rel.loc[start:], color=SERIES[1], lw=0.9,
                label="vol percentile (" + ("diesel crack" if key.endswith("ho_rb") else "own") + ")")
        a2.axhline(0.6, color=SERIES[2], lw=0.6, ls=":")
        a2.axhline(0.8, color=SERIES[1], lw=0.6, ls=":")
        a2.set_ylim(0, 1.05)
        a2.grid(False)
        for c in closures[closures >= start]:
            a.axvline(c, color=CRITICAL, lw=0.6, alpha=0.6)
        a.set_ylim(-8.5, 8.5)
        name = key.split("|")[1]
        a.set_title(f"{LABELS[name]}: z (left), P(revert) and volatility percentile (right); red lines = Hormuz closures")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, loc="upper left", ncols=3, fontsize=7)
    ax[3].plot(diesel.loc[start:].index, diesel.loc[start:], color=SERIES[0], lw=0.9, label="Diesel crack HO − CL ($/bbl)")
    ax[3].plot(sp.loc[start:].index, sp.loc[start:, "crack321"], color=SERIES[1], lw=0.9, label="3-2-1 crack")
    ax[3].plot(sp.loc[start:].index, sp.loc[start:, "rb_bbl"] - sp.loc[start:, "cl_bbl"], color=SERIES[2], lw=0.9,
               label="Gasoline crack RB − CL")
    for c in closures[closures >= start]:
        ax[3].axvline(c, color=CRITICAL, lw=0.6, alpha=0.6)
    ax[3].set_ylabel("$/bbl")
    ax[3].set_title("Spot cracks since July 2025")
    ax[3].legend(loc="upper left", ncols=3, fontsize=7.5)
    fig.tight_layout()
    save(fig, "07_current_regime.png")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "readout_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=None)
    print(json.dumps(main(ap.parse_args().today), indent=2, default=float))
