"""Check that the numbers quoted in report/report.md match the current output files.

Usage:  .venv/bin/python scripts/check_report.py
Each check formats a value from output/tables (or data/derived) the way the report writes it and
asserts that the string appears in the report. Exit code 1 lists any mismatch.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from oilrv.config import ROOT

T = ROOT / "output" / "tables"
D = ROOT / "data" / "derived"
R = (ROOT / "report" / "report.md").read_text()


def tab(name, **kw):
    return pd.read_csv(T / name, **kw)


def neg(x: float, nd: int = 2) -> str:
    """Report style: typographic minus."""
    return f"{x:.{nd}f}".replace("-", "−")


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def k(x: float) -> str:
    """Thousands with one decimal, e.g. 46.9k."""
    return f"{abs(x) / 1e3:.1f}k"


checks: list[tuple[str, str]] = []
add = lambda label, s: checks.append((label, s))

# ---------------------------------------------------------------- data
ds = json.loads((T / "data_summary.json").read_text())
add("days", f"**{ds['n_days']:,}** days")
add("wti min", f"−\\${abs(ds['wti_min']):.2f}")
add("crack latest", f"\\${ds['latest']['crack321']:.2f}/bbl")
add("crack median", f"\\${ds['crack321_median']:.2f}")
add("opec events", f"{ds['events']['OPEC']} OPEC/OPEC+ events")
add("opec unverified", f"{ds['opec_events_verified']['no']} of the {ds['events']['OPEC']} OPEC dates")
add("hurricanes", f"{ds['events']['HURRICANE']} hurricane-strength")
add("blackout share", f"{ds['pct_days_in_event_blackout']}% of days")

# ---------------------------------------------------------------- moments
m = tab("02_moments.csv", index_col=0)
mx = tab("02_moments_excl_apr2020.csv", index_col=0)
for s, lab in [("crack321", "3-2-1 crack"), ("brent_wti", "Brent–WTI"), ("ho_rb", "HO − RB")]:
    row = (f"| {lab} | {m.loc[s, 'std']:.2f} | {m.loc[s, 'excess_kurt']:.1f} | {mx.loc[s, 'excess_kurt']:.1f} | "
           f"{m.loc[s, 't_df']:.1f} | {pct(m.loc[s, 'p_gt_4sd_obs'], 2)} |")
    add(f"moments {s}", row)
bp = tab("02_moments_by_period.csv", index_col=[0, 1])["std"]
add("crack sd 2014-19", f"\\${bp[('crack321', '2014–2019')]:.2f} in 2014–19")
add("crack sd 2026", f"\\${bp[('crack321', '2026')]:.2f} in 2026")
add("ho sd 2022", f"\\${bp[('ho_rb', '2022')]:.2f} in 2022")
add("ho sd 2026", f"\\${bp[('ho_rb', '2026')]:.2f} in 2026")
st = tab("02_stl_strength.csv", index_col=0)
add("crack kruskal", f"| {st.loc['crack321', 'kruskal_p_month']:.3f} |")
add("bw kruskal", f"| {st.loc['brent_wti', 'kruskal_p_month']:.2f} |")

# ---------------------------------------------------------------- diagnostics
ur = tab("03_unit_roots.csv", index_col=[0, 1])
add("ulsd adf", f"(p = {ur.loc[('ho', 'level'), 'adf_p']:.2f})")
add("crack adf", f"the crack (p = {ur.loc[('crack321', 'level'), 'adf_p']:.3f})")
add("bw adf", f"Brent–WTI (p = {ur.loc[('brent_wti', 'level'), 'adf_p']:.3f})")
add("horb adf", f"borderline (p = {ur.loc[('ho_rb', 'level'), 'adf_p']:.2f})")
ss = tab("03_stationarity_summary.csv", index_col=[0, 1])
for s, lab in [("crack321", "(crack)"), ("brent_wti", "(Brent–WTI)"), ("ho_rb", "(HO − RB)")]:
    add(f"adf pass {s}", f"**{pct(ss.loc[(s, 'all'), 'adf_pass_rate'])}** {lab}")
add("hl band rates", " and ".join([", ".join(pct(ss.loc[(s, 'all'), 'hl_in_band_rate']) for s in ["crack321", "brent_wti"]),
                                   pct(ss.loc[("ho_rb", "all"), "hl_in_band_rate"])]))
sv = tab("03_sarima_vs_naive.csv", index_col=0)
add("sarima vs snaive", f"RMSE ratios {sv.loc['crack321', 'ratio_vs_snaive']:.2f} and {sv.loc['ho_rb', 'ratio_vs_snaive']:.2f}")
add("sarima vs rw", f"RMSE ratios {sv.loc['crack321', 'ratio_vs_rw']:.2f} and {sv.loc['ho_rb', 'ratio_vs_rw']:.2f}")
add("sarima dm", f"p = {sv.loc['crack321', 'dm_vs_rw_p']:.2f} and {sv.loc['ho_rb', 'dm_vs_rw_p']:.2f}")
add("n sarima", f"{int(sv.loc['crack321', 'n_forecasts'])} forecasts")
eg = tab("03_engle_granger.csv", index_col=0)
an = json.loads((T / "analysis_summary.json").read_text())["part3"]
for row, pair, lab in [("brent_on_wti", "brent_wti", "Brent on WTI"), ("ho_on_rb", "ho_rb", "HO on RB")]:
    re_ = an["rolling_eg"][pair]
    add(f"eg {row}", f"| {lab} | {eg.loc[row, 'beta']:.2f} | {eg.loc[row, 'eg_p']:.3f} | {eg.loc[row, 'adf_p_fixed_1to1']:.3f} | "
                     f"{pct(re_['eg_pass_rate'])} | {re_['beta_range'][0]:.2f} – {re_['beta_range'][1]:.2f} |")
rj = an["rolling_johansen"]["rank_share"]
add("johansen ranks", f"It is 0 in {pct(rj['0'])} of months, 1 in {pct(rj['1'])}, 2 in only {pct(rj['2'])} and 3 in {pct(rj['3'])}")
add("johansen dist", f"The median distance is {an['rolling_johansen']['median_dist_crack321']:.2f}")

# ---------------------------------------------------------------- model
g = tab("04_gate_and_signal_rates.csv", index_col=0)
add("signal rates", f"|z| ≥ 2 holds on {pct(g.loc['fixed|crack321', 'share_days_abs_z_ge_2'])} (crack), "
                    f"{pct(g.loc['fixed|brent_wti', 'share_days_abs_z_ge_2'])} (Brent–WTI) and "
                    f"{pct(g.loc['fixed|ho_rb', 'share_days_abs_z_ge_2'])} (HO − RB)")
add("johansen usable", f"usable weights in only **{pct(g.loc['johansen|crack321', 'weights_ok_rate'])}**")
add("johansen gate", f"in only {pct(g.loc['johansen|crack321', 'gate_open_rate'])} of months")
for s, lab in [("fixed|crack321", "3-2-1 crack"), ("fixed|brent_wti", "Brent–WTI"), ("fixed|ho_rb", "HO − RB")]:
    add(f"garch fallback {s}", f"| {lab} | {pct(1 - g.loc[s, 'garch_ok_rate'])}")
vt = tab("04_vol_forecast_test.csv", index_col=0)
add("qlike ho", f"t = {neg(vt.loc['fixed|ho_rb', 't_stat'])}")
add("qlike crack", f"t = {neg(vt.loc['fixed|crack321', 't_stat'])}")
add("qlike bw", f"t = {neg(vt.loc['fixed|brent_wti', 't_stat'])}")
vk = json.loads((T / "model_summary.json").read_text())["vecm_known_vector"]
add("rb corrects", f"RB corrects it in {pct(vk['crack321']['share_t_lt_-2_rb'])}")
add("ho corrects crack", f"HO in {pct(vk['crack321']['share_t_lt_-2_ho'])}")
add("cl corrects", f"CL corrects in {pct(vk['crack321']['share_t_gt_2_cl'])}")
add("brent corrects", f"Brent does the correcting in {pct(vk['brent_wti']['share_t_lt_-2_brent'])}")
add("wti corrects", f"against {pct(vk['brent_wti']['share_t_gt_2_cl'])} for WTI")
add("ho corrects horb", f"HO corrects in {pct(vk['ho_rb']['share_t_lt_-2_ho'])} of windows, RB in {pct(vk['ho_rb']['share_t_gt_2_rb'])}")

# ---------------------------------------------------------------- regime
es = tab("05_episodes_summary.csv", index_col=0)
fe = tab("05_filter_eval.csv", index_col=0)
for s, lab in [("crack321", "3-2-1 crack"), ("brent_wti", "Brent–WTI"), ("ho_rb", "HO − RB")]:
    key = f"fixed|{s}"
    add(f"episodes {s}", f"| {lab} | {int(es.loc[key, 'episodes'])} | {pct(es.loc[key, 'share_reverted'])} | {fe.loc[key, 'auc']:.2f} |")
pool = fe.loc["pooled (fixed)"]
add("pooled", f"| **Pooled** | **{int(pool['episodes_total'])}** | **{pct(pool['success_all_oos'])}** (out of sample) | **{pool['auc']:.2f}** |")
add("brier", f"Its Brier score ({pool['brier_model']:.3f}) is worse than a point-in-time base rate ({pool['brier_base_rate']:.3f})")
add("passed", f"only {int(pool['n_pass_filter'])} of {int(pool['episodes_with_p'])} episodes passed")
rr = tab("05_robustness.csv", index_col=0)
lo, hi = rr["auc"].min(), rr["auc"].max()
add("auc range", f"AUC {lo:.2f}–{hi:.2f}")
add("40d base", f"rises to {pct(rr.loc['label horizon 40d', 'success_all_oos'])}")
rs = json.loads((T / "regime_summary.json").read_text())
add("first filter month", "September 2014" if rs["first_month_with_filter"].startswith("2014-09") else rs["first_month_with_filter"])
pr = pd.read_csv(D / "regime_probability.csv", index_col=[0, 1])
share = {k_: (pr.loc[k_, "p_hat"] >= 0.6).mean() for k_ in ["fixed|crack321", "fixed|brent_wti", "fixed|ho_rb"]}
add("p>=0.6 crack", f"only {pct(share['fixed|crack321'])} of days for the crack and Brent–WTI ({pct(share['fixed|brent_wti'])} for Brent–WTI alone)")
add("p>=0.6 horb", f"and {pct(share['fixed|ho_rb'])} for HO − RB")
ep = pd.read_csv(D / "regime_episodes.csv", index_col=0, parse_dates=True)
e26 = ep[ep["key"].str.startswith("fixed|")].loc["2026"]
add("2026 blocked", f"In 2026, {int(e26['reverted'].sum())} of the {len(e26)} blocked episodes would have reverted")

# ---------------------------------------------------------------- backtest
v = tab("06_gate_verdicts.csv", index_col=0)
for s, lab in [("crack321", "3-2-1 crack"), ("brent_wti", "Brent–WTI"), ("ho_rb", "HO − RB")]:
    key = f"fixed|{s}"
    g4 = json.loads(v.loc[key, "G4_value"])
    mark = lambda b: "✓" if str(b) == "True" else "✗"
    row = (f"| {lab} | {neg(float(v.loc[key, 'G1_value']))} {mark(v.loc[key, 'G1_sharpe'])} | "
           f"{int(v.loc[key, 'G2_value'])} {mark(v.loc[key, 'G2_trades'])} | "
           f"{int(v.loc[key, 'G3_value'])} {mark(v.loc[key, 'G3_subperiods'])} | "
           + " / ".join(neg(g4[x]) for x in ["1.5", "2.0", "2.5"]) + f" {mark(v.loc[key, 'G4_grid'])} | "
           f"{mark(v.loc[key, 'G5_stress'])} | **{'PASS' if str(v.loc[key, 'PASS']) == 'True' else 'FAIL'}** |")
    add(f"verdict {s}", row)
lad = tab("06_ladder_summary.csv", index_col=[0, 1])
for key, lab in [("fixed|crack321", "3-2-1 crack"), ("fixed|brent_wti", "Brent–WTI"), ("fixed|ho_rb", "HO − RB"),
                 ("johansen|crack321", "Crack, `johansen` (not gated)")]:
    sh = [neg(lad.loc[(key, st_), "sharpe"]) for st_ in ["core", "garch_sizing", "blackout", "logit_filter"]]
    add(f"ladder {key}", f"| {lab} | " + " | ".join(sh) + f" | {int(lad.loc[(key, 'core'), 'n_trades'])} → "
                         f"{int(lad.loc[(key, 'logit_filter'), 'n_trades'])} |")
add("crack dd", f"from \\${abs(lad.loc[('fixed|crack321', 'core'), 'max_dd']) / 1e3:.0f}k to "
                f"\\${abs(lad.loc[('fixed|crack321', 'garch_sizing'), 'max_dd']) / 1e3:.0f}k")
add("bw dd", f"from \\${abs(lad.loc[('fixed|brent_wti', 'core'), 'max_dd']) / 1e3:.0f}k to "
             f"\\${abs(lad.loc[('fixed|brent_wti', 'garch_sizing'), 'max_dd']) / 1e3:.0f}k")
hb = lad.loc[("fixed|ho_rb", "blackout")]
add("horb unfiltered", f"Sharpe {hb['sharpe']:.2f} (t = {hb['sharpe_t']:.1f}), {pct(hb['win_rate'])} winning trades, "
                       f"\\${hb['total'] / 1e3:.0f}k over 16.7 years")
sub = tab("06_subperiods.csv", index_col=[0, 1])
s1 = sub.loc[("fixed|ho_rb|blackout", "2007–2013"), "net"] / hb["total"]
add("horb 2010-13 share", f"{100 * s1:.0f}% of the profit was made in 2010–13")
ex = tab("06_exit_reasons.csv", index_col=0)
add("crack stops", f"{int(ex.loc['stop_loss', 'fixed|crack321|blackout'])} of {int(lad.loc[('fixed|crack321', 'blackout'), 'n_trades'])} trades")
stw = tab("06_stress_windows.csv", index_col=[0, 1])
add("horb 2026 unfiltered", f"\\${k(stw.loc[('hormuz', 'fixed|ho_rb'), 'unfiltered'])}")
for w, lab in [("covid", "2020"), ("ukraine", "2022"), ("hormuz", "2026 to date")]:
    cells = []
    for key in ["fixed|crack321", "fixed|brent_wti", "fixed|ho_rb"]:
        u, f_ = stw.loc[(w, key), "unfiltered"], stw.loc[(w, key), "filtered"]
        fm = lambda x: "\\$0" if abs(x) < 0.5 else (("+" if x > 0 else "−") + "\\$" + k(x))
        cells.append(f"{fm(u)} → {fm(f_)}")
    add(f"stress {w}", f"| {lab} | " + " | ".join(cells) + " |")
rob = tab("06_robustness.csv", index_col=[0, 1])["sharpe"]
for run, lab in [("window 504", "504-day window"), ("window 1008", "1,008-day window"), ("label 10d", "10-day label"),
                 ("label 40d", "40-day label"), ("drop Apr 2020 week, logit_filter", "Drop April 2020 week (gated)"),
                 ("drop Apr 2020 week, blackout", "Drop April 2020 week (no logit)")]:
    add(f"robust {run}", f"| {lab} | " + " | ".join(neg(rob[(run, f'fixed|{s}')]) for s in ["crack321", "brent_wti", "ho_rb"]) + " |")
cs = tab("06_cost_sweep.csv", index_col=[0, 1])["sharpe"]
add("cost horb", f"from {cs[('fixed|ho_rb', 0.0)]:.2f} (zero cost) to {cs[('fixed|ho_rb', 0.2)]:.2f}")
ib = tab("06_integer_book.csv", index_col=0)
add("book gated", f"| Gated | +\\${k(ib.loc['gated', 'total'])} | {float(ib.loc['gated', 'sharpe']):.2f} | "
                  f"{pct(float(ib.loc['gated', 'max_drawdown_pct']), 1)} | {int(ib.loc['gated', 'n_trades'])} |")
add("book nologit", f"| Without the logit | +\\${k(ib.loc['no logit', 'total'])} | {float(ib.loc['no logit', 'sharpe']):.2f} | "
                    f"{pct(float(ib.loc['no logit', 'max_drawdown_pct']), 1)} | {int(ib.loc['no logit', 'n_trades'])} |")
add("blocked by size", f"blocked an otherwise valid entry on {int(ib.loc['no logit', 'blocked_by_size'])} signal-days without the logit, "
                       f"and {int(ib.loc['gated', 'blocked_by_size'])} with it")

# ---------------------------------------------------------------- readout
ro = json.loads((T / "readout_summary.json").read_text())
s7 = ro["state"]
for key, lab in [("fixed|crack321", "3-2-1 crack"), ("fixed|brent_wti", "Brent–WTI"), ("fixed|ho_rb", "HO − RB")]:
    x = s7[key]
    add(f"readout {key}", f"| {lab} | {x['z']:.1f} | {x['p_revert']:.2f} | {x['sigma_next']:.1f} | "
                          f"{int(round(100 * x['sigma_pct_history']))}th |")
add("crack vol limit", f"(volatility limit {s7['fixed|crack321']['integer_vol_limit']:.2f})")
add("diesel crack", f"The diesel crack is \\${ro['diesel_crack']['level']:.0f}/bbl")
add("last closure", "the last logged closure was 27 September" if ro["last_hormuz_closure"] == "2026-09-27" else ro["last_hormuz_closure"])

bad = [(lab, s) for lab, s in checks if s not in R]
print(f"{len(checks) - len(bad)}/{len(checks)} report numbers match the output files")
for lab, s in bad:
    print(f"  MISMATCH {lab}: expected to find {s!r}")
sys.exit(1 if bad else 0)
