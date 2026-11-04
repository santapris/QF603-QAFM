"""Check that the numbers quoted in report/oil_deck.md match the current output files.

Usage:  .venv/bin/python scripts/check_deck.py
Each check formats a value from output/ or data/ the way the deck writes it and asserts that the string
appears in the deck. Exit code 1 lists any mismatch. Same approach as check_report.py.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from oilrv.config import ROOT

O, D = ROOT / "output", ROOT / "data"
DECK = (ROOT / "report" / "oil_deck.md").read_text(encoding="utf-8")
TC = O / "exploratory" / "trend_carry" / "tables"
HS = O / "exploratory" / "horizon_support" / "tables"
AR = O / "exploratory" / "ar_acf_pacf" / "tables"
checks: list[tuple[str, str]] = []


def add(label: str, s: str) -> None:
    checks.append((label, s))


def m(x: float, nd: int = 2) -> str:
    """Typographic minus, as the deck writes negatives."""
    return f"{x:.{nd}f}".replace("-", "−")


def sg(x: float, nd: int = 2) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):.{nd}f}"


def k(x: float) -> str:
    return f"{abs(x) / 1e3:.0f}k"


# --- back-adjusted level range, spread ranges
f = pd.read_csv(D / "clean" / "futures_spreads.csv", index_col=0, parse_dates=True)
add("crack adj min", f"−${abs(f.crack321.min()):.0f}")
add("crack adj max", f"+${f.crack321.max():.0f}")
add("crack market range", f"${f.crack321_unadj.min():.0f} to ${f.crack321_unadj.max():.0f}")

# --- unit roots (back-adjusted, full sample, regression c)
u = pd.read_csv(AR / "unit_roots.csv")
u = u[(u.period == "full") & (u.regression == "c") & (u.form == "back_adjusted")]
for s in ["crack321", "brent_wti", "ho_rb"]:
    r = u[(u.series == s) & (u.d == 0)].iloc[0]
    add(f"{s} level ADF", f"{r.adf_p:.3f}")
    add(f"{s} level PP", f"{r.pp_p:.3f}")
    r1, r2 = u[(u.series == s) & (u.d == 1)].iloc[0], u[(u.series == s) & (u.d == 2)].iloc[0]
    assert -0.55 < r2.acf1 < -0.44, r2.acf1          # deck: "about −0.5"
    assert r2.sd / r1.sd > 1.3
add("ho_rb d1 KPSS", f"{u[(u.series == 'ho_rb') & (u.d == 1)].kpss_p.iloc[0]:.3f}")
um = pd.read_csv(AR / "unit_roots.csv")
um = um[(um.period == "full") & (um.regression == "c") & (um.form == "market") & (um.d == 0)].set_index("series")
add("BW market ADF", f"{um.loc['brent_wti', 'adf_p']:.3f}")
add("HO-RB market ADF", f"{um.loc['ho_rb', 'adf_p']:.3f}")

# --- AR order, daily
a = pd.read_csv(AR / "ar_on_changes.csv")
a = a[a["sample"] == "full"].set_index("series")
for s, lab in [("crack321", "3-2-1 crack"), ("brent_wti", "Brent − WTI"), ("ho_rb", "HO − RB")]:
    row = [ln for ln in DECK.splitlines() if ln.startswith(f"| {lab} | AR(")]
    assert row, lab
    add(f"{s} BIC/AIC row", f"| {lab} | AR({a.loc[s, 'p_bic']}) | AR({a.loc[s, 'p_aic']}) | {100 * a.loc[s, 'r2_aic']:.1f}% |")
add("BIC AR(0) for 6 of 7", f"{int((a.p_bic == 0).sum())} of {len(a)}")

# --- 0 of 280 lag tests survive
lt = pd.read_csv(AR / "change_lag_tests.csv")
add("lag tests", f"**{int(lt.bh_reject.sum())} of {len(lt)}**")

# --- rolling cointegration
e = pd.read_csv(D / "derived" / "rolling_eg.csv")
sh = e.groupby("pair").apply(lambda d: (d.eg_p < .05).mean())
add("EG share BW", f"Brent vs WTI passes in {100 * sh['brent_wti']:.0f}%")
add("EG share HO-RB", f"HO vs RB in {100 * sh['ho_rb']:.0f}%")
add("EG range", f"{100 * sh.min():.0f}–{100 * sh.max():.0f}%")
j = pd.read_csv(D / "derived" / "rolling_johansen.csv")
add("Johansen rank 0", f"{100 * (j['rank'] == 0).mean():.0f}% of windows")
med = e.groupby("pair").half_life_fixed_1to1.median()
add("rolling half-lives", f"{med.min():.0f} to {med.max():.0f} days")

# --- ARCH, GARCH
mom = pd.read_csv(O / "tables" / "02_moments.csv", index_col=0)
assert (mom["arch_lm_p"] < 0.001).all()
vf = pd.read_csv(O / "tables" / "04_vol_forecast_test.csv", index_col=0)
assert (vf.loc[["fixed|crack321", "fixed|brent_wti", "fixed|ho_rb"], "mean_diff"] < 0).all()
vecm = pd.read_csv(O / "tables" / "04_vecm_free_full_sample.csv", index_col=[0, 1])
add("VECM rb t", f"t = {m(vecm.loc[('alpha_t', 'rb'), 'ect1'], 1)}")

# --- Ljung-Box simulation
lb = pd.read_csv(O / "tables" / "04_ljung_box_size_power.csv")
z = lb[lb.phi2 == 0]
add("LB plain false alarm", f"{100 * min(z.lb_p_10.min(), z.lb_p_20.min()):.0f}–{100 * max(z.lb_p_10.max(), z.lb_p_20.max()):.0f}%")
add("LB robust false alarm", f"{100 * min(z.robust_p_10.min(), z.robust_p_20.min()):.0f}–{100 * max(z.robust_p_10.max(), z.robust_p_20.max()):.0f}%")

# --- mean reversion gates and stress
gv = pd.read_csv(O / "tables" / "06_gate_verdicts.csv", index_col=0)
for key, lab in [("fixed|crack321", "3-2-1 crack"), ("fixed|brent_wti", "Brent − WTI"), ("fixed|ho_rb", "HO − RB")]:
    r = gv.loc[key]
    add(f"{key} sharpe", f"| {lab} | {m(r.G1_value)} ✗ | {int(r.G2_value)} ✗ |")
    assert not r.PASS
st = {kk: json.loads(gv.loc[kk, "G5_value"]) for kk in gv.index if kk.startswith("fixed")}
add("crack 2026", f"−${k(st['fixed|crack321']['hormuz']['unfiltered'])}")
add("BW 2022", f"−${k(st['fixed|brent_wti']['ukraine']['unfiltered'])}")
add("HO-RB 2026", f"−${k(st['fixed|ho_rb']['hormuz']['unfiltered'])}")
ms = json.loads((O / "tables" / "model_summary.json").read_text())["gates"]
add("|z|>=2 shares", f"{100 * ms['fixed|crack321']['share_days_abs_z_ge_2']:.0f}%, {100 * ms['fixed|brent_wti']['share_days_abs_z_ge_2']:.0f}% and {100 * ms['fixed|ho_rb']['share_days_abs_z_ge_2']:.0f}%")
ld = pd.read_csv(O / "tables" / "06_ladder_summary.csv")
bw = ld[(ld.key == "fixed|brent_wti") & (ld.step == "logit_filter")].total.iloc[0]
add("BW total loss", f"${abs(bw) / 1e3:.0f}k")

# --- pairs
pg = pd.read_csv(O / "exploratory" / "pairs" / "tables" / "gate_verdicts.csv", index_col=0)
add("pairs sharpe", f"**{m(pg.loc['book', 'G1_value'])}** on {int(pg.loc['book', 'G2_value'])} round trips")
cmpd = pd.read_csv(O / "exploratory" / "pairs" / "tables" / "comparisons.csv", index_col=0)
add("pairs no gate", f"{cmpd.loc['no_gate', 'sharpe']:.2f} (t = {cmpd.loc['no_gate', 'sharpe_t']:.2f})")
cs = pd.read_csv(O / "exploratory" / "pairs" / "tables" / "carry_split.csv", index_col=0)
add("pairs convergence", f"${abs(cs.loc['all', 'convergence_usd']) / 1e3:.0f}k")
add("pairs carry", f"${cs.loc['all', 'carry_usd'] / 1e3:.0f}k")

# --- trend and carry
tg = pd.read_csv(TC / "gate_verdicts.csv", index_col=0)
for b, lab in [("trend", "Trend"), ("carry", "Carry"), ("combined", "**50/50 book**")]:
    r = tg.loc[b]
    sh_ = f"**{r.G1_value:.2f} ✗**" if b == "combined" else f"{r.G1_value:.2f} ✗"
    add(f"{b} gates", f"| {lab} | {sh_} | {int(r.G2_value):,} ✓ |")
sb = pd.read_csv(TC / "summary_books.csv", index_col=0)
add("combined t", f"t-statistic {sb.loc['combined', 'sharpe_t']:.2f}")
add("years", f"{sb.loc['combined', 'years']:.1f} years")
cb = pd.read_csv(TC / "cost_breakdown.csv", index_col=0)
add("cost trend", f"{100 * cb.loc['trend', 'cost_share_of_gross']:.0f}%")
add("cost carry", f"{100 * cb.loc['carry', 'cost_share_of_gross']:.0f}%")
cor = pd.read_csv(TC / "correlation_with_mean_reversion.csv", index_col=0)
add("corr", f"({m(cor.loc['combined', 'mean_reversion'])})")
ann = pd.read_csv(TC / "annual.csv", header=[0, 1], index_col=0).drop(index="year", errors="ignore")
ann.index = ann.index.astype(int)
c = ann[("net", "combined")].astype(float)
add("2023-25", f"−${abs(c.loc[2023:2025].sum()) / 1e3:.0f}k")
add("2026 share", f"{100 * c.loc[2026] / c.sum():.0f}% of the profit")
grid = json.loads(json.dumps(json.loads((TC / "summary.json").read_text())["verdicts"]["combined"]["G4_value"]))
add("trend 504", f"{grid['trend lookback=504']:.2f}")

# --- monthly and support tests
ar = pd.read_csv(HS / "monthly_ar.csv")
me = ar[(ar.window == "2010-01 to 2026-09") & (ar.how == "month-end")].set_index("series")
mm = ar[(ar.window == "2010-01 to 2026-09") & (ar.how == "month-mean")].set_index("series")
fu = ar[(ar.window == "2007-01 to 2026-10") & (ar.how == "month-end")].set_index("series")
add("band", f"±{me.band.iloc[0]:.3f}")
add("n", f"{int(me.n.iloc[0])} changes")
for s, lab in [("crack321", "3-2-1 crack"), ("brent_wti", "Brent − WTI"), ("ho_rb", "HO − RB")]:
    add(f"{s} month-end row", f"| {lab} | {sg(me.loc[s, 'acf1'])} | {me.loc[s, 'ar1_p']:.3f} | AR({me.loc[s, 'p_bic']}) | AR({me.loc[s, 'p_aic']}) |")
add("month-mean lag1", f"{sg(mm.loc['crack321', 'acf1'])}, {sg(mm.loc['brent_wti', 'acf1'])} and {sg(mm.loc['ho_rb', 'acf1'])}")
sim = pd.read_csv(HS / "averaging_simulation.csv")["mean"].iloc[0]
add("simulation", f"simulated: {sim:.2f}")
add("A1 lag1 text", f"crack {sg(me.loc['crack321', 'acf1'])[0]}{abs(me.loc['crack321', 'acf1']):.2f}".replace("+", "+"))
add("A3 half-lives", f"{fu.loc['brent_wti', 'level_half_life_months']:.0f} months for Brent − WTI and {fu.loc['ho_rb', 'level_half_life_months']:.0f} for HO − RB")
for s, lab in [("crack321", "3-2-1 crack"), ("brent_wti", "Brent − WTI"), ("ho_rb", "HO − RB")]:
    hl = me.loc[s, "level_half_life_months"]
    hl_s = "none" if not np.isfinite(hl) or me.loc[s, "level_b"] >= 1 else f"{hl:.0f}"
    add(f"A3 {s}", f"| {lab} | {me.loc[s, 'level_adf_p']:.3f} | {me.loc[s, 'level_kpss_p']:.3f} | {hl_s} |")

vr = pd.read_csv(HS / "variance_ratio.csv").set_index("series")
for s, lab in [("cl", "WTI"), ("brent", "Brent"), ("rb", "RBOB"), ("ho", "ULSD"), ("crack321", "3-2-1 crack"),
               ("brent_wti", "Brent − WTI"), ("ho_rb", "HO − RB")]:
    add(f"VR {s}", f"| {lab} | {vr.loc[s, 'vr_12m']:.2f} | {vr.loc[s, 'p']:.2f} | {m(vr.loc[s, 'trend_gross_sharpe_monthly'])} |")

cf = pd.read_csv(HS / "carry_forward_returns.csv").set_index(["instrument", "regressor"])
for i, lab in [("cl", "WTI"), ("brent", "Brent")]:
    s_, z_ = cf.loc[(i, "sign")], cf.loc[(i, "size")]
    add(f"carry {i}", f"| {lab} | {sg(s_.coef)} | {s_.t:.2f} | {sg(z_.coef)} | {m(z_.t) if z_.t < 0 else f'{z_.t:.2f}'} |")
add("rb sign t", f"RBOB t = {m(cf.loc[('rb', 'sign'), 't'], 1)}")
add("ho sign t", f"ULSD t = {cf.loc[('ho', 'sign'), 't']:.1f}")

pdyn = pd.read_csv(HS / "passive_dynamic.csv").set_index("instrument")
for i, lab in [("cl", "WTI"), ("brent", "Brent")]:
    add(f"dyn {i}", f"| {lab} | {sg(pdyn.loc[i, 'mean_position_units'])} | {100 * pdyn.loc[i, 'dynamic_share_of_gross']:.0f}% | {pdyn.loc[i, 'dynamic_t']:.1f} |")
add("dyn ho", f"{100 * pdyn.loc['ho', 'dynamic_share_of_gross']:.0f}% (t = {pdyn.loc['ho', 'dynamic_t']:.2f})")

ob = pd.read_csv(HS / "oil_beta.csv").set_index("book")
for b, lab in [("trend", "Trend"), ("carry", "Carry"), ("combined", "50/50 book")]:
    add(f"beta {b}", f"| {lab} | {ob.loc[b, 'raw_sharpe']:.2f} | {ob.loc[b, 'alpha_t']:.2f} | {ob.loc[b, 'sharpe_from_alpha']:.2f} |")

# =============================================================================== ratio work (added 2026-11-04)
RS = O / "exploratory" / "ratio_stationarity" / "tables"
RM = O / "exploratory" / "ratio_mr" / "tables"
pct = lambda x: f"{int(float(x) * 100 + 0.5 + 1e-9)}%"      # round half up, as the deck writes percentages
NAMES = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}

sd_ = pd.read_csv(RS / "scale_dependence.csv").set_index("series")
dp_ = pd.read_csv(RS / "diff_on_price.csv").set_index("series")
dol, lr_ = sd_.corr_rolling_sd_dollar_change_vs_price, sd_.corr_rolling_sd_log_ratio_return_vs_price.abs()
add("differences: dollar vol corr", f"(correlation +{dol.min():.2f} to +{dol.max():.2f})")
add("differences: log vol corr", f"(−{lr_.min():.2f} to −{lr_.max():.2f})")
add("differences: intercepts", f"intercepts −${abs(dp_.loc['brent_wti', 'a_in_diff_eq_a_plus_cP']):.2f} and −${abs(dp_.loc['ho_rb', 'a_in_diff_eq_a_plus_cP']):.2f}")
assert abs(dp_.loc["brent_wti", "t_a_hac"]) < 2 and abs(dp_.loc["ho_rb", "t_a_hac"]) < 2          # "not different from zero"
assert 7.5 <= dp_.loc["crack321", "a_in_diff_eq_a_plus_cP"] < 8.5 and round(100 * dp_.loc["crack321", "c_in_diff_eq_a_plus_cP"]) == 18
add("differences: crack fixed", "about **$8 fixed**")
add("differences: crack proportional", "about 18% of the crude price")
add("roll gap ADF", f"p = {sd_.loc['crack321', 'adf_p_accumulated_roll_gap']:.2f} for the crack")

ur_ = pd.read_csv(RS / "unit_roots.csv")
ur_ = ur_[(ur_.freq == "daily") & (ur_.period == "full") & (ur_.regression == "c") & (ur_.d == 0)].set_index(["series", "form"])
ks_ = pd.read_csv(RS / "kpss_size.csv").set_index(["series", "form"])
za_ = pd.read_csv(RS / "zivot_andrews.csv").set_index(["series", "form"])
back = {s: u[(u.series == s) & (u.d == 0)].iloc[0] for s in NAMES}


def pz(x: float) -> str:
    return f"{x:.4f}" if x < 0.001 else (f"{x:.3f}" if x < 0.01 else f"{x:.2f}")


for s, lab in NAMES.items():
    seas = "_deseasonalised" if s != "brent_wti" else ""
    hd, hl_ = ks_.loc[(s, "difference" + seas), "half_life_days"], ks_.loc[(s, "log_ratio" + seas), "half_life_days"]
    d_, l_ = ur_.loc[(s, "difference")], ur_.loc[(s, "log_ratio")]
    add(f"test 1b row {s}", f"| {lab} | {back[s].adf_p:.3f} | {d_.adf_p:.3f} / {d_.pp_p:.3f} | **{l_.adf_p:.3f} / {l_.pp_p:.3f}** | "
                           f"{hd:.0f} → {hl_:.0f} | {pz(za_.loc[(s, 'difference'), 'za_p'])} → {pz(za_.loc[(s, 'log_ratio'), 'za_p'])} |")
    assert l_.kpss_p <= 0.0105 and d_.kpss_p <= 0.016                                         # "KPSS rejects everywhere"
add("test 1b: strict rule", "KPSS rejects everywhere")
add("kpss false alarms log", f"KPSS rejects {pct(ks_.loc[('crack321', 'log_ratio_deseasonalised'), 'kpss_rejects_if_truly_stationary'])}, "
                             f"{pct(ks_.loc[('brent_wti', 'log_ratio'), 'kpss_rejects_if_truly_stationary'])} and "
                             f"{pct(ks_.loc[('ho_rb', 'log_ratio_deseasonalised'), 'kpss_rejects_if_truly_stationary'])} of the log-ratios")
add("kpss false alarms dollar", f"and {pct(ks_.loc[('crack321', 'difference_deseasonalised'), 'kpss_rejects_if_truly_stationary'])}, "
                                f"{pct(ks_.loc[('brent_wti', 'difference'), 'kpss_rejects_if_truly_stationary'])} and "
                                f"{pct(ks_.loc[('ho_rb', 'difference_deseasonalised'), 'kpss_rejects_if_truly_stationary'])} of the dollar gaps")
lh = ks_[ks_.index.get_level_values(1).str.startswith("log")].half_life_days
assert round(lh.min()) == 35 and round(lh.max()) == 50
add("half-life range", "half-life 35–50 days")
co_ = pd.read_csv(RS / "cointegration_summary.csv").set_index(["pair", "form"])
add("log slopes", f"The log slope is {co_.loc[('brent_wti', 'logs'), 'beta_full']:.3f} for Brent on WTI and {co_.loc[('ho_rb', 'logs'), 'beta_full']:.3f} for HO on RB")
add("dollar slopes", f"(slopes {co_.loc[('brent_wti', 'levels'), 'beta_full']:.2f} and {co_.loc[('ho_rb', 'levels'), 'beta_full']:.2f})")
add("log EG shares", f"a rope is found in {pct(co_.loc[('brent_wti', 'logs'), 'share_eg_p_lt_05'])} of Brent/WTI windows ({pct(co_.loc[('brent_wti', 'levels'), 'share_eg_p_lt_05'])} in dollars) "
                     f"and {pct(co_.loc[('ho_rb', 'logs'), 'share_eg_p_lt_05'])} of HO/RB windows ({pct(co_.loc[('ho_rb', 'levels'), 'share_eg_p_lt_05'])} in dollars)")
ek_ = pd.read_csv(RS / "eras.csv")
for era in ["2007–10", "2011–13", "2014–19", "2020–26"]:
    k_ = ek_[ek_.era == era.replace("–", "-")].set_index("series").implied_k
    add(f"implied k {era}", f"| {era} | {k_['crack321']:.2f} | {k_['brent_wti']:.2f} | {k_['ho_rb']:.2f} |")
ac_ = pd.read_csv(RS / "acf_log_ratio_changes.csv")
assert int(ac_[ac_.form == "held_log_ratio_return"].bh_reject.sum()) == 0
add("log-ratio acf", "no lag survives the correction")
ch_ = pd.read_csv(RS / "changes_summary.csv")
ch_ = ch_[ch_.form == "held_log_ratio_return"].set_index("series")
add("log-ratio BIC", f"BIC picks AR({ch_.loc['crack321', 'bic_order']}), AR({ch_.loc['brent_wti', 'bic_order']}) and AR({ch_.loc['ho_rb', 'bic_order']}) "
                     f"(R² of the AIC model {100 * ch_.loc['crack321', 'r2_aic']:.1f}%, {100 * ch_.loc['brent_wti', 'r2_aic']:.1f}%, {100 * ch_.loc['ho_rb', 'r2_aic']:.1f}%)")

rj_ = pd.read_csv(RS / "roll_jumps.csv").set_index("series")
add("roll-day moves", f"(crack {rj_.loc['crack321', 'median_abs_move_roll_day']:.3f} against {rj_.loc['crack321', 'median_abs_move_other_day']:.3f}, "
                      f"Brent/WTI {rj_.loc['brent_wti', 'median_abs_move_roll_day']:.3f} against {rj_.loc['brent_wti', 'median_abs_move_other_day']:.3f}, "
                      f"HO/RB {rj_.loc['ho_rb', 'median_abs_move_roll_day']:.3f} against {rj_.loc['ho_rb', 'median_abs_move_other_day']:.3f})")
add("roll jump sums", f"add up to {m(rj_.loc['crack321', 'sum_roll_jumps'])}, {m(rj_.loc['brent_wti', 'sum_roll_jumps'])} and {sg(rj_.loc['ho_rb', 'sum_roll_jumps'])}, "
                      f"against ranges of {rj_.loc['crack321', 'range_of_market_log_ratio']:.2f}, {rj_.loc['brent_wti', 'range_of_market_log_ratio']:.2f} and {rj_.loc['ho_rb', 'range_of_market_log_ratio']:.2f}")
add("held sums", f"({sg(rj_.loc['crack321', 'sum_held_returns'])}, {sg(rj_.loc['brent_wti', 'sum_held_returns'])}, {sg(rj_.loc['ho_rb', 'sum_held_returns'])})")
tr_ = pd.read_csv(RS / "tradeable_reversion.csv")
tr_ = tr_[tr_.period == "full"].set_index(["series", "horizon"])
for s, lab in [("ho_rb", "HO / RB"), ("crack321", "3-2-1 crack"), ("brent_wti", "Brent / WTI")]:
    a21, a63 = tr_.loc[(s, 21)], tr_.loc[(s, 63)]
    bold = "**" if s == "ho_rb" else ""
    add(f"holder row {s}", f"| {lab} | {bold}{m(a21.held_slope, 3)} ({m(a21.held_t)}){bold} | {bold}{m(a63.held_slope, 3)} ({m(a63.held_t)}){bold} | "
                           f"{pct(a21.roll_share_of_market_reversion)} / {pct(a63.roll_share_of_market_reversion)} |")

vr_ = pd.read_csv(RS / "variance_ratios.csv")
v252 = vr_[(vr_.basis == "held_log_ratio") & (vr_.freq == "daily") & (vr_.q == 252)].set_index("series").vr
add("vr one year", f"**{pct(v252['crack321'])}, {pct(v252['brent_wti'])} and {pct(v252['ho_rb'])}**")
vo = vr_[(vr_.basis == "held_log") & (vr_.freq == "daily") & (vr_.q == 252)].set_index("series").vr
add("vr outrights", f"Brent {vo['brent']:.2f} and ULSD {vo['ho']:.2f} at one year")
add("vr outrights A4", f"Brent {vo['brent']:.2f}, ULSD {vo['ho']:.2f}, WTI {vo['cl']:.2f} and RBOB {vo['rb']:.2f}")
pr_ = vr_[vr_.basis == "held_log_ratio"].p_robust
add("vr p range", f"(robust p from {pr_.min():.2f} to {pr_.max():.2f})")
sa_ = pd.read_csv(RS / "trend_sign_agreement.csv")
add("trend sign", f"on only {100 * sa_.share_disagree.min():.0f}–{100 * sa_.share_disagree.max():.0f}% of days")
me_ = pd.read_csv(RS / "unit_roots.csv")
me_ = me_[(me_.freq == "month_end") & (me_.period == "full") & (me_.regression == "c") & (me_.d == 0)].set_index(["series", "form"])
add("A3 month-end log", f"ADF p is {me_.loc[('crack321', 'log_ratio'), 'adf_p']:.3f} for the crack, {me_.loc[('brent_wti', 'log_ratio'), 'adf_p']:.3f} for Brent/WTI "
                        f"and {me_.loc[('ho_rb', 'log_ratio'), 'adf_p']:.3f} for HO/RB ({me_.loc[('ho_rb', 'log_ratio_deseasonalised'), 'adf_p']:.3f} once")
assert me_.loc[("brent_wti", "log_ratio"), "kpss_p"] >= 0.10

# --- ratio mean reversion (pre-registered 2026-11-04)
rmj = json.loads((RM / "summary.json").read_text())
V, SP, CO = rmj["verdicts"], rmj["spreads"], rmj["comparisons"]
ok = lambda b: "✓" if b else "✗"


def g5(v: dict, y: str) -> str:
    return f"✗ ({m(v['G5_value'][y] / 1e3, 0).replace('−', '−$')}k in {'2026' if y == 'hormuz' else '2022'})"


for s, lab in NAMES.items():
    v, sp = V[s], SP[s]
    g1 = f"{m(v['G1_value'])} {ok(v['G1_sharpe'])}" + (f" (t = {sp['sharpe_t']:.2f})" if s == "ho_rb" else "")
    g5s = ok(v["G5_stress"]) if v["G5_stress"] else g5(v, "hormuz")
    add(f"ratio MR row {s}", f"| {lab} | {g1} | {v['G2_value']} {ok(v['G2_round_trips'])} | {v['G3_value']} {ok(v['G3_subperiods'])} | {ok(v['G4_grid'])} | {g5s} |")
vb = V["book"]
add("ratio MR book", f"| **Three-spread book** | **{vb['G1_value']:.3f} {ok(vb['G1_sharpe'])}** | {vb['G2_value']} {ok(vb['G2_round_trips'])} | {vb['G3_value']} {ok(vb['G3_subperiods'])} | "
                     f"{ok(vb['G4_grid'])} | {g5(vb, 'hormuz')} |")
add("ratio MR vs dollar", f"give a book Sharpe of {m(CO['dollar_market']['sharpe'])}, against {sg(CO['base']['sharpe'])} for the log-ratio")
cm_ = pd.read_csv(RM / "comparisons.csv", index_col=0)
add("ratio MR HO/RB dollar", f"({cm_.loc['dollar_market', 'sharpe|ho_rb']:.2f} → {cm_.loc['base', 'sharpe|ho_rb']:.2f})")
rb_ = pd.read_csv(RM / "robustness.csv").set_index(["setting", "value"])
add("ratio MR zero cost", f"at zero cost the book Sharpe is {rb_.loc[('cost_usd_per_bbl', 0.0), 'sharpe']:.2f}")
add("ratio MR windows", f"({m(rb_.loc[('window', 504.0), 'sharpe'])} at 504 days, {rb_.loc[('window', 1008.0), 'sharpe']:.2f} at 1,008)")
tz_ = pd.read_csv(RM / "trades.csv", parse_dates=["fill_date"])
t26 = tz_[(tz_.spread == "brent_wti") & (tz_.fill_date >= "2026-01-01")].sort_values("net_usd").iloc[0]
add("ratio MR 2026 Brent/WTI", f"entered in January with {t26.units:.0f} units was stopped for {m(t26.net_usd / 1e3, 0).replace('−', '−$')}k")
gs_ = pd.read_csv(RM / "gate_open_by_year.csv").drop(columns="year").mean()
add("ratio MR gate open", f"open on {100 * gs_.min():.0f}–{100 * gs_.max():.0f}% of days")
cr_ = pd.read_csv(RM / "correlation.csv", index_col=0)
add("ratio MR corr trend", f"**{m(cr_.loc['ratio_mr', 'trend_carry'])} correlated with the trend and carry book**")
add("ratio MR corr dollar", f"{sg(cr_.loc['ratio_mr', 'dollar_mean_reversion'])} with the registered dollar strategy")
add("recommendation ho_rb sleeve", f"stand-alone Sharpe {SP['ho_rb']['sharpe']:.2f}, t = {SP['ho_rb']['sharpe_t']:.2f}, daily P&L correlation {m(cr_.loc['ratio_mr', 'trend_carry'])} with the book")


from oilrv.config import load_config as _lc
cfg_slow = _lc()["exploratory_ratio_slow"]

# =============================================================================== slow ratio version (added 2026-11-04)
SL = O / "exploratory" / "ratio_slow" / "tables"
slj = json.loads((SL / "summary.json").read_text())
assert slj["reference_check_match"]                                    # the daily re-run reproduces the registered result
sV, sSP = slj["verdicts"], slj["spreads"]
vb_ = sV["book"]
assert (not vb_["G1_sharpe"]) and vb_["G2_round_trips"] and not vb_["G3_subperiods"] and not vb_["G4_grid"] and not vb_["G5_stress"]
add("slow: gates failed", "It fails G1, G3, G4 and G5.")
add("slow: sharpe vs daily", f"Book Sharpe {m(slj['book']['sharpe'])} against {sg(rmj['book']['sharpe'])} daily, on {slj['round_trips']} trades instead of {rmj['round_trips']}")
add("slow: HO/RB", f"falls from {rmj['spreads']['ho_rb']['sharpe']:.2f} to {m(sSP['ho_rb']['sharpe'])}")
add("slow: fragile", f"the same spread falls to {m(sSP['ho_rb']['sharpe'])} when decisions go monthly")
add("slow: summary row", f"a monthly-decision version fails ({m(slj['book']['sharpe'])})")

v63 = vr_[(vr_.basis == "held_log_ratio") & (vr_.freq == "daily") & (vr_.q == 63)].set_index("series").vr
v21 = vr_[(vr_.basis == "held_log_ratio") & (vr_.freq == "daily") & (vr_.q == 21)].set_index("series").vr
add("horizon: 63d and 1y", f"(variance ratio {v63['crack321']:.2f}, {v63['brent_wti']:.2f} and {v63['ho_rb']:.2f} at 63 days, "
                           f"{v252['crack321']:.2f}, {v252['brent_wti']:.2f} and {v252['ho_rb']:.2f} at one year)")
add("horizon: 21d", f"({v21['crack321']:.2f}, {v21['brent_wti']:.2f} and {v21['ho_rb']:.2f} at 21 days)")
dh = {s: rmj["spreads"][s]["avg_hold"] for s in NAMES}
add("horizon: daily holds", f"**{dh['crack321']:.0f}, {dh['brent_wti']:.0f} and {dh['ho_rb']:.0f} days**")

ex_ = pd.read_csv(SL / "exit_reasons.csv").set_index("exit_reason")
n_tot = int(ex_.to_numpy().sum())
assert n_tot == slj["round_trips"]
add("slow: exits", f"**{int(ex_.loc['time_stop'].sum())} of {n_tot} trades ended on the time stop**, and only {int(ex_.loc['take_profit'].sum())} reached take-profit")
st_ = pd.read_csv(SL / "trades.csv")
sc_ = st_[st_.exit_reason != "open_at_end"]
add("slow: time stop loss", f"The time stops lost ${abs(sc_[sc_.exit_reason == 'time_stop'].net_usd.sum()) / 1e3:.0f}k in total")
hl_ = sc_.groupby("spread").half_life.mean()
add("slow: fitted half-life", f"({hl_.min():.0f} to {hl_.max():.0f} days)")
hd_ = sc_.groupby("spread").hold_days.mean()
add("slow: holds", f"average holds were {hd_['crack321']:.0f}, {hd_['brent_wti']:.0f} and {hd_['ho_rb']:.0f} days")

s_hr = rmj["spreads"]["ho_rb"]["sharpe"]
add("slow: years for t", f"**{(2 / s_hr) ** 2:.0f} years for t = 2 and {(3 / s_hr) ** 2:.0f} years for t = 3**")
add("slow: best sharpe", f"(HO/RB daily, {s_hr:.2f})")
worst = st_.sort_values("net_usd").iloc[:2]
w_ho = worst[worst.spread == "ho_rb"].iloc[0]
w_bw = st_[(st_.spread == "brent_wti")].sort_values("net_usd").iloc[0]
assert str(w_ho.fill_date).startswith("2026") and str(w_bw.fill_date).startswith("2011")
add("slow: worst trades", f"HO/RB in 2026 (−${abs(w_ho.net_usd) / 1e3:.0f}k on one trade) and Brent/WTI in 2011 (−${abs(w_bw.net_usd) / 1e3:.0f}k)")
assert cfg_slow["variants_tried_on_sample"] == 6 and cfg_slow["book_min_t_to_call_evidence"] == 3.0
add("slow: sixth look", "the **sixth** strategy variant tried on one sample")
add("slow: t above 3", "t above 3.0")

missing = [(lab, s) for lab, s in checks if s not in DECK]
print(f"{len(checks) - len(missing)} of {len(checks)} quoted numbers match")
for lab, s in missing:
    print(f"  MISMATCH {lab}: {s!r}")
sys.exit(1 if missing else 0)
