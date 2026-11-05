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
add("differences: dollar vol corr", f"S: +{dol.min():.2f} to +{dol.max():.2f}")
add("differences: log vol corr", f"ℓ: −{lr_.min():.2f} to −{lr_.max():.2f}")
add("differences: intercepts", f"a = −${abs(dp_.loc['brent_wti', 'a_in_diff_eq_a_plus_cP']):.2f} and −${abs(dp_.loc['ho_rb', 'a_in_diff_eq_a_plus_cP']):.2f}")
assert abs(dp_.loc["brent_wti", "t_a_hac"]) < 2 and abs(dp_.loc["ho_rb", "t_a_hac"]) < 2          # "not different from zero"
assert 7.5 <= dp_.loc["crack321", "a_in_diff_eq_a_plus_cP"] < 8.5 and round(100 * dp_.loc["crack321", "c_in_diff_eq_a_plus_cP"]) == 18
add("differences: crack fixed", "a ≈ $8 fixed refining cost")
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

# --- monthly log-ratio: ACF/PACF, unit roots (run_ratio_monthly.py)
RMO = O / "exploratory" / "ratio_monthly" / "tables"
ms_ = pd.read_csv(RMO / "monthly_summary.csv").set_index(["series", "form"])
mu_ = pd.read_csv(RMO / "monthly_unit_roots.csv")
mu_ = mu_[mu_.window == "2010-26"].set_index(["series", "form", "what", "regression"])
ev_ = pd.read_csv(RMO / "month_end_vs_mean.csv").set_index(["series", "how"])
LABR = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}
for s, lab in LABR.items():
    lv, d0 = ms_.loc[(s, "log_ratio_level")], me.loc[s]
    dhl = "none" if d0.level_b >= 1 else f"{d0.level_half_life_months:.0f} months"
    add(f"step7 {s}", f"| {lab} | {d0.level_b:.3f}, {dhl} | **{lv.ar1_b:.2f}, {lv.half_life_months:.1f} months** | "
                      f"{lv.adf_p:.3f} / {lv.pp_p:.3f} / {lv.kpss_p:.3f} |")
    h = ms_.loc[(s, "held_return")]
    add(f"A1b {s}", f"| {lab} | {m(h.acf1)} | {h.ar1_p:.3f} | AR({int(h.p_bic)}) | AR({int(h.p_aic)}) | "
                    f"{int(h.bh_rejections)} of {int(h.bh_tests)} |")
lvl = [ms_.loc[(s, "log_ratio_level"), "pacf1"] for s in LABR]
add("step7 pacf", f"one spike at lag 1 ({lvl[0]:.2f}, {lvl[1]:.2f}, {lvl[2]:.2f})")
hr = [ms_.loc[(s, "held_return"), "acf1"] for s in LABR]
add("step7 held lag1", f"({m(hr[0])}, {m(hr[1])}, {m(hr[2])})")
add("step7 dollar lag1", f"(+{me.loc['crack321', 'acf1']:.2f}, +{me.loc['brent_wti', 'acf1']:.2f})")
hp = [ms_.loc[(s, "held_return"), "ar1_p"] for s in LABR]
add("step7 ar1 p", f"AR(1) p = {min(hp):.2f} to {max(hp):.2f}")
assert all(ms_.loc[(s, "held_return"), "bh_rejections"] == 0 for s in LABR)
add("step7 ho ds", f"calendar-month effects are removed (ADF {mu_.loc[('ho_rb', 'log_ratio_ds', 'level', 'c'), 'adf_p']:.3f})")
for s in ["crack321", "brent_wti"]:
    r = mu_.loc[(s, "log_ratio", "level", "ct")]
    assert r.stationary_all, s
    add(f"A3 trend {s}", f"({r.adf_p:.3f} / {r.pp_p:.3f} / {r.kpss_p:.3f})")
assert mu_.loc[("ho_rb", "log_ratio_ds", "level", "ct"), "stationary_all"]
assert not mu_.xs("level", level="what").loc[lambda t: t.index.get_level_values("form").str.startswith("dollar")].stationary_all.any()
assert mu_.xs("change", level="what").stationary_all.all()
hl_ = [ms_.loc[(s, "log_ratio_level"), "half_life_months"] for s in LABR]
add("A3 log half-lives", f"half-lives of {hl_[0]:.1f}, {hl_[1]:.1f} and {hl_[2]:.1f} months")
e_end = [ev_.loc[(s, "month_end"), "change_acf1"] for s in LABR]
e_mean = [ev_.loc[(s, "month_mean"), "change_acf1"] for s in LABR]
add("A2 log averaging", f"{m(e_end[0])}, {m(e_end[1])} and {m(e_end[2])}; of the monthly-mean change "
                        f"{sg(e_mean[0])}, {sg(e_mean[1])} and {sg(e_mean[2])} (HO/RB p = {ev_.loc[('ho_rb', 'month_mean'), 'change_ar1_p']:.3f})")
mk = ms_.loc[("ho_rb", "market_log_ratio_change")]
add("A1b market horb", f"({int(mk.bh_rejections)} of {int(mk.bh_tests)} lags survive, robust Ljung–Box p = {mk.lb_robust_p_12:.3f})")

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

# =============================================================================== moving anchor, k tie-out, 2013, Cushing, log VECM, gate power (added 2026-11-04)
GP = O / "exploratory" / "gate_power" / "tables" if False else O / "exploratory" / "gate_power"
kc_ = pd.read_csv(RS / "k_consistency.csv").set_index("series")
p1 = lambda x: f"{100 * x:.1f}%"
add("k tie-out", f"{p1(kc_.loc['crack321', 'premium_from_dollar_regression'])} against {p1(kc_.loc['crack321', 'premium_from_mean_log_ratio'])}, "
                 f"{p1(kc_.loc['brent_wti', 'premium_from_dollar_regression'])} against {p1(kc_.loc['brent_wti', 'premium_from_mean_log_ratio'])}, "
                 f"{p1(kc_.loc['ho_rb', 'premium_from_dollar_regression'])} against {p1(kc_.loc['ho_rb', 'premium_from_mean_log_ratio'])}")
assert (kc_.gap_dollar_vs_log.abs() < 0.01).all()

ho_ = pd.read_csv(RS / "heating_oil_2013.csv")
ho_ = ho_[ho_.break_date.str.startswith("2013-05")].set_index("series")
add("2013 HO/RB", f"(+{100 * ho_.loc['ho_rb', 'jump_log_ratio']:.1f}% in May 2013, {round(100 * ho_.loc['ho_rb', 'percentile_of_abs_jump_among_all_month_starts'])}th percentile")
add("2013 Brent/WTI", f"The {100 * abs(ho_.loc['brent_wti', 'jump_log_ratio']):.1f}% fall in Brent/WTI in 2013 ({round(100 * ho_.loc['brent_wti', 'percentile_of_abs_jump_among_all_month_starts'])}th percentile)")
cu_ = pd.read_csv(RS / "cushing_k.csv")
lv_, wk_ = cu_.iloc[0], cu_.iloc[2]
add("cushing levels", f"(t = {lv_.t_hac:.1f} in levels, R² {100 * lv_.r2:.1f}%; in weekly changes t = {wk_.t_hac:.1f}, R² {100 * wk_.r2:.1f}%)")

ua_ = pd.read_csv(RS / "moving_anchor_model.csv").set_index("series")
ae_ = pd.read_csv(RS / "anchor_error.csv").set_index("series")
for s, lab in [("crack321", "3-2-1 crack"), ("brent_wti", "Brent / WTI"), ("ho_rb", "HO / RB")]:
    add(f"anchor row {s}", f"| {lab} | {ua_.loc[s, 'half_life_fixed_mean_days']:.0f} days | {ua_.loc[s, 'half_life_moving_anchor_days']:.1f} days | "
                           f"{p1(ua_.loc[s, 'sigma_k_annual'])} | {p1(ua_.loc[s, 'sd_deviation_u'])} |")
add("anchor error", f"**{ae_.loc['brent_wti', 'rmse_anchor_error_over_sd_u']:.1f}, {ae_.loc['crack321', 'rmse_anchor_error_over_sd_u']:.1f} and {ae_.loc['ho_rb', 'rmse_anchor_error_over_sd_u']:.1f}**")
sh_ = ae_.share_of_those_where_filtered_deviation_lt_1sd
add("false extremes", f"**{pct(sh_['brent_wti'])}, {pct(sh_['crack321'])} and {pct(sh_['ho_rb'])}**")
hl_u = ua_.half_life_moving_anchor_days
add("anchor: reverts in", f"fast (2 to 18 days)")
assert round(hl_u.min()) == 2 and round(hl_u.max()) == 18 and 0.01 <= ua_.sd_deviation_u.min() < 0.0105 and 0.069 <= ua_.sd_deviation_u.max() < 0.07
add("anchor: summary row", "2 to 18 days")
add("anchor: drift range", "drifts 8 to 10% a year")
assert 0.083 <= ua_.sigma_k_annual.min() and ua_.sigma_k_annual.max() < 0.1055
add("anchor: error range", "0.8 to 5.5 deviation-sd")
add("anchor: ranking", "(−0.49, 0.22, 0.38)")

vk_ = pd.read_csv(RS / "vecm_logs_known.csv")
cl_ = vk_[(vk_.spread == "crack321") & (vk_.equation == "held log return of CL")].iloc[0]
bw_ = vk_[(vk_.spread == "brent_wti") & (vk_.equation == "held log return of CL")].iloc[0]
add("vecm log WTI", f"(WTI α = +{cl_.alpha:.3f} a day, t = {cl_.t:.1f}, in the crack; t = {bw_.t:.1f} in Brent/WTI)")
sp_ = vk_[vk_.equation.str.startswith("held log-ratio")].set_index("spread")
add("vecm log speed", f"−{abs(sp_.loc['crack321', 'alpha']):.3f}, −{abs(sp_.loc['brent_wti', 'alpha']):.3f} and −{abs(sp_.loc['ho_rb', 'alpha']):.3f} a day "
                      f"(t = {m(sp_.loc['crack321', 't'], 1)}, {m(sp_.loc['brent_wti', 't'], 1)} and {m(sp_.loc['ho_rb', 't'], 1)}), "
                      f"half-lives of {sp_.loc['crack321', 'implied_half_life_days']:.0f}, {sp_.loc['brent_wti', 'implied_half_life_days']:.0f} and {sp_.loc['ho_rb', 'implied_half_life_days']:.0f} days")
assert (ch_.arch_lm_p_10 < 0.001).all()
add("arch log-ratio", "also rejects on daily log-ratio returns (p < 0.001)")

# gate power and the model worlds
gw_ = pd.read_csv(GP / "model_worlds.csv")
g1_ = pd.read_csv(GP / "g1_power.csv").set_index("true_sharpe")
ds_ = pd.read_csv(GP / "deflated_sharpe.csv", index_col=0)
wA = gw_[gw_.world.isin(["A", "B"])].mean_sharpe
add("worlds A/B", f"a Sharpe of **{wA.min():.1f} to {wA.max():.1f}**")
wc = gw_[gw_.world == "C"].set_index("spread").mean_sharpe
add("worlds C", f"**{wc['crack321']:.2f}, {wc['brent_wti']:.2f} and {wc['ho_rb']:.2f}**")
rt = gw_[gw_.world == "C"].set_index("spread").mean_round_trips_per_year * 16.718
add("worlds trade counts", f"(26, 16 and 26 simulated against 29, 15 and 27 realised)")
assert [round(rt[s]) for s in ("crack321", "brent_wti", "ho_rb")] == [26, 16, 26]
assert [SP[s]["n_trades"] for s in ("crack321", "brent_wti", "ho_rb")] == [29, 15, 27]
cc = gw_[gw_.world == "C"].set_index("spread")
assert cc.loc["brent_wti", "realised_daily_decisions"] < cc.loc["brent_wti", "p05"] and cc.loc["ho_rb", "realised_daily_decisions"] < cc.loc["ho_rb", "p05"]
assert cc.loc["crack321", "p05"] < cc.loc["crack321", "realised_daily_decisions"] < cc.loc["crack321", "p95"]
dd = gw_[gw_.world == "D"].mean_sharpe
add("worlds D", f"earns {dd.min():.2f} to {dd.max():.2f}")
add("g1 power se", "standard error of about 0.25")
assert 0.24 <= g1_.loc[0.0, "se"] <= 0.26
add("g1 power 0.45", f"passes G1 only **{pct(g1_.loc[0.45, 'p_pass_G1'])}** of the time")
add("g1 power 0", f"passes **{pct(g1_.loc[0.0, 'p_pass_G1'])}** of the time")
add("g1 six nulls", f"at least one pass {pct(g1_.loc[0.0, 'p_at_least_one_of_six_passes_if_all_true_sharpes_equal_this'])} of the time")
add("dsr", f"**{ds_.loc[6, 'dsr']:.2f}** with six trials and {ds_.loc[12, 'dsr']:.2f} with twelve")

# =============================================================================== restructured deck: ARIMA, results matrix, integer book (added 2026-11-04)
RINT = O / "exploratory" / "ratio_integer" / "tables"
ar_ = pd.read_csv(RS / "arima_orders.csv")
fa_ = pd.read_csv(RS / "arima_factors.csv")
oo_ = pd.read_csv(RS / "arima_oos.csv")
wc_ = pd.read_csv(RS / "arima_wti_check.csv")
ib_ = pd.read_csv(RINT / "integer_book.csv", index_col=0)
ibs_ = pd.read_csv(RINT / "by_spread.csv")


def aob(s, f, c="BIC"):
    return ar_[(ar_.series == s) & ar_.form.str.startswith(f) & (ar_.criterion == c)].iloc[0]


def slow_hl(s):
    return aob(s, "log-ratio, d = 0").slowest_ar_root_half_life_days


# title
add("title answer", f"with a net Sharpe of {sb.loc['combined', 'sharpe']:.2f}.")

# step 5: the ARIMA slide
c0, b0, h0 = "3-2-1 crack", "Brent / WTI", "HO / RB"
add("arima row crack", f"| 3-2-1 crack | ARIMA{aob(c0, 'dollar').order} | ARIMA{aob(c0, 'log-ratio, d = 0').order} | {slow_hl(c0):.0f} days | ARIMA{aob(c0, 'held').order}, partly cancelling terms |")
add("arima row bw", f"| Brent / WTI | ARIMA{aob(b0, 'dollar').order}, coefficients 0.06 and −0.04 | ARIMA{aob(b0, 'log-ratio, d = 0').order} | {slow_hl(b0):.0f} days | ARIMA{aob(b0, 'held').order}, coefficients under 0.14 |")
add("arima row horb", f"| HO / RB | ARIMA{aob(h0, 'dollar').order} | ARIMA{aob(h0, 'log-ratio, d = 0').order} | {slow_hl(h0):.0f} days | ARIMA{aob(h0, 'held').order} |")
bw_d = json.loads(aob(b0, "dollar").params)
assert round(bw_d["ar.L1"], 2) == 0.06 and round(bw_d["ar.L2"], 2) == -0.04
assert max(abs(v) for k_, v in json.loads(aob(b0, "held").params).items() if k_.startswith("ar")) < 0.14
assert (ar_[ar_.form.str.startswith("log price")].arch_lm_p10 < 0.001).all() and (ar_.arch_lm_p10 < 0.001).all()
assert wc_.iloc[1].bic_order == "(0,1,0)"
for s in ("Brent", "RBOB", "ULSD"):
    assert aob(s, "log price").order == "(0,1,0)"
add("arima products", "ARIMA(0,1,0) with drift (BIC), and so is WTI once 9 March to 29 May 2020 is excluded")
slow = fa_[fa_.form.str.startswith("log-ratio, d = 0") & (fa_.polynomial == "AR") & (fa_.imag.abs() < 1e-6)]
mods = slow.groupby("series").modulus.min()
assert f"{mods.min():.3f}" == "1.007" and f"{mods.max():.2f}" == "1.02"
pull = 100 * (1 - 1 / mods)
assert 0.65 <= pull.min() < 0.75 and round(pull.max()) == 2
add("arima roots", "one root of 1.007 to 1.02 (half-life 35 to 98 days)")
add("arima pull", "closes only 0.7% to 2% of the gap a day")
assert round(slow_hl(h0)) == 35 and round(slow_hl(b0)) == 98
assert json.loads(aob(h0, "log-ratio, d = 1").params)["ma.L1"] < -0.99 + 1e-3 and f"{json.loads(aob(h0, 'log-ratio, d = 1').params)['ma.L1']:.2f}" == "-0.99"
add("arima theta", "moving-average term of −0.99")
fb = fa_[(fa_.series == b0) & fa_.form.str.startswith("log-ratio, d = 0") & (fa_.polynomial == "AR") & (fa_.imag > 1e-6)].iloc[0]
rho, ang = 1 / fb.modulus, abs(np.angle(1 / complex(fb.real, fb.imag)))
phi_s = 1 / mods[b0]
quad_a, quad_b = 2 * np.real(1 / complex(fb.real, fb.imag)), rho ** 2
add("arima factored", f"AR = (1 − {phi_s:.3f} L)(1 − {quad_a:.3f} L + {quad_b:.3f} L²)")
add("arima oscillation", f"it decays by {rho:.2f} a day, period about {round(2 * np.pi / ang):.0f} days")
r13 = oo_[oo_.model.isin(["ar1", "ar3"])]
add("arima oos", f"(RMSE ratio {r13.rmse_ratio_vs_random_walk.min():.2f} to {r13.rmse_ratio_vs_random_walk.max():.2f}, Diebold–Mariano |t| of at most {r13.dm_t.abs().max():.1f})")
mn = oo_[oo_.model == "mean"].pivot_table(index="series", columns="horizon_days", values="rmse_ratio_vs_random_walk")
assert (mn.drop(index="HO / RB") > 1).all().all() and (mn.loc["HO / RB", [1, 5, 21]] > 1).all() and mn.loc["HO / RB", 63] < 1

# step 6
add("step6 ho_rb t", f"(t = {m(tr_.loc[('ho_rb', 21)].held_t)} at 21 days, {m(tr_.loc[('ho_rb', 63)].held_t)} at 63 days)")
weak = pd.concat([tr_.loc[(s, h)] for s in ("crack321", "brent_wti") for h in (21, 63)], axis=1).T
add("step6 weak t", f"|t| of at most {weak.held_t.abs().max():.2f}")
rs_ = weak.roll_share_of_market_reversion
add("step6 roll share", f"{pct(rs_.min())} to {pct(rs_.max())} of their apparent reversion is the roll")

# results matrix
ib_d, ib_m = ib_.loc["daily decisions (registered ratio rules)"], ib_.loc["monthly decisions (slow version)"]
reg_int = float(pd.read_csv(O / "tables" / "06_integer_book.csv", index_col=0).loc["gated", "sharpe"])
gk = ["fixed|crack321", "fixed|brent_wti", "fixed|ho_rb"]
g5w = [min(json.loads(gv.loc[k_, "G5_value"])[w]["unfiltered"] for w in ("ukraine", "hormuz")) for k_ in gk]
add("matrix MR", f"| Mean reversion, dollar gap (crack / Brent − WTI / HO − RB) | {' / '.join(m(gv.loc[k_, 'G1_value']) for k_ in gk)} ✗ | "
                 f"{' / '.join(str(int(gv.loc[k_, 'G2_value'])) for k_ in gk)} ✗ | {' / '.join(str(int(gv.loc[k_, 'G3_value'])) for k_ in gk)} | "
                 f"{' / '.join('✓' if gv.loc[k_, 'G4_grid'] else '✗' for k_ in gk)} | {' / '.join('−$' + k(x) for x in g5w)} ✗ | {reg_int:.2f} |")
add("matrix ratio daily", f"| Ratio mean reversion, daily decisions (book of 3) | {V['book']['G1_value']:.3f} ✗ | {V['book']['G2_value']} ✓ | {V['book']['G3_value']} ✓ | ✓ | "
                          f"−${k(V['book']['G5_value']['hormuz'])} ✗ | {ib_d.sharpe:.2f} |")
add("matrix ratio monthly", f"| Ratio mean reversion, monthly decisions | {m(sV['book']['G1_value'])} ✗ | {sV['book']['G2_value']} ✓ | {sV['book']['G3_value']} ✗ | ✗ | "
                            f"−${k(sV['book']['G5_value']['hormuz'])} ✗ | {m(ib_m.sharpe)} |")
pj = json.loads((O / "exploratory" / "pairs" / "tables" / "summary.json").read_text())["verdicts"]["book"]
assert pj["G3_value"] == 1 and not pj["G4_grid"] and not pj["G5_stress"]
add("matrix pairs", f"| Pairs (book of 6) | {m(pg.loc['book', 'G1_value'])} ✗ | {int(pg.loc['book', 'G2_value'])} ✓ | {pj['G3_value']} ✗ | ✗ | −${k(min(pj['G5_value'].values()))} ✗ | not run |")
tcb = tg.loc["combined"]
assert tcb.G4_grid and tcb.G5_stress and min(json.loads(tcb.G5_value).values()) > 0
add("matrix trend carry", f"| **Trend and carry (50/50 book)** | **{tcb.G1_value:.2f} ✗** | {int(tcb.G2_value):,} ✓ | {int(tcb.G3_value)} ✓ | ✓ | profit in 2022 and 2026 ✓ | not run |")
add("matrix G1 miss", f"misses G1 by {0.5 - tcb.G1_value:.2f}")
bwr = float(pd.read_csv(RM / "daily_books.csv", index_col=0, parse_dates=True).loc["2026-01-01":, "brent_wti"].sum())
bwi = float(ibs_[(ibs_.run.str.startswith("daily")) & (ibs_.spread == "brent_wti")].y2026.iloc[0])
add("matrix integer effect", f"from {V['book']['G1_value']:.3f} to {ib_d.sharpe:.2f} and cuts the 2026 Brent/WTI loss from ${abs(bwr) / 1e3:.0f}k to ${abs(bwi) / 1e3:.0f}k")
add("matrix noise", "a standard error of about 0.25")

# why mean reversion fails
add("why exits", f"{int(ex_.loc['time_stop'].sum())} of {n_tot} trades ended on the time stop")
add("why sizing", f"cuts that loss from ${abs(bwr) / 1e3:.0f}k to ${abs(bwi) / 1e3:.0f}k")
add("why sharpe", f"The Sharpe is still only {ib_d.sharpe:.2f}")
add("why ratio sharpe", f"(Sharpe {m(slj['book']['sharpe'])})")

# what would change the answer
yrs = (pd.Timestamp("2026-10-26") - pd.Timestamp("1990-01-02")).days / 365.25
assert round(yrs) == 37
add("change years", "takes Brent/WTI from 16.7 to about 37 years")
add("change power", f"(about {(2 / s_hr) ** 2:.0f} years for t = 2 at the best Sharpe we saw)")

# caveats and summary
add("caveat sizing", f"has a Sharpe of {ib_d.sharpe:.2f} and the 2026 Brent/WTI loss is ${abs(bwi) / 1e3:.0f}k")
add("summary integer", f"The daily ratio strategy has a Sharpe of {ib_d.sharpe:.2f} on capped sizing ({V['book']['G1_value']:.3f} on the research stream)")
add("summary arima", f"log-ratios are d = 0 with a slow root (half-life {min(slow_hl(s) for s in (c0, b0, h0)):.0f} to {max(slow_hl(s) for s in (c0, b0, h0)):.0f} days)")

# appendix: ARIMA in full and the integer book
add("af1 wti", f"On the full sample the (2,1,2) beats the random walk by only {wc_.iloc[0].bic_random_walk - wc_.iloc[0].bic_best:.0f} BIC points")
add("af1 wti excl", f"Excluding 9 March to 29 May 2020, BIC picks {wc_.iloc[1].bic_order}")
add("af2 horb lb", f"(robust Ljung–Box p = {aob(h0, 'log-ratio, d = 0').lb_robust_p20:.3f} for d = 0)")
add("af2 held lb", f"passes (p = {aob(h0, 'held').lb_robust_p20:.2f})")
add("af3 complex", f"The complex pair {fb.real:.3f} ± {abs(fb.imag):.3f}i has modulus {fb.modulus:.3f}")
add("af3 ma", "The MA roots (2.79 and 16.5)")
ma_b = fa_[(fa_.series == b0) & fa_.form.str.startswith("log-ratio, d = 0") & (fa_.polynomial == "MA")].modulus.sort_values()
assert f"{ma_b.iloc[0]:.2f}" == "2.79" and f"{ma_b.iloc[1]:.1f}" == "16.5"
hb = oo_[(oo_.series == h0) & (oo_.horizon_days.isin([21, 63])) & oo_.model.isin(["ar1", "ar3"])]
add("af3 horb best", f"({hb.rmse_ratio_vs_random_walk.min():.2f} to {hb.rmse_ratio_vs_random_walk.max():.2f}), with t of at most {hb.dm_t.max():.1f}")
add("int trips", f"take the same {ib_d.n_round_trips:.0f} round trips")
add("int units", f"with units of {ib_d.mean_units:.1f} on average (cap 3)")
add("int tail", f"The worst trade falls from $190k to ${abs(ib_d.worst_trade_usd) / 1e3:.0f}k")
add("int dd", f"the maximum drawdown is {100 * ib_d.max_drawdown_pct:.1f}% against {100 * ib_.loc['registered dollar-gap strategy (reference)', 'max_drawdown_pct']:.1f}%")
add("int slow horb", f"HO/RB's 2026 loss is still ${abs(float(ibs_[(ibs_.run.str.startswith('monthly')) & (ibs_.spread == 'ho_rb')].y2026.iloc[0])) / 1e3:.0f}k")

# =============================================================================== month-end lens: frequency ladder, phase, power (added 2026-11-04)
lad_ = pd.read_csv(RS / "frequency_ladder.csv")
phs_ = pd.read_csv(RS / "frequency_phase.csv")
pwr_ = pd.read_csv(RS / "kpss_power_month_end.csv").set_index("series")
mma_ = pd.read_csv(RS / "arima_month_end.csv")
RAW_, DS_ = "raw", "calendar-month effects removed"
LL = lambda sp, form, fq: lad_[(lad_.series == sp) & (lad_.form == form) & (lad_.freq == fq)].iloc[0]
PHx = lambda sp, form, col: phs_[(phs_.series == sp) & (phs_.form == form)][col]
SH_ = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}

for sp, lab in SH_.items():
    r_, d_ = LL(sp, RAW_, "month-end"), LL(sp, DS_, "month-end")
    add(f"month-end row {sp}", f"| {lab} | {r_.adf_p:.3f} / {r_.pp_p:.3f} | {d_.adf_p:.3f} / {d_.pp_p:.3f} | {d_.half_life_days_approx:.0f} | {d_.za_p:.3f} |")
arma_ = {sp: float(aob(SH_[sp], "log-ratio, d = 0").slowest_ar_root_half_life_days) for sp in SH_}
add("month-end vs ARMA", f"The month-end half-lives ({LL('crack321', DS_, 'month-end').half_life_days_approx:.0f}, {LL('brent_wti', DS_, 'month-end').half_life_days_approx:.0f} and {LL('ho_rb', DS_, 'month-end').half_life_days_approx:.0f} days) "
                         f"match the slow roots of the daily ARIMA fits ({arma_['crack321']:.0f}, {arma_['brent_wti']:.0f} and {arma_['ho_rb']:.0f}). "
                         f"The daily AR(1) gives {LL('crack321', DS_, 'daily').half_life_days_approx:.0f}, {LL('brent_wti', DS_, 'daily').half_life_days_approx:.0f} and {LL('ho_rb', DS_, 'daily').half_life_days_approx:.0f}")
bwa = PHx("brent_wti", RAW_, "adf_p")
phase_of_max = phs_.loc[bwa.idxmax(), "freq"]
assert phase_of_max == "day 25" and bwa.drop(bwa.idxmax()).max() < 0.05
add("phase BW", f"except day 25 (ADF p = {bwa.max():.2f})")
hra = PHx("ho_rb", RAW_, "adf_p")
add("phase HORB", f"its raw month-end ADF fails ({hra.min():.2f} to {hra.max():.2f})")
assert PHx("ho_rb", DS_, "adf_p").max() < 0.05
cka, ckd = PHx("crack321", RAW_, "adf_p"), PHx("crack321", DS_, "adf_p")
add("phase crack", f"({cka.min():.3f} to {cka.max():.3f} raw) and mostly fails once seasonal effects are removed ({ckd.min():.2f} to {ckd.max():.2f})")
assert PHx("crack321", RAW_, "pp_p").max() < 0.05 and PHx("crack321", DS_, "pp_p").max() < 0.05 and (ckd > 0.05).sum() >= 4

pct1 = lambda x: f"{int(float(x) * 100 + 0.5 + 1e-9)}%"
add("power step4", f"is rejected by KPSS {pct1(pwr_.loc['crack321', 'kpss_rejects_if_truly_stationary'])}, {pct1(pwr_.loc['brent_wti', 'kpss_rejects_if_truly_stationary'])} and "
                   f"{pct1(pwr_.loc['ho_rb', 'kpss_rejects_if_truly_stationary'])} of the time, and ADF fails to reject it only "
                   f"{100 * pwr_.loc['crack321', 'adf_fails_to_reject_if_truly_stationary']:.0f}%, {100 * pwr_.loc['brent_wti', 'adf_fails_to_reject_if_truly_stationary']:.0f}% and "
                   f"{100 * pwr_.loc['ho_rb', 'adf_fails_to_reject_if_truly_stationary']:.1f}% of the time")
mmb = {sp: mma_[(mma_.series == sp) & (mma_.criterion == "BIC")].iloc[0] for sp in SH_}
assert mmb["crack321"].order == "(1,0,0)" and mmb["brent_wti"].order == "(1,0,0)" and mmb["ho_rb"].order == "(1,0,1)"
add("month-end arima", f"slowest-root half-lives of about {mmb['crack321'].slowest_ar_root_half_life_days * 21:.0f}, {mmb['brent_wti'].slowest_ar_root_half_life_days * 21:.0f} and {mmb['ho_rb'].slowest_ar_root_half_life_days * 21:.0f} days")
add("a3 crack", f"the same removal gives ADF p = {LL('crack321', DS_, 'month-end').adf_p:.3f} (PP {LL('crack321', DS_, 'month-end').pp_p:.3f})")
add("a3 gap", f"fails ADF at p = {sd_.loc['crack321', 'adf_p_accumulated_roll_gap']:.2f} for the crack")
add("a3 halflives", f"about {LL('crack321', DS_, 'month-end').half_life_days_approx:.0f}, {LL('brent_wti', DS_, 'month-end').half_life_days_approx:.0f} and {LL('ho_rb', DS_, 'month-end').half_life_days_approx:.0f} days. Different object")
hl_r = lambda sp: (PHx(sp, DS_, "half_life_days_approx").min(), PHx(sp, DS_, "half_life_days_approx").max())
add("phase half-lives", f"Brent/WTI {hl_r('brent_wti')[0]:.0f} to {hl_r('brent_wti')[1]:.0f} days, HO/RB {hl_r('ho_rb')[0]:.0f} to {hl_r('ho_rb')[1]:.0f}, crack {hl_r('crack321')[0]:.0f} to {hl_r('crack321')[1]:.0f}")
ksz_ = pd.read_csv(RS / "kpss_size.csv")
dk_ = ksz_[ksz_.form.str.startswith("log")].kpss_rejects_if_truly_stationary
add("power ladder", f"only {100 * pwr_.adf_fails_to_reject_if_truly_stationary.min():.1f}% to {100 * pwr_.adf_fails_to_reject_if_truly_stationary.max():.0f}% of the time")
add("power kpss", f"KPSS rejects a stationary series {pct1(pwr_.kpss_rejects_if_truly_stationary.min())} to {pct1(pwr_.kpss_rejects_if_truly_stationary.max())} of the time at 238 points, against {pct1(dk_.min())} to {pct1(dk_.max())} for the log-ratios at daily frequency")
assert (pwr_.n == 238).all()

# =============================================================================== Steps 8-9: GARCH and VECM in the main deck (added 2026-11-05)
GV = ROOT / "output" / "exploratory" / "garch_vecm" / "tables"
gb_ = pd.read_csv(GV / "garch_band.csv")
gw_, gg_ = gb_[gb_.sigma == "frozen window sd"], gb_[gb_.sigma == "GARCH"]
assert 0.135 <= gw_.outside_stress.min() and gw_.outside_stress.max() < 0.155 and 0.06 <= gg_.outside_stress.min() and gg_.outside_stress.max() < 0.075
assert gw_.outside_calm.between(0.015, 0.025).all() and gg_.outside_calm.between(0.03, 0.04).all()
add("garch band", "the frozen ±2 sd band **14% to 15%** of the time, and the GARCH band **6% to 7%**")
gs_ = pd.read_csv(GV / "garch_sizing.csv")
gsg, gsw = gs_[gs_.sigma == "GARCH"], gs_[gs_.sigma == "frozen window sd"]
add("garch risk", f"on **{100 * gsg.share_over_2R.min():.1f}% to {100 * gsg.share_over_2R.max():.1f}%** of days with GARCH, against **{100 * gsw.share_over_2R.min():.1f}% to {100 * gsw.share_over_2R.max():.1f}%** "
                  f"with the frozen sd ({100 * gsw.share_over_2R_stress.min():.0f}% to {100 * gsw.share_over_2R_stress.max():.0f}% of stress days)")
add("garch risk max", f"reaches **${gsw.set_index('spread').loc['HO − RB', 'max_risk'] / 1e3:.0f}k a day** in 2022")
pm_ = pd.read_csv(O / "tables" / "04_param_medians_by_period.csv").set_index(["key", "period"])
vhl = {s: np.log(.5) / np.log(pm_.loc[(f"fixed|{s}", "2020–2026"), "garch_a"] + pm_.loc[(f"fixed|{s}", "2020–2026"), "garch_b"]) for s in NAMES}
add("garch vol half-life", f"**{vhl['crack321']:.0f}, {vhl['brent_wti']:.0f} and {vhl['ho_rb']:.0f} trading days**")
vfq = pd.read_csv(O / "tables" / "04_vol_forecast_test.csv", index_col=0)
add("garch qlike t", f"t = {m(vfq.loc['fixed|crack321', 't_stat'], 1)}, {m(vfq.loc['fixed|brent_wti', 't_stat'], 1)} and {m(vfq.loc['fixed|ho_rb', 't_stat'], 1)})")
lad_ = pd.read_csv(O / "tables" / "06_ladder_summary.csv").set_index(["key", "step"])
add("garch ladder HORB", f"| HO − RB | {m(lad_.loc[('fixed|ho_rb', 'core'), 'sharpe'])} → +{lad_.loc[('fixed|ho_rb', 'garch_sizing'), 'sharpe']:.2f} | "
                         f"**−${abs(lad_.loc[('fixed|ho_rb', 'core'), 'max_dd']) / 1e3:.0f}k → −${abs(lad_.loc[('fixed|ho_rb', 'garch_sizing'), 'max_dd']) / 1e3:.0f}k** |")
lp_ = pd.read_csv(GV / "local_projection.csv").set_index(["spread", "h"])
add("lp crack 63", f"crude closes about **{100 * lp_.loc[('crack321', 63), 'closed_by_B']:.0f}%** of a crack gap within 63 days while the products move about **{-100 * lp_.loc[('crack321', 63), 'closed_by_A']:.0f}% further away**")
add("lp t 63", "(t = " + ", ".join(f"{lp_.loc[(s, 63), 'closed_total'] / lp_.loc[(s, 63), 'se_total']:.1f}" for s in ("crack321", "brent_wti")) +
               f" and {lp_.loc[('ho_rb', 63), 'closed_total'] / lp_.loc[('ho_rb', 63), 'se_total']:.1f})")
gd_ = pd.read_csv(GV / "gap_decay.csv").set_index("spread")
for s_, lab_ in (("crack321", "3-2-1 crack"), ("brent_wti", "Brent / WTI"), ("ho_rb", "HO / RB")):
    r_ = gd_.loc[s_]
    add(f"gap decay {s_}", f"| {lab_} | {r_.half_life:.0f} days | {100 * r_.open_at_exit:.0f}% ({dh[s_]:.0f} days) / {100 * r_.lp_open_21:.0f}% | "
                           f"{100 * r_.open_63:.0f}% / {100 * r_.lp_open_63:.0f}% | {100 * r_.open_252:.0f}% / {100 * r_.lp_open_252:.0f}% |")

missing = [(lab, s) for lab, s in checks if s not in DECK]
print(f"{len(checks) - len(missing)} of {len(checks)} quoted numbers match")
for lab, s in missing:
    print(f"  MISMATCH {lab}: {s!r}")
sys.exit(1 if missing else 0)
