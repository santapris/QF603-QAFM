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

missing = [(lab, s) for lab, s in checks if s not in DECK]
print(f"{len(checks) - len(missing)} of {len(checks)} quoted numbers match")
for lab, s in missing:
    print(f"  MISMATCH {lab}: {s!r}")
sys.exit(1 if missing else 0)
