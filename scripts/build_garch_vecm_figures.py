"""Figures 19-24 for the GARCH (Step 8) and VECM (Step 9) slides of report/oil_deck.md, plus the numbers behind them.

Usage:  .venv/Scripts/python.exe scripts/build_garch_vecm_figures.py
Reads data/clean/futures_spreads.csv, data/derived/backtest_daily_fixed_*.csv (the GARCH forecast `sigma_next`
the strategy used, point in time), output/tables/04_param_medians_by_period.csv and
output/exploratory/ratio_stationarity/tables/vecm_logs_known.csv. Nothing is refitted.

Writes report/deck_figures/19_garch_band.png, 20_garch_sizing_chain.png, 21_vecm_who_moves.png,
22_vecm_gap_decay.png, copies 23 (GARCH forecast vs realised) and 24 (rolling VECM), and
output/exploratory/garch_vecm/tables/{garch_band,garch_sizing,local_projection}.csv.

Two checks are new here, both descriptive (added 2026-11-05, after every result was seen):
  * band coverage: share of days the move falls outside ±2 sigma, GARCH against the frozen window sd, calm and stress days;
  * local projection: cumulative held-contract log return of each side over the next h days, regressed on today's
    (deseasonalised) market log-ratio, HAC errors with h lags. It is the VECM's alpha generalised to horizon h, and an
    empirical version of the (1 + alpha)^h decay curve. Full-sample, overlapping, persistent regressor: read the
    shape, not the third decimal.
"""
from __future__ import annotations

import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from oilrv.config import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import build_deck_figures as B  # noqa: E402  (style, LAB, COL, WIN, shade, save, fut)
import run_ar_acf_pacf as AR  # noqa: E402  (deseason)

sys.stdout.reconfigure(encoding="utf-8")
D, O = ROOT / "data", ROOT / "output"
TAB = O / "exploratory" / "garch_vecm" / "tables"
WINDOW, R = 756, 6000.0
M = {"crack321": 3000.0, "brent_wti": 1000.0, "ho_rb": 1000.0}
G, W, GREY = "#047857", "#dc2626", "#9ca3af"
SIDES = {"crack321": ("products (2 RB + HO)", "crude (3 CL)"), "brent_wti": ("Brent", "WTI"), "ho_rb": ("ULSD (HO)", "RBOB")}
HOLD = {"crack321": 31, "brent_wti": 23, "ho_rb": 26}   # daily ratio strategy, average hold (check_deck: "horizon: daily holds")


def stress_mask(idx):
    m = pd.Series(False, index=idx)
    for a, b in B.WIN.values():
        m |= (idx >= a) & (idx <= b)
    return m


def vol_data() -> dict[str, pd.DataFrame]:
    """dx: back-adjusted daily change. g: GARCH sigma for that day (forecast made the close before).
    w: frozen window sd, refit at each month start on the previous 756 changes (the staircase the GARCH replaces)."""
    f = B.fut()
    out = {}
    for k in B.LAB:
        d = pd.read_csv(D / "derived" / f"backtest_daily_fixed_{k}.csv", index_col=0, parse_dates=True)
        dx = f[k].dropna().diff().dropna()
        idx, v = dx.index, dx.to_numpy()
        month = pd.Series(idx.to_period("M"), index=idx)
        starts = np.flatnonzero((month != month.shift()).to_numpy())
        w = pd.Series(np.nan, index=idx)
        for j, i0 in enumerate(starts):
            if i0 >= WINDOW:
                w.iloc[i0:(starts[j + 1] if j + 1 < len(starts) else len(v))] = v[i0 - WINDOW:i0].std(ddof=1)
        out[k] = pd.DataFrame({"dx": dx, "g": d["sigma_next"].shift(1).reindex(idx), "w": w}).dropna()
    return out


# --------------------------------------------------------------------------- 19: the band
def fig19(V):
    rows = []
    for k, df in V.items():
        st = stress_mask(df.index)
        for lab, s in (("GARCH", df.g), ("frozen window sd", df.w)):
            out = df.dx.abs() > 2 * s
            rows.append({"spread": B.LAB[k], "sigma": lab, "outside_all": out.mean(), "outside_calm": out[~st].mean(),
                         "outside_stress": out[st].mean(), "n_stress": int(st.sum()), "n_calm": int((~st).sum())})
    cov = pd.DataFrame(rows)

    fig = plt.figure(figsize=(12.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, .05, 1])
    ax = fig.add_subplot(gs[0, 0])
    w = V["ho_rb"].loc["2021-07-01":"2023-06-30"]
    ax.bar(w.index, w.dx, width=1.0, color="#6b7280", lw=0)
    ax.fill_between(w.index, -2 * w.g, 2 * w.g, color=G, alpha=.18, lw=0, step="mid", label="GARCH ±2σ̂ (forecast the day before)")
    ax.plot(w.index, 2 * w.g, color=G, lw=.9, drawstyle="steps-mid")
    ax.plot(w.index, -2 * w.g, color=G, lw=.9, drawstyle="steps-mid")
    ax.plot(w.index, 2 * w.w, color=W, lw=1.3, ls="--", drawstyle="steps-post", label="±2 × frozen 3-year sd")
    ax.plot(w.index, -2 * w.w, color=W, lw=1.3, ls="--", drawstyle="steps-post")
    out_w = w[w.dx.abs() > 2 * w.w]
    ax.scatter(out_w.index, out_w.dx, s=7, color=W, zorder=3, label="outside the frozen band")
    ax.axhline(0, color="#444", lw=.5)
    ax.axvline(pd.Timestamp("2022-02-24"), color="#111", lw=.7, ls=":")
    ax.text(pd.Timestamp("2022-02-28"), ax.get_ylim()[1] * .9, "24 Feb 2022", fontsize=8)
    ax.set_ylabel("daily change, \\$/bbl")
    ax.set_title("HO − RB, daily moves: the GARCH band widens with the storm")
    ax.legend(loc="lower left", fontsize=8, ncol=1)

    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(3)
    for i, (lab, c) in enumerate((("frozen window sd", W), ("GARCH", G))):
        sub = cov[cov.sigma == lab].set_index("spread").loc[list(B.LAB.values())]
        ax2.bar(x + (i - .5) * .38, 100 * sub.outside_stress, .36, color=c, label=f"{lab}, stress days")
        ax2.scatter(x + (i - .5) * .38, 100 * sub.outside_calm, marker="_", s=260, color="#111", zorder=3,
                    label="calm days" if i == 1 else None)
    ax2.axhline(100 * 2 * (1 - 0.97725), color="#444", lw=.7, ls=":")
    ax2.text(-.55, 100 * 2 * (1 - 0.97725) + .3, "4.6%: a normal curve", fontsize=7.5, ha="left")
    ax2.set_xticks(x, list(B.LAB.values()), fontsize=8.5)
    ax2.set_ylabel("% of days outside ±2σ")
    ax2.set_title("How often a day breaks the band")
    ax2.set_ylim(0, 19)
    ax2.legend(fontsize=7.5, loc="upper center", ncol=1)
    B.save(fig, "19_garch_band.png")
    return cov


# --------------------------------------------------------------------------- 20: the sizing chain
def fig20(V):
    rows = []
    risk = {}
    for k, df in V.items():
        for lab, s in (("GARCH", df.g), ("frozen window sd", df.w)):
            pnl = R * df.dx / s                       # one day's P&L if the rule's units were held every day
            rr = pnl.rolling(21).std()
            risk[(k, lab)] = rr
            st = stress_mask(rr.index)
            rows.append({"spread": B.LAB[k], "sigma": lab, "median_risk": rr.median(), "share_over_2R": (rr > 2 * R).mean(),
                         "share_over_2R_stress": (rr[st] > 2 * R).mean(), "max_risk": rr.max(), "p95_risk": rr.quantile(.95)})
    # days after 24 Feb 2022 until units are half of the 23 Feb level (HO - RB), each rule
    df = V["ho_rb"]
    u_g, u_w = R / (df.g * M["ho_rb"]), R / (df.w * M["ho_rb"])
    t0 = pd.Timestamp("2022-02-23")
    half = {}
    for lab, u in (("GARCH", u_g), ("frozen window sd", u_w)):
        after = u.loc[t0:]
        hit = after[after <= .5 * after.iloc[0]]
        half[lab] = int(after.index.get_loc(hit.index[0])) if len(hit) else None
    tab = pd.DataFrame(rows)

    w0, w1 = "2019-01-01", "2026-10-26"
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True)
    d = df.loc[w0:w1]
    axs[0].plot(d.index, d.g, color=G, lw=.9, label="GARCH σ̂")
    axs[0].plot(d.index, d.w, color=W, lw=1.4, drawstyle="steps-post", label="frozen 3-year sd")
    axs[0].set_ylabel("\\$/bbl a day")
    axs[0].set_title("1. Forecast size of tomorrow's move")
    axs[1].plot(d.index, u_g.loc[w0:w1], color=G, lw=.9)
    axs[1].plot(d.index, u_w.loc[w0:w1], color=W, lw=1.4, drawstyle="steps-post")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("units (log scale)")
    axs[1].set_title("2. Units = \\$6,000 / (σ × \\$1,000 per unit)")
    for lab, c in (("GARCH", G), ("frozen window sd", W)):
        axs[2].plot(d.index, risk[("ho_rb", lab)].loc[w0:w1] / 1e3, color=c, lw=1.1 if lab == "GARCH" else 1.3)
    axs[2].axhline(R / 1e3, color="#111", lw=.8, ls="--")
    axs[2].text(d.index[5], R / 1e3 + 1.5, "budget: \\$6k a day", fontsize=8)
    axs[2].set_ylabel("\\$k a day")
    axs[2].set_title("3. Risk actually taken: 21-day sd of daily P&L if the position were held every day")
    for a in axs:
        B.shade(a)
        a.set_xlim(pd.Timestamp(w0), pd.Timestamp(w1))
    axs[0].legend(loc="upper left", fontsize=8.5)
    B.save(fig, "20_garch_sizing_chain.png")
    return tab, half


# --------------------------------------------------------------------------- VECM data
def vecm_data() -> dict[str, dict]:
    f = B.fut()
    g = lambda c: f[c].astype(float)
    spec = {"crack321": (2 * g("crack321__rb_bbl_unadj") + g("crack321__ho_bbl_unadj"), 3 * g("crack321__cl_bbl_unadj"),
                         2 * g("crack321__rb_bbl") + g("crack321__ho_bbl"), 3 * g("crack321__cl_bbl"), True),
            "brent_wti": (g("brent_wti__brent_bbl_unadj"), g("brent_wti__cl_bbl_unadj"), g("brent_wti__brent_bbl"),
                          g("brent_wti__cl_bbl"), False),
            "ho_rb": (g("ho_rb__ho_bbl_unadj"), g("ho_rb__rb_bbl_unadj"), g("ho_rb__ho_bbl"), g("ho_rb__rb_bbl"), True)}
    out = {}
    for k, (A, Bm, Aa, Ba, seas) in spec.items():
        lr = np.log(A / Bm)
        out[k] = dict(gap=AR.deseason(lr) if seas else lr, hA=np.log1p(Aa.diff() / A.shift(1)), hB=np.log1p(Ba.diff() / Bm.shift(1)))
    return out


HS = [1, 5, 10, 21, 42, 63, 84, 126, 189, 252]


def local_projection(VD) -> pd.DataFrame:
    rows = []
    for k, d in VD.items():
        gap = (d["gap"] - d["gap"].mean()).rename("gap")
        for side in ("hA", "hB"):
            r = d[side]
            for h in HS:
                y = r[::-1].rolling(h).sum()[::-1].shift(-1).rename("y")     # sum of returns t+1 .. t+h
                df = pd.concat([y, gap], axis=1).dropna()
                fit = sm.OLS(df.y, sm.add_constant(df.gap)).fit(cov_type="HAC", cov_kwds={"maxlags": h})
                rows.append({"spread": k, "side": side, "h": h, "beta": fit.params["gap"], "se": fit.bse["gap"]})
    lp = pd.DataFrame(rows)
    a = lp[lp.side == "hA"].set_index(["spread", "h"])
    b = lp[lp.side == "hB"].set_index(["spread", "h"])
    closed = pd.DataFrame({"closed_by_A": -a.beta, "closed_by_B": b.beta, "se_A": a.se, "se_B": b.se})
    tot = []
    for k, d in VD.items():
        gap = (d["gap"] - d["gap"].mean()).rename("gap")
        r = d["hA"] - d["hB"]                                               # held log-ratio return
        for h in HS:
            y = r[::-1].rolling(h).sum()[::-1].shift(-1).rename("y")
            df = pd.concat([y, gap], axis=1).dropna()
            fit = sm.OLS(df.y, sm.add_constant(df.gap)).fit(cov_type="HAC", cov_kwds={"maxlags": h})
            tot.append({"spread": k, "h": h, "closed_total": -fit.params["gap"], "se_total": fit.bse["gap"]})
    closed = closed.join(pd.DataFrame(tot).set_index(["spread", "h"]))
    closed["still_open"] = 1 - closed.closed_total
    return closed.reset_index()


# --------------------------------------------------------------------------- 21: who moves
def fig21(lp):
    vk = pd.read_csv(O / "exploratory" / "ratio_stationarity" / "tables" / "vecm_logs_known.csv")
    first = {"crack321": {"RB", "HO"}, "brent_wti": {"BRENT"}, "ho_rb": {"HO"}}
    name = {"RB": "RBOB", "HO": "ULSD", "CL": "WTI", "BRENT": "Brent"}
    bars = []
    for k in B.LAB:
        s = vk[(vk.spread == k) & vk.equation.str.startswith("held log return of ")]
        for _, r in s.iterrows():
            leg = r.equation.split()[-1]
            sign = -1 if leg in first[k] else 1
            bars.append((f"{B.LAB[k]}: {name[leg]}", sign * r.t, k))

    fig = plt.figure(figsize=(15, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=.55)
    # (a) schematic
    ax = fig.add_subplot(gs[0, 0])
    t = np.arange(0, 60)
    b_ = np.where(t < 20, 0.0, 0.06 * (1 - np.exp(-(t - 20) / 12.0)))
    ax.plot(t, np.where(t < 20, 1.0, 1.06), color="#c4531a", lw=2.2, label="first leg (e.g. products): the shock hits here")
    ax.plot(t, 1.0 + b_ - 0.002, color="#1f5fa8", lw=2.2, label="second leg (e.g. crude): catches up")
    ax.annotate("", xy=(21, 1.0), xytext=(21, 1.06), arrowprops=dict(arrowstyle="<->", color="#111"))
    ax.text(19.5, 1.03, "the gap\nopens", fontsize=8.5, va="center", ha="right")
    ax.text(31, 1.072, "α ≈ 0: stays put", fontsize=8.5, color="#c4531a")
    ax.annotate("α > 0: closes a share\nof the gap each day", xy=(33, 1.0 + 0.06 * (1 - np.exp(-13 / 12))),
                xytext=(34, 1.005), fontsize=8.5, color="#1f5fa8", arrowprops=dict(arrowstyle="->", color="#1f5fa8"))
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_ylim(.975, 1.10)
    ax.set_xlabel("time")
    ax.set_ylabel("log price")
    ax.set_title("(a) What a VECM measures")
    ax.legend(loc="upper left", fontsize=7.5)
    # (b) VECM t-stats
    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(bars))[::-1]
    first_leg = {"RBOB", "ULSD", "Brent"}
    ax.barh(y, [b[1] for b in bars], height=.62,
            color=["#c4531a" if (b[0].split(": ")[1] in first_leg and not b[0].startswith("HO − RB: RBOB")) else "#1f5fa8" for b in bars])
    for i in (2.5, 4.5):
        ax.axhline(i, color="#bbb", lw=.6)
    ax.axvline(0, color="#444", lw=.6)
    for v in (-2, 2):
        ax.axvline(v, color="#444", lw=.7, ls="--")
    ax.set_yticks(y, [b[0] for b in bars], fontsize=8.5)
    ax.set_xlabel("t-statistic (+ = closes the gap)")
    ax.set_title("(b) VECM, daily: which leg closes it")
    # (c) local projection at 63 days: share of the gap closed by each side
    ax = fig.add_subplot(gs[0, 2])
    s = lp[lp.h == 63].set_index("spread").loc[list(B.LAB)]
    x = np.arange(3)
    ax.bar(x - .2, 100 * s.closed_by_A, .38, color="#c4531a", yerr=196 * s.se_A, capsize=3, label="first leg (products / Brent / ULSD)")
    ax.bar(x + .2, 100 * s.closed_by_B, .38, color="#1f5fa8", yerr=196 * s.se_B, capsize=3, label="second leg (crude / WTI / RBOB)")
    ax.axhline(0, color="#444", lw=.6)
    ax.set_xticks(x, list(B.LAB.values()), fontsize=8.5)
    ax.set_ylabel("% of today's gap closed within 63 days")
    ax.set_title("(c) In the data: who closed it within 63 days")
    ax.legend(fontsize=7.5, loc="upper left")
    B.save(fig, "21_vecm_who_moves.png")
    return pd.DataFrame(bars, columns=["leg", "corrective_t", "spread"])


# --------------------------------------------------------------------------- 22: the gap decays slowly
def fig22(lp):
    vk = pd.read_csv(O / "exploratory" / "ratio_stationarity" / "tables" / "vecm_logs_known.csv")
    sp = vk[vk.equation.str.startswith("held log-ratio")].set_index("spread")
    pm = pd.read_csv(O / "tables" / "04_param_medians_by_period.csv")
    pm = pm[pm.period == "2020–2026"].set_index("key")
    h = np.arange(0, 253)
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, k in zip(axs, B.LAB):
        al = sp.loc[k, "alpha"]
        model = (1 + al) ** h
        ab = pm.loc[f"fixed|{k}", "garch_a"] + pm.loc[f"fixed|{k}", "garch_b"]
        ax.plot(h, 100 * model, color=B.COL[k], lw=2, label="VECM: (1 + α)^days")
        e = lp[lp.spread == k].set_index("h")
        ax.errorbar(e.index, 100 * e.still_open, yerr=196 * e.se_total, fmt="o", color="#111", ms=4, capsize=2, lw=.8,
                    label="measured (local projection, 95%)")
        ax.plot(h, 100 * ab ** h, color="#6b7280", lw=1, ls="--", label="a volatility shock (GARCH a + b)")
        ax.text(200, 100 * ab ** 200 + 4, f"vol, half-life {np.log(.5) / np.log(ab):.0f} d", fontsize=7.5, color="#6b7280", ha="center")
        hd = HOLD[k]
        ax.axvline(hd, color="#111", lw=.8, ls=":")
        ax.scatter([hd], [100 * (1 + al) ** hd], color=B.COL[k], zorder=4, s=30)
        ax.annotate(f"average exit, {hd} days:\n{100 * (1 + al) ** hd:.0f}% still open (VECM),\n"
                    f"{100 * e.loc[21, 'still_open']:.0f}% measured at 21 days", (hd, 100 * (1 + al) ** hd),
                    xytext=(hd + 30, 104), fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#555"))
        ax.axhline(50, color="#bbb", lw=.6)
        ax.set_title(B.LAB[k])
        ax.set_xlabel("days held")
        ax.set_xlim(0, 252)
        rows.append({"spread": k, "alpha": al, "half_life": np.log(.5) / np.log(1 + al), "open_at_exit": (1 + al) ** hd,
                     "open_63": (1 + al) ** 63, "open_252": (1 + al) ** 252, "vol_half_life": np.log(.5) / np.log(ab),
                     "lp_open_21": e.loc[21, "still_open"], "lp_open_63": e.loc[63, "still_open"], "lp_open_126": e.loc[126, "still_open"],
                     "lp_open_252": e.loc[252, "still_open"]})
    axs[0].set_ylabel("% of today's gap still open")
    axs[0].set_ylim(-10, 115)
    axs[0].set_ylim(-10, 125)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=[0, .07, 1, 1])
    fig.savefig(B.OUT / "22_vecm_gap_decay.png")
    plt.close(fig)
    print("wrote 22_vecm_gap_decay.png")
    return pd.DataFrame(rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    V = vol_data()
    cov = fig19(V)
    sizing, half = fig20(V)
    lp = local_projection(vecm_data())
    bars = fig21(lp)
    decay = fig22(lp)
    shutil.copy(O / "exploratory" / "horizon_tests" / "figures" / "garch_forecast_vs_realised.png", B.OUT / "23_garch_forecast_vs_realised.png")
    shutil.copy(O / "figures" / "04_vecm_and_johansen_weights.png", B.OUT / "24_vecm_rolling.png")
    cov.to_csv(TAB / "garch_band.csv", index=False)
    sizing.to_csv(TAB / "garch_sizing.csv", index=False)
    lp.to_csv(TAB / "local_projection.csv", index=False)
    decay.to_csv(TAB / "gap_decay.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nband coverage\n", cov.round(4).to_string(index=False))
    print("\nrisk taken\n", sizing.round(3).to_string(index=False))
    print("\nHO-RB days to halve units after 23 Feb 2022:", half)
    print("\nVECM corrective t\n", bars.round(2).to_string(index=False))
    print("\nlocal projection\n", lp.round(3).to_string(index=False))
    print("\ndecay\n", decay.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
