"""The statistical pipeline for TREND and CARRY, parallel to the nine steps for mean reversion.

Usage:  PYTHONPATH=src python scripts/run_trend_carry_tests.py [--reps 600] [--quick]
Requires the Bloomberg export and the trend/carry backtest output. Writes
output/exploratory/trend_carry/tables/tests_*.csv and figures/trend_carry_tests.png.

WHY A DIFFERENT PIPELINE, NOT THE SAME NINE STEPS
-------------------------------------------------
The mean-reversion pipeline is a SPECIFICATION SEARCH: it asks what process generated
the level (unit root or not, how many AR terms, is there a cointegrating vector). That is
the right question when the claim is "the level returns to an anchor", because the claim
is a statement about the data-generating process.

Trend and carry make a different claim: "this SIGNAL forecasts returns, and the return it
earns is compensation for bearing a risk." That is a PREDICTIVE and RISK-PREMIUM question.
Re-running ADF or Johansen on a signal answers nothing about it. The parallel steps are:

  mean reversion                        trend / carry
  --------------------------------      ------------------------------------------
  3. unit root, cointegration           A. is the SIGNAL itself stationary and usable?
  4-5. SARIMA / AR order: forecastable?  B. predictive regression, with a bias-corrected null
  9. VECM: what pulls it back?          C. economic mechanism (inventory; run separately)
  (none)                                D. passive vs dynamic: is it just a long bias?
  (none)                                E. factor spanning: is it just oil beta?
  (none)                                F. how much is the Sharpe inflated by the search?

D, E and F have no mean-reversion equivalent because mean reversion never got far enough
to need them: it failed at step 3. They are the tests that matter once a backtest shows a
positive Sharpe, and they are the ones that stop "trend and carry won" being a result
pulled out of thin air.

Added 2026-11-02, AFTER the trend/carry backtest was seen. Exploratory, not a gate, and it
cannot change the registered verdict. It can only qualify how much the 0.45 Sharpe is worth.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from oilrv import futures
from oilrv import horizon as H
from oilrv.config import ROOT

TC = ROOT / "output/exploratory/trend_carry"
TAB, FIG = TC / "tables", TC / "figures"
START = "2010-02-01"
LABEL = {"cl": "WTI", "brent": "Brent", "ho": "ULSD", "rb": "RBOB"}
REGISTERED_CARRY = ("cl", "brent")          # the pre-registered carry sleeve
HORIZONS = (21, 63, 126)


def save(df: pd.DataFrame, name: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / name)
    print(f"  wrote tables/{name}")


# --------------------------------------------------------------------- data
def build() -> dict:
    """Back-adjusted P&L series and the M1-M2 curve slope for each root."""
    fut = futures.build_futures()
    panels, roots = futures.load_panels(), futures.leg_roots()
    badj = {"cl": fut["crack321__cl_bbl"], "brent": fut["brent_wti__brent_bbl"],
            "rb": fut["crack321__rb_bbl"], "ho": fut["crack321__ho_bbl"]}
    slope = {}
    for leg, panel in panels.items():
        one = {leg: panel}
        h1 = futures.held_months(one, roots, k=1, roll_bdays=5).dropna()
        h2 = futures.held_months(one, roots, k=2, roll_bdays=5).dropna()
        p1 = pd.Series([panel.at[t, m] for t, m in zip(h1.index, h1)], index=h1.index)
        p2 = pd.Series([panel.at[t, m] for t, m in zip(h2.index, h2)], index=h2.index)
        sc = 42.0 / 100.0 if roots[leg] in ("XB", "HO") else 1.0
        slope[leg] = ((p1 - p2) * sc).dropna()
    return {"badj": badj, "slope": slope}


# ------------------------------------------------------------------ A. signal
def part_a(d: dict) -> pd.DataFrame:
    """Is the signal stationary? A non-stationary signal cannot be a trading rule:
    its sign would be a one-way bet, not a repeatable condition."""
    print("\n[A] Is the CARRY SIGNAL itself stationary? (the step-3 analogue)")
    print("    A signal must revert, or 'long when backwardated' is just a long bias.\n")
    rows = []
    for leg in ("ho", "brent", "cl", "rb"):
        s = d["slope"][leg]
        s = s[s.index >= pd.Timestamp(START)]
        x = s.to_numpy()
        t_adf, p_adf, _ = H.adf(x)
        k_stat, k_p = H.kpss(x)
        b = H.ols(x[1:], np.column_stack([x[:-1], np.ones(len(x) - 1)]))["b"][0]
        hl = np.log(0.5) / np.log(b) if 0 < b < 1 else np.inf
        rows.append({"instr": LABEL[leg], "adf_t": t_adf, "adf_p": p_adf,
                     "kpss_stat": k_stat, "kpss_p": k_p, "ar1": b, "half_life_days": hl,
                     "share_backwardated": float((s > 0).mean()), "n": len(s)})
    df = pd.DataFrame(rows).set_index("instr")
    print(df.round(3).to_string())
    save(df, "tests_signal_stationarity.csv")
    return df


# -------------------------------------------------------- B. predictive test
def predictive(badj: pd.Series, sig: pd.Series, k: int) -> dict:
    """Forward k-day P&L ($/bbl) regressed on the signal. Newey-West at k+5."""
    b = badj.dropna()
    fwd = (b.shift(-k) - b).rename("f")
    df = pd.concat([sig.reindex(b.index).ffill().rename("s"), fwd], axis=1).dropna()
    df = df[df.index >= pd.Timestamp(START)]
    X = np.column_stack([df["s"].to_numpy(), np.ones(len(df))])
    r = H.ols(df["f"].to_numpy(), X, hac_lags=k + 5)
    return {"beta": float(r["b"][0]), "t": float(r["t"][0]), "r2": float(r["r2"]),
            "n": len(df)}


def predictive_null(n: int, sd: float, ar1: float, sig_sd: float, rho: float,
                    k: int, reps: int, seed: int) -> np.ndarray:
    """Null distribution of the t above when the signal has NO forecasting power.

    The signal is persistent (AR(1) close to 1) and its innovations are correlated with
    the return (rho). That is the Stambaugh (1999) setting, where OLS t is biased and
    the normal critical value over-rejects. Returns are iid by construction here, so any
    mass beyond +/-1.96 is the bias, not predictability.
    """
    rng = np.random.default_rng(seed)
    out = []
    sd_e = sig_sd * np.sqrt(max(1e-12, 1 - ar1 ** 2))
    for _ in range(reps):
        z = rng.standard_normal((n, 2))
        r = z[:, 0] * sd                                   # iid returns: no predictability
        e = (rho * z[:, 0] + np.sqrt(max(0.0, 1 - rho ** 2)) * z[:, 1]) * sd_e
        s = np.empty(n)
        s[0] = e[0] / np.sqrt(max(1e-12, 1 - ar1 ** 2))
        for i in range(1, n):
            s[i] = ar1 * s[i - 1] + e[i]
        cum = np.concatenate([[0.0], np.cumsum(r)])
        fwd = cum[k:] - cum[:-k]
        ss = s[:len(fwd)]
        X = np.column_stack([ss, np.ones(len(fwd))])
        tt = H.ols(fwd, X, hac_lags=k + 5)["t"][0]
        if np.isfinite(tt):
            out.append(tt)
    return np.array(out)


def part_b(d: dict, reps: int) -> pd.DataFrame:
    print("\n[B] Does the signal FORECAST returns? (the step-4/5 analogue)")
    print("    Forward k-day P&L on the signal, Newey-West, against a simulated null")
    print("    that has the signal's persistence but no predictability (Stambaugh bias).\n")
    rows = []
    for leg in ("ho", "brent", "cl", "rb"):
        b = d["badj"][leg].dropna()
        sd = b.diff().std()
        for name, sig in (("carry (M1-M2)", d["slope"][leg]),
                          ("trend (252d change)", (b - b.shift(252)))):
            s = sig.reindex(b.index).ffill().dropna()
            s = s[s.index >= pd.Timestamp(START)]
            ar1 = float(H.ols(s.to_numpy()[1:],
                              np.column_stack([s.to_numpy()[:-1],
                                               np.ones(len(s) - 1)]))["b"][0])
            al = pd.concat([s.diff().rename("ds"), b.diff().rename("dr")], axis=1).dropna()
            rho = float(np.corrcoef(al["ds"], al["dr"])[0, 1])
            for k in HORIZONS:
                r = predictive(b, s, k)
                null = predictive_null(len(s), sd, min(ar1, 0.9999), s.std(), rho,
                                       k, reps, seed=k + hash(leg) % 1000)
                p = float((np.abs(null) >= abs(r["t"])).mean())
                rows.append({"instr": LABEL[leg], "signal": name, "k": k, **r,
                             "signal_ar1": ar1, "rho_innov": rho,
                             "crit95_sim": float(np.percentile(np.abs(null), 95)),
                             "p_sim": p})
    df = pd.DataFrame(rows).set_index(["signal", "instr", "k"]).sort_index()
    print(df[["beta", "t", "r2", "crit95_sim", "p_sim"]].round(4).to_string())
    save(df, "tests_predictive_regression.csv")
    return df


# ------------------------------------------------- D. passive vs dynamic
def part_d(d: dict) -> pd.DataFrame:
    """Koijen-Moskowitz-Pedersen-Vrugt decomposition.

    A carry strategy's position is sign(carry). Split it into its time-series average
    (the PASSIVE part: a constant long or short) and the deviation from that average
    (the DYNAMIC part: the actual timing). If the return is all passive, the signal is
    doing no work and the sleeve is a directional bet wearing a carry label.
    """
    print("\n[D] Is carry a TIMING signal, or just a structural long?")
    print("    position = average position (passive) + deviation (dynamic).\n")
    rows = []
    for leg in ("ho", "brent", "cl", "rb"):
        b = d["badj"][leg].dropna()
        sig = np.sign(d["slope"][leg]).reindex(b.index).ffill().shift(1)
        ret = b.diff()
        df = pd.concat([sig.rename("s"), ret.rename("r")], axis=1).dropna()
        df = df[df.index >= pd.Timestamp(START)]
        passive = df["s"].mean()
        pnl_tot = df["s"] * df["r"]
        pnl_pas = passive * df["r"]
        pnl_dyn = (df["s"] - passive) * df["r"]
        sh = lambda x: x.mean() / x.std() * np.sqrt(252)
        rows.append({"instr": LABEL[leg], "avg_position": passive,
                     "total_usd_per_bbl": pnl_tot.sum(),
                     "passive_usd": pnl_pas.sum(), "dynamic_usd": pnl_dyn.sum(),
                     "dynamic_share": pnl_dyn.sum() / pnl_tot.sum() if pnl_tot.sum() else np.nan,
                     "sharpe_total": sh(pnl_tot), "sharpe_dynamic": sh(pnl_dyn),
                     "t_dynamic": pnl_dyn.mean() / pnl_dyn.std() * np.sqrt(len(pnl_dyn))})
    df = pd.DataFrame(rows).set_index("instr")
    print(df.round(3).to_string())
    save(df, "tests_passive_vs_dynamic.csv")
    return df


# ------------------------------------------------------ E. factor spanning
def part_e(d: dict, books: pd.DataFrame) -> pd.DataFrame:
    """Does the book earn anything a passive long in oil would not?

    Regress each sleeve's daily P&L on the WTI and Brent daily changes. The intercept is
    what survives. A sleeve whose alpha dies once oil beta is removed is not a strategy.
    """
    print("\n[E] Is it just oil beta? Sleeve P&L on WTI and Brent daily changes.\n")
    f = pd.concat([d["badj"]["cl"].diff().rename("wti"),
                   d["badj"]["brent"].diff().rename("brent")], axis=1)
    rows = []
    for col in books.columns:
        y = books[col]
        df = pd.concat([y.rename("y"), f], axis=1, sort=False).dropna()
        df = df[df.index >= pd.Timestamp(START)]
        X = np.column_stack([df["wti"], df["brent"], np.ones(len(df))])
        r = H.ols(df["y"].to_numpy(), X, hac_lags=21)
        raw = df["y"].mean() / df["y"].std() * np.sqrt(252)
        resid = df["y"].to_numpy() - X @ r["b"]
        rows.append({"book": col, "beta_wti": r["b"][0], "beta_brent": r["b"][1],
                     "alpha_usd_per_day": r["b"][2], "alpha_t": r["t"][2],
                     "r2": r["r2"], "sharpe_raw": raw,
                     "sharpe_alpha": r["b"][2] / resid.std() * np.sqrt(252)})
    df = pd.DataFrame(rows).set_index("book")
    print(df.round(4).to_string())
    save(df, "tests_factor_spanning.csv")
    return df


# ------------------------------------- F. search cost: bootstrap + deflation
def stationary_bootstrap_sharpe(x: np.ndarray, reps: int, mean_block: int,
                                seed: int = 0) -> np.ndarray:
    """Politis-Romano stationary bootstrap of the annualised Sharpe.

    Resamples blocks of random length, so serial dependence and fat tails are preserved.
    The usual t = Sharpe * sqrt(years) assumes iid normal returns and overstates
    significance when neither holds.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    p = 1.0 / mean_block
    out = np.empty(reps)
    for b in range(reps):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for j in range(n):
            idx[j] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        s = x[idx]
        out[b] = s.mean() / s.std() * np.sqrt(252)
    return out


def deflated_sharpe(sr_ann: float, n: int, skew: float, kurt: float,
                    k_trials: int, sr_trial_sd_ann: float) -> dict:
    """Bailey & Lopez de Prado (2014).

    With K trials the best Sharpe is high even under the null. SR0 is the Sharpe the
    luckiest of K independent null strategies would be expected to show; the deflated
    Sharpe is the probability the observed one beats that, after correcting for the
    non-normal shape of the return distribution.
    """
    g = 0.5772156649
    sr = sr_ann / np.sqrt(252)                      # per-period, to match n
    sd_tr = sr_trial_sd_ann / np.sqrt(252)
    z1 = stats.norm.ppf(1 - 1.0 / k_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (k_trials * np.e))
    sr0 = sd_tr * ((1 - g) * z1 + g * z2)
    den = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / den)
    return {"sr_annual": sr_ann, "expected_max_null_sr": sr0 * np.sqrt(252),
            "deflated_sharpe_prob": dsr}


def part_f(books: pd.DataFrame, reps: int) -> pd.DataFrame:
    print("\n[F] How much of the Sharpe is the search itself?")
    print("    Block bootstrap (no iid assumption) and the deflated Sharpe over all")
    print("    strategy streams the project ran.\n")
    streams = stream_sharpes()
    k = len(streams)
    sd_tr = float(np.std(list(streams.values()), ddof=1))
    print(f"    streams counted ({k}): "
          + ", ".join(f"{a} {b:+.2f}" for a, b in streams.items()))
    print(f"    sd of stream Sharpes = {sd_tr:.3f}\n")
    rows = []
    for col in books.columns:
        x = books[col].to_numpy()
        x = x[~np.isnan(x)]
        d_ann = np.sqrt(252)
        sr = x.mean() / x.std() * d_ann
        boot = stationary_bootstrap_sharpe(x, reps, mean_block=63, seed=7)
        ds = deflated_sharpe(sr, len(x), float(stats.skew(x)),
                             float(stats.kurtosis(x) + 3), k, sd_tr)
        rows.append({"book": col, "sharpe": sr,
                     "t_iid": sr * np.sqrt(len(x) / 252),
                     "boot_sd": boot.std(), "boot_p_gt_0": float((boot <= 0).mean()),
                     "boot_ci5": float(np.percentile(boot, 5)),
                     "boot_ci95": float(np.percentile(boot, 95)),
                     **ds, "k_trials": k})
    df = pd.DataFrame(rows).set_index("book")
    print(df.round(4).to_string())
    save(df, "tests_search_cost.csv")
    return df


def stream_sharpes() -> dict:
    """Every strategy stream the project backtested, for the multiple-testing count."""
    out = {}
    lad = pd.read_csv(ROOT / "output/tables/06_ladder_summary.csv", index_col=[0, 1])
    for key in ("fixed|crack321", "fixed|brent_wti", "fixed|ho_rb", "johansen|crack321"):
        v, nm = key.split("|")
        tag = f"MR {nm}" + ("" if v == "fixed" else " (johansen)")
        out[tag] = float(lad.loc[(key, "logit_filter"), "sharpe"])
    pr = ROOT / "output/exploratory/pairs/tables/summary_pairs.csv"
    if pr.exists():
        out["pairs book"] = float(pd.read_csv(pr, index_col=0)["sharpe"].mean())
    bk = pd.read_csv(TC / "tables/summary_books.csv", index_col=0)
    for s in ("trend", "carry"):
        out[s] = float(bk.loc[s, "sharpe"])
    return out


# -------------------------------------------------------------------- figure
def figure(pred: pd.DataFrame, pas: pd.DataFrame, sc: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))
    order = ["ULSD", "Brent", "WTI", "RBOB"]

    # panel 1: predictive t vs the simulated critical value
    c = pred.xs("carry (M1-M2)", level="signal")
    w = 0.26
    for j, k in enumerate(HORIZONS):
        ax[0].bar([i + (j - 1) * w for i in range(4)],
                  [c.loc[(n, k), "t"] for n in order], width=w,
                  color=["#047857", "#10b981", "#6ee7b7"][j], label=f"{k}d")
    crit = c["crit95_sim"].mean()
    ax[0].axhline(crit, color="#111", ls="--", lw=1)
    ax[0].axhline(-crit, color="#111", ls="--", lw=1)
    ax[0].text(-.45, crit + .10, "5% critical value (simulated)", ha="left",
               fontsize=8.5, color="#333")
    ax[0].axhline(0, color="#111", lw=.8)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(order, fontsize=10)
    ax[0].set_ylabel("t-statistic", fontsize=10.5)
    ax[0].set_title("B. Does carry forecast returns?", fontweight="bold", fontsize=12)
    ax[0].legend(frameon=False, fontsize=9, ncol=3, loc="lower left")

    # panel 2: passive vs dynamic
    bw = 0.27
    ax[1].bar([i - bw for i in range(4)], [pas.loc[n, "total_usd_per_bbl"] for n in order],
              width=bw, color="#1e3a5f", label="total")
    ax[1].bar(range(4), [pas.loc[n, "passive_usd"] for n in order], width=bw,
              color="#9ca3af", label="passive (constant position)")
    ax[1].bar([i + bw for i in range(4)], [pas.loc[n, "dynamic_usd"] for n in order],
              width=bw, color="#047857", label="dynamic (the timing)")
    ax[1].axhline(0, color="#111", lw=.8)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(order, fontsize=10)
    ax[1].set_ylabel("cumulative P&L, $/bbl, one unit", fontsize=10.5)
    ax[1].set_title("D. Is it timing, or a structural long?", fontweight="bold",
                    fontsize=12)
    ax[1].legend(frameon=False, fontsize=8.5, loc="lower left")

    # panel 3: Sharpe vs what the search alone would produce
    bks = list(sc.index)
    ax[2].bar(range(len(bks)), [sc.loc[b, "sharpe"] for b in bks], width=.55,
              color="#2563eb", label="observed Sharpe")
    for i, b in enumerate(bks):
        ax[2].plot([i - .33, i + .33], [sc.loc[b, "boot_ci5"]] * 2, color="#111", lw=1.4)
        ax[2].plot([i - .33, i + .33], [sc.loc[b, "boot_ci95"]] * 2, color="#111", lw=1.4)
        ax[2].plot([i, i], [sc.loc[b, "boot_ci5"], sc.loc[b, "boot_ci95"]],
                   color="#111", lw=1.1)
    ax[2].axhline(sc["expected_max_null_sr"].iloc[0], color="#dc2626", ls="--", lw=1.4)
    ax[2].text((len(bks) - 1) / 2, 1.0,
               f"red line: the luckiest of {int(sc['k_trials'].iloc[0])} null streams "
               f"would show {sc['expected_max_null_sr'].iloc[0]:.2f}",
               ha="center", fontsize=9, color="#dc2626", fontweight="bold")
    ax[2].axhline(0.5, color="#111", lw=1.2)
    ax[2].text(-.45, .515, "G1 gate 0.5", fontsize=8.5)
    ax[2].axhline(0, color="#111", lw=.8)
    ax[2].set_xticks(range(len(bks)))
    ax[2].set_xticklabels(bks, fontsize=10)
    ax[2].set_ylabel("annualised Sharpe", fontsize=10.5)
    ax[2].set_title("F. Is 0.45 more than the search buys?", fontweight="bold",
                    fontsize=12)
    ax[2].legend(frameon=False, fontsize=9, loc="lower left")
    ax[2].set_ylim(-.33, 1.12)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=.25, lw=.7)
        a.set_axisbelow(True)
    fig.suptitle("The statistical pipeline for trend and carry", fontsize=14.5,
                 fontweight="bold", y=1.0)
    fig.text(.5, .915, "Bars with 90% stationary-bootstrap intervals in panel F. "
             "Everything is exploratory and cannot change the registered verdict.",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0, 1, .9])
    fig.savefig(FIG / "trend_carry_tests.png", dpi=170, facecolor="white")
    print("  wrote figures/trend_carry_tests.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 80 if a.quick else a.reps
    if not futures.available():
        raise SystemExit("Bloomberg export not found")
    d = build()
    books = pd.read_csv(TAB / "daily_books.csv", index_col=0, parse_dates=True)
    part_a(d)
    pred = part_b(d, reps)
    pas = part_d(d)
    part_e(d, books)
    sc = part_f(books, reps)
    figure(pred, pas, sc)
    print(f"\nDone. Outputs in {TC}")


if __name__ == "__main__":
    main()
