"""DESCRIPTIVE: what Sharpe should the registered rule earn if the fitted models were true, and how likely is a pass? -> output/exploratory/gate_power/.

Usage:  .venv/Scripts/python.exe scripts/run_gate_power.py
Requires run_ratio_stationarity.py (steps 10 and 11) and run_ratio_mr.py. Added 2026-11-04 after every strategy result was seen.
It is a model-based bridge from the statistics to the gates. It is not a backtest and it changes no verdict.

Three parts.
1. G1 power: the net Sharpe a 16.7-year sample reports is noisy (SE about sqrt((1 + SR^2 / 2) / years)), so a strategy whose true
   Sharpe is exactly 0.5 passes G1 about half the time, and one with true Sharpe 0 passes about 3% of the time.
2. The registered rule (enter at |z| >= 2 when re-armed, exit at |z| <= 0.5, stop at 4, time stop 3 half-lives, fill one close later,
   $0.05/bbl) simulated on four model worlds per spread, fitted in run_ratio_stationarity.py step 10:
     A  fixed k (AR(1) around a constant mean), rule uses the true mean and sd          (best case if k never moved)
     B  fixed k, rule estimates mean, sd and half-life on the trailing 756 days        (the registered rule, k fixed)
     C  moving k (local level + AR(1)), rule estimates on the trailing 756 days        (the registered rule, k moves)
     D  moving k, rule knows log k_t exactly and trades only the deviation u_t         (what perfect anchor tracking would be worth)
   Positions are constant-size (the simulated volatility is constant), P&L is in log-ratio units per unit of side notional and the
   cost is 0.05 / P_B per unit position change (P_B the mean price of the denominator, $/bbl). No rolls, no gaps, no stops that slip.
3. Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) of the trend and carry book across the six variants tried on this sample.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import kurtosis, norm, skew

from oilrv.config import ROOT, load_config
from oilrv.plotting import REF_LINE, SERIES, use_style

OUT = ROOT / "output" / "exploratory" / "gate_power"
RS = ROOT / "output" / "exploratory" / "ratio_stationarity" / "tables"
LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}
YEARS, T, W, STEP = 16.7183, 4987, 756, 21
N_PATHS = 200


# --------------------------------------------------------------------------- 1. G1 power
def sharpe_se(sr: float, years: float = YEARS) -> float:
    """Lo (2002) standard error of an annualised Sharpe ratio estimated from iid returns."""
    return float(np.sqrt((1 + 0.5 * sr ** 2) / years))


def g1_power(true_sr: float, bar: float = 0.5, years: float = YEARS) -> float:
    return float(1 - norm.cdf((bar - true_sr) / sharpe_se(true_sr, years)))


# --------------------------------------------------------------------------- 2. the rule on a model world
def make_world(rng, p: dict, world: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(l, u) of shape (n, T): the log-ratio and its deviation from the anchor."""
    if world in ("A", "B"):
        phi, s_e = p["phi_fixed"], p["sd_total"] * np.sqrt(1 - p["phi_fixed"] ** 2)
        lvl = np.zeros((n, T))
    else:
        phi, s_e = p["phi"], p["sigma_ar"]
        lvl = np.cumsum(rng.normal(0.0, p["sigma_level"], (n, T)), axis=1)
    u = lfilter([1.0], [1.0, -phi], rng.normal(0.0, s_e, (n, T + 300)), axis=1)[:, 300:]
    return lvl + u, u


def rolling_fit(x: np.ndarray, w: int = W, step: int = STEP):
    """Trailing-window mean, sd and AR(1) half-life, refreshed every `step` days and held in between (as the monthly refit)."""
    n, t = x.shape
    cs, cs2 = np.cumsum(np.c_[np.zeros(n), x], axis=1), np.cumsum(np.c_[np.zeros(n), x ** 2], axis=1)
    prod = np.c_[np.zeros((n, 1)), x[:, :-1] * x[:, 1:]]                   # x_{t-1} x_t at position t
    cp = np.cumsum(np.c_[np.zeros(n), prod], axis=1)
    mean, sd, hl = np.full((n, t), np.nan), np.full((n, t), np.nan), np.full((n, t), np.nan)
    for m in range(w, t, step):
        a, b = m - w, m                                                    # window rows a..b-1
        s1, s2 = cs[:, b] - cs[:, a], cs2[:, b] - cs2[:, a]
        mu = s1 / w
        var = (s2 - w * mu ** 2) / (w - 1)
        # OLS of x_t on x_{t-1} (with a constant) over the k = w - 1 pairs t = a+1 .. b-1
        k = w - 1
        sp, sc = s1 - x[:, b - 1], s1 - x[:, a]                            # sum of x_{t-1}, sum of x_t
        spp, spc = s2 - x[:, b - 1] ** 2, cp[:, b] - cp[:, a + 1]          # sum of x_{t-1}^2, sum of x_{t-1} x_t
        phi = (k * spc - sp * sc) / (k * spp - sp ** 2)
        h = np.where((phi > 0) & (phi < 1), np.log(2) / -np.log(np.clip(phi, 1e-9, 1 - 1e-12)), np.inf)
        hi = min(m + step, t)
        mean[:, m:hi], sd[:, m:hi], hl[:, m:hi] = mu[:, None], np.sqrt(var)[:, None], h[:, None]
    return mean, sd, hl


def run_rule(l: np.ndarray, z: np.ndarray, hl: np.ndarray, gate: bool, cost: float, entry=2.0, exit_=0.5, stop=4.0, tstop=3.0):
    """The registered rule on each path. Returns daily P&L (n, T) in log-ratio units and the trade counts."""
    n, t = l.shape
    dl = np.diff(l, axis=1, prepend=l[:, :1])
    pnl, trades = np.zeros((n, t)), np.zeros(n, int)
    for i in range(n):
        zi, hli, dli = z[i], hl[i], dl[i]
        pos, pend, armed, held, hl_in = 0, None, False, 0, np.nan
        for s in range(W, t):
            pnl_s = 0.0
            if pos != 0:
                pnl_s += pos * dli[s]
                held += 1
            if pend is not None:                                            # fill the decision made yesterday
                kind, side = pend
                if kind == "enter":
                    pos, held, hl_in = side, 0, hli[s - 1]
                    pnl_s -= cost
                else:
                    pos = 0
                    pnl_s -= cost
                    trades[i] += 1
                    armed = False
                pend = None
            pnl[i, s] = pnl_s
            zz = zi[s]
            if not np.isfinite(zz) or s == t - 1:
                continue
            if pos == 0:
                if abs(zz) < entry:
                    armed = True
                ok = (not gate) or (5 <= hli[s] <= 120)
                if armed and ok and entry <= abs(zz) < stop:
                    pend = ("enter", -int(np.sign(zz)))
            else:
                if abs(zz) >= stop or abs(zz) <= exit_ or held > tstop * hl_in:
                    pend = ("exit", 0)
    return pnl, trades


def sharpe_paths(pnl: np.ndarray) -> np.ndarray:
    x = pnl[:, W:]
    return x.mean(axis=1) / x.std(axis=1) * np.sqrt(252)


def simulate(name: str, p: dict, cost: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for world in "ABCD":
        l, u = make_world(rng, p, world, N_PATHS)
        if world in ("A", "D"):
            sd_u = p["sd_total"] if world == "A" else p["sd_u"]
            hl_true = np.log(2) / -np.log(p["phi_fixed"] if world == "A" else p["phi"])
            z = u / sd_u
            z[:, :W] = np.nan
            hl = np.full_like(l, hl_true)
            pnl, tr = run_rule(l, z, hl, gate=False, cost=cost)
        else:
            mean, sd, hl = rolling_fit(l)
            z = (l - mean) / sd
            pnl, tr = run_rule(l, z, hl, gate=True, cost=cost)
        sr = sharpe_paths(pnl)
        out.append({"spread": name, "world": world, "mean_sharpe": float(sr.mean()), "p05": float(np.percentile(sr, 5)),
                    "p50": float(np.median(sr)), "p95": float(np.percentile(sr, 95)),
                    "share_sharpe_gt_0.5": float((sr > 0.5).mean()), "mean_round_trips_per_year": float(tr.mean() / YEARS),
                    "sharpes": sr})
    return out


# --------------------------------------------------------------------------- 3. deflated Sharpe
def deflated_sharpe(returns: pd.Series, trial_sharpes_daily: np.ndarray, n_trials: int) -> dict:
    """Probability that the true Sharpe exceeds the expected maximum Sharpe of `n_trials` skill-less trials."""
    r = returns.dropna()
    sr = r.mean() / r.std()                                                 # per day
    g = 0.5772156649
    v = float(np.var(trial_sharpes_daily, ddof=1))
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    ku, sk = kurtosis(r, fisher=False), skew(r)
    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    dsr = norm.cdf((sr - sr0) * np.sqrt(len(r) - 1) / denom)
    return {"sharpe_annual": float(sr * np.sqrt(252)), "expected_max_null_sharpe_annual": float(sr0 * np.sqrt(252)),
            "dsr": float(dsr), "n_trials": n_trials, "sd_trial_sharpes_annual": float(np.sqrt(v) * np.sqrt(252)),
            "skew": float(sk), "kurtosis": float(ku), "n_days": int(len(r))}


def main() -> dict:
    use_style()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    # 1. G1 power
    srs = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0]
    pw = pd.DataFrame({"true_sharpe": srs, "se": [sharpe_se(s) for s in srs], "p_pass_G1": [g1_power(s) for s in srs]})
    # six independent skill-less variants: the chance that at least one clears the G1 bar (an upper bound; the variants are related)
    pw["p_at_least_one_of_six_passes_if_all_true_sharpes_equal_this"] = 1 - (1 - pw.p_pass_G1) ** 6
    pw.to_csv(OUT / "g1_power.csv", index=False)

    # 2. the rule on model worlds
    mdl = pd.read_csv(RS / "moving_anchor_model.csv").set_index("series")
    kc = pd.read_csv(RS / "k_consistency.csv").set_index("series")
    real = json.loads((ROOT / "output/exploratory/ratio_mr/tables/summary.json").read_text())["spreads"]
    slow = json.loads((ROOT / "output/exploratory/ratio_slow/tables/summary.json").read_text())["spreads"]
    rows, allsr = [], {}
    for i, n in enumerate(LAB):
        m = mdl.loc[n]
        p = {"phi_fixed": m.phi_fixed_mean, "sd_total": m.sd_total, "phi": m.phi_moving_anchor, "sigma_level": m.sigma_level_daily,
             "sd_u": m.sd_deviation_u, "sigma_ar": m.sd_deviation_u * np.sqrt(1 - m.phi_moving_anchor ** 2)}
        cost = cfg["strategy"]["cost_usd_per_bbl"] / kc.loc[n, "mean_price_P_B"]
        res = simulate(n, p, cost, seed=100 + i)
        for r in res:
            allsr[(n, r["world"])] = r.pop("sharpes")
            r["realised_daily_decisions"] = real[n]["sharpe"]
            r["realised_monthly_decisions"] = slow[n]["sharpe"]
            r["percentile_of_realised_in_this_world"] = float((allsr[(n, r["world"])] < real[n]["sharpe"]).mean())
            r["cost_log_per_change"] = cost
            rows.append(r)
        print(n, [(r["world"], round(r["mean_sharpe"], 2)) for r in res], flush=True)
    wt = pd.DataFrame(rows)
    wt.to_csv(OUT / "model_worlds.csv", index=False)

    # sensitivity of world C to the price level (cost per change = 0.05 / P_B)
    sens = []
    for P in (40.0, 120.0):
        for i, n in enumerate(LAB):
            m = mdl.loc[n]
            p = {"phi_fixed": m.phi_fixed_mean, "sd_total": m.sd_total, "phi": m.phi_moving_anchor, "sigma_level": m.sigma_level_daily,
                 "sd_u": m.sd_deviation_u, "sigma_ar": m.sd_deviation_u * np.sqrt(1 - m.phi_moving_anchor ** 2)}
            rng = np.random.default_rng(900 + i)
            l, _ = make_world(rng, p, "C", N_PATHS // 2)
            mean, sd, hl = rolling_fit(l)
            pnl, tr = run_rule(l, (l - mean) / sd, hl, gate=True, cost=cfg["strategy"]["cost_usd_per_bbl"] / P)
            sens.append({"spread": n, "price_level": P, "mean_sharpe_world_C": float(sharpe_paths(pnl).mean())})
    pd.DataFrame(sens).to_csv(OUT / "price_level_sensitivity.csv", index=False)

    # 3. deflated Sharpe of the trend and carry book
    tc = pd.read_csv(ROOT / "output/exploratory/trend_carry/tables/daily_books.csv", index_col=0, parse_dates=True)
    dm = lambda s: pd.read_csv(ROOT / "data/derived" / f"backtest_daily_fixed_{s}.csv", index_col=0, parse_dates=True)["pnl_usd"]
    mr = pd.concat([dm(s) for s in ["crack321", "brent_wti", "ho_rb"]], axis=1).mean(axis=1)
    pairs = pd.read_csv(ROOT / "output/exploratory/pairs/tables/daily_books.csv", index_col=0, parse_dates=True)["book"]
    rat = pd.read_csv(ROOT / "output/exploratory/ratio_mr/tables/daily_books.csv", index_col=0, parse_dates=True)["book"]
    slw = pd.read_csv(ROOT / "output/exploratory/ratio_slow/tables/daily_books.csv", index_col=0, parse_dates=True)["book"]
    trials = {"mean reversion (dollar)": mr, "pairs": pairs, "trend": tc["trend"], "carry": tc["carry"],
              "ratio mean reversion": rat, "slow ratio mean reversion": slw}
    d_sr = np.array([s.dropna().mean() / s.dropna().std() for s in trials.values()])
    tr_t = pd.DataFrame({"variant": list(trials), "annual_sharpe": d_sr * np.sqrt(252)})
    tr_t.to_csv(OUT / "six_variants.csv", index=False)
    dsr = {n_: deflated_sharpe(tc["combined"], d_sr, n_) for n_ in (6, 12, 24)}
    pd.DataFrame(dsr).T.to_csv(OUT / "deflated_sharpe.csv")

    summary = {"g1_power": pw.round(4).to_dict("records"), "worlds": wt.round(3).to_dict("records"), "deflated": dsr,
               "six_variants": tr_t.round(3).to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    plot(wt, pw)
    return summary


def plot(wt: pd.DataFrame, pw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    names = list(LAB)
    w = 0.2
    cols = {"A": "#9bb9e0", "B": SERIES[0], "C": SERIES[1], "D": SERIES[2]}
    labs = {"A": "A  fixed k, known", "B": "B  fixed k, estimated", "C": "C  moving k, estimated", "D": "D  moving k, known anchor"}
    for j, wd in enumerate("ABCD"):
        v = wt[wt.world == wd].set_index("spread").reindex(names)
        ax[0].bar(np.arange(3) + (j - 1.5) * w, v.mean_sharpe, w * 0.92, color=cols[wd], label=labs[wd])
    real = wt[wt.world == "C"].set_index("spread").reindex(names).realised_daily_decisions
    ax[0].scatter(np.arange(3), real, marker="D", color="#000", zorder=5, label="realised (daily decisions)")
    ax[0].axhline(0.5, color=REF_LINE, ls="--", lw=0.8)
    ax[0].text(2.45, 0.52, "G1 bar", fontsize=8, ha="right")
    ax[0].axhline(0, color=REF_LINE, lw=0.7)
    ax[0].set_xticks(np.arange(3), [LAB[n] for n in names])
    ax[0].set_ylabel("net Sharpe of the registered rule")
    ax[0].set_title("The rule on four model worlds")
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax[1].plot(pw.true_sharpe, pw.p_pass_G1, marker="o", color=SERIES[0])
    ax[1].axhline(0.5, color=REF_LINE, lw=0.7)
    ax[1].set_xlabel("true Sharpe")
    ax[1].set_ylabel("probability of passing G1 (> 0.5)")
    ax[1].set_title("G1 over 16.7 years")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "gate_power.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
