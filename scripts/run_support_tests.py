"""DESCRIPTIVE: direct tests of the trend and carry signals -> output/exploratory/horizon_support/tables/.

Usage:  .venv/bin/python scripts/run_support_tests.py     (about 1 min: it refits the GARCH sizing)
Requires build_data.py (futures). Added 2026-11-03, after every registered and exploratory result was seen.
Not a gate or a trading rule, so it changes no verdict.

1. variance_ratio.csv      Lo-MacKinlay robust variance ratio of 12-month against 1-month changes (month-end,
                           back-adjusted). Above 1: moves persist (trend). Below 1: moves partly reverse.
2. carry_forward_returns   Change in the held contract over the next 21 trading days regressed on the sign of the
                           curve slope (M1 - M2) and on its size (% of M1). HAC (21 lags) t-statistics.
3. passive_dynamic.csv     Each position split into its average (passive) and its deviation from the average
                           (dynamic). A large dynamic share means the signal is timing, not a disguised long.
4. oil_beta.csv            Daily sleeve P&L regressed on daily WTI and Brent changes. Alpha is the intercept.
Carry on RBOB and ULSD is outside the registered carry sleeve (seasonal curves); it is shown for comparison only.
The sleeves are rebuilt here, and the GARCH refit can differ slightly from the committed run, so sleeve Sharpes can
differ in the second decimal from output/exploratory/trend_carry/tables/summary_books.csv.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from oilrv import trend_carry as TC
from oilrv.config import ROOT, load_config, set_source

warnings.filterwarnings("ignore")
OUT = ROOT / "output" / "exploratory" / "horizon_support" / "tables"


def hac(y, X, lags):
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def variance_ratio(x: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay (1988) heteroskedasticity-robust variance ratio and two-sided p-value."""
    r = np.diff(x)
    n, mu = len(r), r.mean()
    s1 = ((r - mu) ** 2).sum() / n
    rq = np.convolve(r, np.ones(q), "valid")
    vr = ((rq - q * mu) ** 2).sum() / len(rq) / q / s1
    d = (r - mu) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * n * (d[j:] * d[:-j]).sum() / d.sum() ** 2 for j in range(1, q))
    z = (vr - 1) / np.sqrt(theta / n)
    return float(vr), float(2 * (1 - stats.norm.cdf(abs(z))))


def main() -> None:
    set_source("futures")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg4 = copy.deepcopy(cfg)
    cfg4["exploratory_trend_carry"]["carry"]["instruments"] = ["cl", "brent", "rb", "ho"]
    inst = TC.build_instruments(cfg=cfg4)
    sig = TC.sigmas(inst, cfg)
    base = TC.run_sleeves({k: v for k, v in inst.items()}, cfg, sig=sig)
    start = base["start"]

    # 1. variance ratios, 12 months against 1 month
    rows = []
    for n, f in inst.items():
        m = f["x"].astype(float).resample("ME").last().dropna()
        vr, p = variance_ratio(m.to_numpy(), 12)
        tr = np.sign(m - m.shift(12))
        pnl = (tr.shift(1) * m.diff()).dropna()
        rows.append({"series": n, "n_months": len(m), "vr_12m": vr, "p": p,
                     "trend_gross_sharpe_monthly": pnl.mean() / pnl.std() * np.sqrt(12)})
    pd.DataFrame(rows).to_csv(OUT / "variance_ratio.csv", index=False)

    # 2. carry against the next 21 trading days
    rows = []
    for n in ["cl", "brent", "rb", "ho"]:
        f = inst[n].loc[start:]
        slope = f["m1"] - f["m2"]
        fwd = f["x"].shift(-21) - f["x"]
        for lab, X in [("sign", np.sign(slope)), ("size", 100 * slope / f["m1"])]:
            r = hac(fwd, sm.add_constant(X.rename("c")), 21)
            rows.append({"instrument": n, "regressor": lab, "coef": r.params["c"], "t": r.tvalues["c"],
                         "n": int(r.nobs), "registered_carry": n in ("cl", "brent")})
    pd.DataFrame(rows).to_csv(OUT / "carry_forward_returns.csv", index=False)

    # 3. passive against dynamic position, carry sleeve (registered WTI, Brent; RBOB, ULSD for comparison)
    rows = []
    for n in ["cl", "brent", "rb", "ho"]:
        f = inst[n]
        s = TC.carry_signal(f, "m1_m2")
        if n not in sig:
            sig[n] = TC.sigmas({n: f}, cfg)[n]
        d = TC.simulate(f, s, sig[n], start, cfg)
        M, dx = float(f["M"].iloc[0]), d["x"].diff().fillna(0.0)
        mu = d["pos"].mean()
        dyn = (d["pos"] - mu) * M * dx
        t = hac(dyn, np.ones(len(dyn)), 10).tvalues.iloc[0]
        rows.append({"instrument": n, "registered_carry": n in ("cl", "brent"), "net_sharpe": TC.sharpe(d["pnl"]),
                     "mean_position_units": mu, "dynamic_share_of_gross": dyn.sum() / d["gross"].sum(),
                     "dynamic_t": t})
    pd.DataFrame(rows).to_csv(OUT / "passive_dynamic.csv", index=False)

    # 4. oil beta of the registered sleeves
    dx = pd.concat([inst["cl"]["x"].diff(), inst["brent"]["x"].diff()], axis=1, keys=["cl", "brent"])
    rows = []
    for k in ["trend", "carry", "combined"]:
        y = base[k]
        r = hac(y, sm.add_constant(dx.loc[y.index]), 10)
        rows.append({"book": k, "raw_sharpe": TC.sharpe(y), "alpha_usd_per_day": r.params["const"],
                     "alpha_t": r.tvalues["const"], "sharpe_from_alpha": r.params["const"] / y.std() * np.sqrt(252),
                     "beta_wti_usd_per_1bbl": r.params["cl"], "beta_brent_usd_per_1bbl": r.params["brent"]})
    pd.DataFrame(rows).to_csv(OUT / "oil_beta.csv", index=False)
    for f in ["variance_ratio", "carry_forward_returns", "passive_dynamic", "oil_beta"]:
        print(f"--- {f}\n", pd.read_csv(OUT / f"{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
