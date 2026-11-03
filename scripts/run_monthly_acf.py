"""DESCRIPTIVE: ACF/PACF, AR order and unit-root checks on MONTHLY changes of the three spreads
-> output/exploratory/horizon_support/tables/.

Usage:  .venv/bin/python scripts/run_monthly_acf.py
Requires build_data.py (futures). Series: back-adjusted 3-2-1 crack, Brent - WTI and HO - RB from
`trend_carry.build_instruments`. Month-end values (last close of each month) are the main construction;
monthly means are shown to illustrate the averaging effect (Working 1960: averaging a random walk gives
a lag-1 autocorrelation of changes of about 0.25).

Main window 2010-01 to 2026-09 (n = 200 changes, band +-1.96/sqrt(n) = 0.139), the registered trading window.
The full sample (2007-2026) is reported as a check. Not a gate or a trading rule; changes no verdict.
Helpers come from run_ar_acf_pacf.py (robust ACF/PACF, AIC/BIC on one common sample, unit-root tests).
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import run_ar_acf_pacf as R  # noqa: E402
from oilrv import diagnostics as dg  # noqa: E402
from oilrv import trend_carry as TC  # noqa: E402
from oilrv.config import ROOT, load_config, set_source  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "output" / "exploratory" / "horizon_support" / "tables"
SERIES = ["crack321", "brent_wti", "ho_rb"]
WINDOWS = {"2010-01 to 2026-09": ("2010-01", "2026-09"), "2007-01 to 2026-10": (None, None)}
K = 24


def monthly(x: pd.Series, how: str, lo, hi) -> pd.Series:
    x = x.astype(float)
    if lo:
        x = x.loc[lo:hi]
    r = x.resample("ME")
    return (r.last() if how == "month-end" else r.mean()).dropna()


def main() -> None:
    set_source("futures")
    OUT.mkdir(parents=True, exist_ok=True)
    inst = TC.build_instruments(cfg=load_config())
    acf_rows, ar_rows = [], []
    for n in SERIES:
        for wname, (lo, hi) in WINDOWS.items():
            for how in ["month-end", "month-mean"]:
                m = monthly(inst[n]["x"], how, lo, hi)
                d = m.diff().dropna()
                N, band = len(d), 1.96 / np.sqrt(len(d))
                a, p = R.acf_robust(d, K), R.pacf_robust(d, K)
                for lag in range(1, K + 1):
                    acf_rows.append({"series": n, "window": wname, "how": how, "n": N, "lag": lag,
                                     "acf": a.acf[lag], "pacf": p.pacf[lag], "band": band})
                ic = R.select_ar(d, 12)
                r1, _ = R.fit_ar(d, 1)
                ur = R.order_d(m, "c", 1)
                b, hl = dg.ar1(m.to_numpy())
                ar_rows.append({"series": n, "window": wname, "how": how, "n": N, "band": band,
                                "acf1": a.acf[1], "pacf1": p.pacf[1], "p_aic": int(ic.aic.idxmin()),
                                "p_bic": int(ic.bic.idxmin()), "ar1_phi": r1.params[1], "ar1_p": r1.pvalues[1],
                                "level_adf_p": ur.adf_p[0], "level_kpss_p": ur.kpss_p[0], "level_b": b,
                                "level_half_life_months": hl})
    pd.DataFrame(acf_rows).to_csv(OUT / "monthly_acf_pacf.csv", index=False)
    ar = pd.DataFrame(ar_rows)
    ar.to_csv(OUT / "monthly_ar.csv", index=False)

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2007-01-01", periods=5000)
    v = [R.acf_robust(pd.Series(rng.standard_normal(5000).cumsum(), idx).resample("ME").mean().diff().dropna(), 1).acf[1]
         for _ in range(200)]
    pd.DataFrame({"statistic": ["lag1_acf_of_monthly_mean_changes_random_walk"], "mean": [np.mean(v)],
                  "sd": [np.std(v)], "reps": [200]}).to_csv(OUT / "averaging_simulation.csv", index=False)

    show = ar[(ar.window == "2010-01 to 2026-09") & (ar.how == "month-end")]
    print(show[["series", "n", "band", "acf1", "pacf1", "p_aic", "p_bic", "ar1_phi", "ar1_p"]].round(3).to_string(index=False))
    print("random-walk monthly-mean lag-1 ACF:", round(float(np.mean(v)), 3))


if __name__ == "__main__":
    main()
