"""DIAGNOSTIC: did GARCH(1,1)-t actually absorb the volatility clustering?

Usage:  PYTHONPATH=src python scripts/run_garch_diagnostics.py [--check]
Requires the Bloomberg export. Writes output/exploratory/horizon_tests/tables/
garch_residual_diagnostics.csv and figures/garch_residual_diagnostics.png.

Step 7 of the deck (ARCH-LM) shows the daily changes cluster in volatility. That
justifies fitting GARCH, but it does not show the GARCH worked. The test that does
is on the STANDARDISED residuals z_t = (dx_t - mu) / sigma_t:

  * Ljung-Box on z    : is the direction still predictable?      (should be no)
  * Ljung-Box on z^2  : is the size still predictable?           (should now be no)
  * ARCH-LM on z      : same question, regression form

sigma_t is the one-day-ahead forecast made at t-1, exactly as the strategy uses it:
refit monthly on the prior 756 days, parameters frozen for the month (config.yaml
`estimation.window`, model.fit_garch / garch_filter). The comparison benchmark
standardises by the frozen window sd, the staircase the repo already uses in
vol_forecast_test. If GARCH works, z^2 should be unpredictable under GARCH and still
clustered under the staircase.

The estimator is a numpy/scipy GARCH(1,1) with unit-variance Student-t errors, because
`arch` is not installed in every environment. `--check` recovers known parameters from
a simulation. Fit rules mirror model.fit_garch: fall back to the window sd if the
optimiser fails or a + b >= 1.

Added 2026-11-02, after every registered and exploratory result was seen. Diagnostic
only: not a gate, changes no verdict.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import lfilter
from scipy.special import gammaln

from oilrv import futures
from oilrv.config import ROOT

OUT = ROOT / "output/exploratory/horizon_tests"
TAB, FIG = OUT / "tables", OUT / "figures"
WINDOW = 756
LAGS = 20
SERIES = {"WTI (CL)": "crack321__cl_bbl", "Brent (CO)": "brent_wti__brent_bbl",
          "RBOB (XB)": "crack321__rb_bbl", "ULSD (HO)": "crack321__ho_bbl",
          "3-2-1 crack": "crack321", "Brent - WTI": "brent_wti", "HO - RB": "ho_rb"}


# ------------------------------------------------------------------ estimator
def _variance(eps: np.ndarray, omega: float, a: float, b: float) -> np.ndarray:
    """sigma^2_t for t = 0..n-1, with sigma^2_0 = sample variance."""
    s0 = eps.var()
    x = omega + a * eps[:-1] ** 2
    y, _ = lfilter([1.0], [1.0, -b], x, zi=[b * s0])
    return np.concatenate([[s0], y])


def _nll(theta: np.ndarray, dx: np.ndarray) -> float:
    mu, omega, a, b, nu = theta
    eps = dx - mu
    s2 = _variance(eps, omega, a, b)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e12
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(eps ** 2 / (s2 * (nu - 2))))
    return -float(ll.sum())


def fit_garch_t(dx: np.ndarray) -> dict:
    """GARCH(1,1)-t by maximum likelihood. Same return shape as model.fit_garch."""
    dx = np.asarray(dx, float)
    sd = dx.std(ddof=1)
    z = dx / sd                                           # rescale for the optimiser
    fb = {"mu": float(dx.mean()), "omega": np.nan, "a": np.nan, "b": np.nan, "nu": np.nan,
          "sig2_next": float(dx.var(ddof=1)), "ok": False}
    best = None
    for a0, b0, nu0 in [(0.08, 0.90, 6.0), (0.15, 0.80, 4.0), (0.05, 0.93, 10.0)]:
        x0 = [z.mean(), (1 - a0 - b0), a0, b0, nu0]
        try:
            r = optimize.minimize(_nll, x0, args=(z,), method="L-BFGS-B",
                                  bounds=[(-1, 1), (1e-8, 5), (1e-4, 0.6),
                                          (0.0, 0.9999), (2.1, 100)])
        except Exception:                                  # noqa: BLE001
            continue
        if best is None or r.fun < best.fun:
            best = r
    if best is None or not best.success:
        return fb
    mu, omega, a, b, nu = best.x
    if not (a + b < 1):
        return fb
    eps = z - mu
    s2 = _variance(eps, omega, a, b)
    sig2_next = omega + a * eps[-1] ** 2 + b * s2[-1]
    return {"mu": float(mu * sd), "omega": float(omega * sd ** 2), "a": float(a),
            "b": float(b), "nu": float(nu), "sig2_next": float(sig2_next * sd ** 2),
            "ok": True}


def filter_sigma2(g: dict, dx_new: np.ndarray) -> np.ndarray:
    """Variance forecast FOR each day of dx_new, made at the previous close,
    parameters frozen. Element 0 is the seed sig2_next."""
    n = len(dx_new)
    if not g["ok"]:
        return np.full(n, g["sig2_next"])
    out = np.empty(n)
    s2 = g["sig2_next"]
    for i, x in enumerate(dx_new):
        out[i] = s2
        s2 = g["omega"] + g["a"] * (x - g["mu"]) ** 2 + g["b"] * s2
    return out


# ------------------------------------------------------------------ tests
def ljung_box(x: np.ndarray, lags: int = LAGS) -> tuple[float, float]:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    den = (x * x).sum()
    rho = np.array([(x[k:] * x[:-k]).sum() / den for k in range(1, lags + 1)])
    q = n * (n + 2) * (rho ** 2 / (n - np.arange(1, lags + 1))).sum()
    return float(q), float(stats.chi2.sf(q, lags))


def acf1(x: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    return float((x[1:] * x[:-1]).sum() / (x * x).sum())


def arch_lm(x: np.ndarray, lags: int = 5) -> tuple[float, float]:
    e2 = (np.asarray(x, float) - np.mean(x)) ** 2
    y = e2[lags:]
    X = np.column_stack([e2[lags - k: len(e2) - k] for k in range(1, lags + 1)]
                        + [np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    lm = len(y) * r2
    return float(lm), float(stats.chi2.sf(lm, lags))


# ------------------------------------------------------------------ pipeline
def rolling_standardise(dx: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Point-in-time standardised residuals under GARCH and under the frozen window sd."""
    idx = dx.index
    v = dx.to_numpy()
    month = pd.Series(idx.to_period("M"), index=idx)
    starts = np.flatnonzero((month != month.shift()).to_numpy())
    starts = starts[starts >= WINDOW]
    zg = np.full(len(v), np.nan)
    zw = np.full(len(v), np.nan)
    fitted = np.zeros(len(v), bool)                       # True on days run on a real GARCH fit
    ok = 0
    for j, i0 in enumerate(starts):
        i1 = starts[j + 1] if j + 1 < len(starts) else len(v)
        win = v[i0 - WINDOW: i0]
        g = fit_garch_t(win)
        ok += g["ok"]
        s2 = filter_sigma2(g, v[i0:i1])
        zg[i0:i1] = (v[i0:i1] - g["mu"]) / np.sqrt(s2)
        fitted[i0:i1] = g["ok"]
        zw[i0:i1] = (v[i0:i1] - win.mean()) / win.std(ddof=1)
    rolling_standardise.fitted = pd.Series(fitted, index=idx)
    return (pd.Series(zg, index=idx).dropna(), pd.Series(zw, index=idx).dropna(),
            ok / len(starts))


def build() -> dict[str, pd.Series]:
    fut = futures.build_futures()
    return {k: fut[c].dropna().diff().dropna() for k, c in SERIES.items()}


def check() -> None:
    """Recover known GARCH-t parameters from a simulation."""
    rng = np.random.default_rng(0)
    n, om, a, b, nu = 3000, 0.05, 0.08, 0.90, 6.0
    e = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    x = np.empty(n)
    s2 = om / (1 - a - b)
    for i in range(n):
        x[i] = np.sqrt(s2) * e[i]
        s2 = om + a * x[i] ** 2 + b * s2
    g = fit_garch_t(x)
    print(f"true  a={a} b={b} nu={nu}  a+b={a + b:.2f}")
    print(f"fit   a={g['a']:.3f} b={g['b']:.3f} nu={g['nu']:.1f} a+b={g['a'] + g['b']:.3f} ok={g['ok']}")
    z = (x - g["mu"]) / np.sqrt(_variance(x - g["mu"], g["omega"], g["a"], g["b"]))
    print("LB on z^2 after fit:", ljung_box(z ** 2), "(raw x^2:", ljung_box(x ** 2), ")")


def main() -> None:
    if "--check" in sys.argv:
        return check()
    if not futures.available():
        raise SystemExit("Bloomberg export not found")
    rows = []
    for name, dx in build().items():
        zg, zw, okrate = rolling_standardise(dx)
        for lab, z in (("GARCH(1,1)-t", zg), ("frozen window sd", zw)):
            q1, p1 = ljung_box(z.to_numpy())
            q2, p2 = ljung_box(z.to_numpy() ** 2)
            lm, plm = arch_lm(z.to_numpy())
            fit_only = z[rolling_standardise.fitted.reindex(z.index).fillna(False)] \
                if lab.startswith("GARCH") else z
            rows.append({"series": name, "standardised by": lab, "n": len(z),
                         "z^2 lag-1 ACF": acf1(z.to_numpy() ** 2),
                         "LB20 on z^2 p, fitted months only":
                             ljung_box(fit_only.to_numpy() ** 2)[1],
                         "z^2 lag-1 ACF, fitted only": acf1(fit_only.to_numpy() ** 2),
                         "fit_ok_rate": okrate if lab.startswith("GARCH") else np.nan,
                         "z sd": z.std(), "z kurtosis": stats.kurtosis(z) + 3,
                         "LB20 on z p": p1, "LB20 on z^2 p": p2,
                         "ARCH-LM5 on z p": plm})
        print(f"  {name}: done")
    df = pd.DataFrame(rows).set_index(["series", "standardised by"])
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "garch_residual_diagnostics.csv")
    print(df.round(4).to_string())
    figure(df)


def figure(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    names = list(SERIES)
    n = int(df["n"].iloc[0])
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    w = 0.27
    spec = (("frozen window sd", "z^2 lag-1 ACF", "#dc2626", "frozen window sd (benchmark)"),
            ("GARCH(1,1)-t", "z^2 lag-1 ACF", "#047857", "GARCH(1,1)-t, all months"),
            ("GARCH(1,1)-t", "z^2 lag-1 ACF, fitted only", "#86efac",
             "GARCH(1,1)-t, months with a real fit"))
    for j, (lab, col, c, leg) in enumerate(spec):
        ax.bar([i + (j - 1) * w for i in range(len(names))],
               [df.loc[(k, lab), col] for k in names], width=w, color=c, label=leg)
    b = 1.96 / np.sqrt(n)
    ax.axhspan(-b, b, color="#9ca3af", alpha=.3, lw=0)
    ax.text(-.45, -b - .004, "no-clustering band", ha="left", va="top", fontsize=8.5,
            color="#555")
    ax.axhline(0, color="#111", lw=.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("lag-1 autocorrelation of z²", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25, lw=.7)
    ax.set_axisbelow(True)
    fig.suptitle("GARCH removes most of the volatility clustering, not all of it",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(.5, .895, "Out-of-sample standardised residuals, monthly refit on 756 days. "
             "Shorter bar = less clustering left; inside the band = none detectable.",
             ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0, 1, .9])
    fig.savefig(FIG / "garch_residual_diagnostics.png", dpi=170, facecolor="white")
    print("  wrote figures/garch_residual_diagnostics.png")


if __name__ == "__main__":
    main()
