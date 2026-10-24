"""Step 06: model formation.

`fit_window` estimates everything the strategy needs for one spread from one 756-day window.
`rolling_fits` repeats it at every month start; `model_path` carries the frozen parameters forward
day by day, giving the point-in-time spread S_t, z_t and the GARCH sigma forecast that a trader
would have had at each close.

Fair value (pre-registered): the spread minus its in-window month-of-year effect (seasonal spreads
only), then the window mean and sd give z:

    R_t = S_t - seas[month(t)],    z_t = (R_t - mu) / sd,    half-life from AR(1) on R.

Units: spreads and their daily changes in $/bbl; sigma in $/bbl per day.

Weights. The `fixed` variant trades the config weights. The `johansen` variant (crack only) uses
the first Johansen vector on (CL, RB, HO) from the same window, scaled to -3 CL; a window gives no
usable vector (gate closed) when the trace rank is 0, the vector is not finite, or any scaled weight
exceeds `JOHANSEN_MAX_WEIGHT` in size (the CL loading is then near zero and the scaling explodes).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from numpy.exceptions import ComplexWarning
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from .config import load_config
from .diagnostics import adf_p, ar1, month_starts, window_before

JOHANSEN_MAX_WEIGHT = 10.0
MONTHS = range(1, 13)


# --------------------------------------------------------------------------- weights
def spread_from_legs(legs: pd.DataFrame, weights: dict, divisor: float) -> pd.Series:
    """Spread in $/bbl from legs already in $/bbl (columns `<leg>_bbl`)."""
    return sum(w * legs[f"{k}_bbl"] for k, w in weights.items()) / divisor


def johansen_weights(legs_win: pd.DataFrame, order=("cl", "rb", "ho")) -> dict:
    """First Johansen vector on the window's (CL, RB, HO) levels, scaled to -3 CL.

    Returns {'weights': {leg: w} or None, 'jo_rank': int}.
    """
    x = legs_win[[f"{k}_bbl" for k in order]].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        j = coint_johansen(x, det_order=0, k_ar_diff=1)
    evec = np.real(j.evec)
    rank = 0
    for r in range(len(order)):
        if j.lr1[r] > j.cvt[r, 1]:
            rank = r + 1
        else:
            break
    v = evec[:, 0] * (-3.0 / evec[0, 0]) if evec[0, 0] != 0 else np.full(len(order), np.nan)
    ok = rank >= 1 and np.isfinite(v).all() and np.abs(v).max() <= JOHANSEN_MAX_WEIGHT
    return {"weights": dict(zip(order, v.tolist())) if ok else None, "jo_rank": rank}


# --------------------------------------------------------------------------- 6a. VECM
def fit_vecm_known(legs_win: pd.DataFrame, weights: dict, divisor: float) -> dict:
    """VECM with the cointegrating vector fixed at the traded weights (rank 1, one lag).

    For each leg k:  d leg_k,t = c_k + alpha_k * ect_{t-1} + sum_j g_kj * d leg_j,t-1 + e,
    with ect = the spread in $/bbl. With the vector known, equation-by-equation OLS is the VECM
    estimator. A leg with weight w_k corrects the spread when w_k * alpha_k < 0. `spread_speed` is
    the implied daily pull of the spread on itself, sum_k w_k * alpha_k / divisor.
    """
    lv = pd.DataFrame({k: legs_win[f"{k}_bbl"] for k in weights})
    ect = spread_from_legs(legs_win, weights, divisor)
    d = lv.diff()
    X = pd.concat([ect.shift(1).rename("ect"), d.shift(1).add_suffix("_lag")], axis=1)
    out, speed = {}, 0.0
    for k, w in weights.items():
        df = pd.concat([d[k].rename("y"), X], axis=1).dropna()
        r = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit()
        out[f"vecm_alpha_{k}"] = float(r.params["ect"])
        out[f"vecm_t_{k}"] = float(r.tvalues["ect"])
        speed += w * float(r.params["ect"])
    out["vecm_spread_speed"] = speed / divisor
    return out


def fit_vecm_free(legs: pd.DataFrame, order=("cl", "rb", "ho"), rank: int = 2) -> dict:
    """Unrestricted VECM(1) on the legs (statsmodels), for comparison with the fixed-vector one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VECM(legs[[f"{k}_bbl" for k in order]].to_numpy(), k_ar_diff=1, coint_rank=rank,
                   deterministic="ci").fit()
    return {"alpha": pd.DataFrame(res.alpha, index=list(order), columns=[f"ect{i + 1}" for i in range(rank)]),
            "alpha_t": pd.DataFrame(res.alpha / res.stderr_alpha, index=list(order),
                                    columns=[f"ect{i + 1}" for i in range(rank)]),
            "beta": pd.DataFrame(res.beta, index=list(order), columns=[f"ect{i + 1}" for i in range(rank)])}


# --------------------------------------------------------------------------- 6d. GARCH
def fit_garch(dx: np.ndarray) -> dict:
    """GARCH(1,1) with Student-t errors on daily spread changes ($/bbl).

    Returns the parameters plus the one-step variance forecast at the window end (`sig2_next`),
    which seeds the frozen-parameter filter. If the fit fails, does not converge, or is
    non-stationary (a + b >= 1), `ok` is False and sigma falls back to the sd of dx
    over the window.
    """
    dx = np.asarray(dx, dtype=float)
    fallback = {"mu": float(dx.mean()), "omega": np.nan, "a": np.nan, "b": np.nan, "nu": np.nan,
                "sig2_next": float(dx.var(ddof=1)), "ok": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = arch_model(dx, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    except Exception:                                   # noqa: BLE001  (any numerical failure -> fallback)
        return fallback
    p = r.params
    a, b = float(p["alpha[1]"]), float(p["beta[1]"])
    if r.convergence_flag != 0 or not (a + b < 1):
        return fallback
    sig2_next = p["omega"] + a * r.resid[-1] ** 2 + b * r.conditional_volatility[-1] ** 2
    return {"mu": float(p["mu"]), "omega": float(p["omega"]), "a": a, "b": b, "nu": float(p["nu"]),
            "sig2_next": float(sig2_next), "ok": True}


def garch_filter(g: dict, dx_new: np.ndarray) -> np.ndarray:
    """Carry a fitted GARCH forward over new data with the parameters held fixed.

    Element i is the forecast of sigma for the day AFTER dx_new[i], made at that day's close.
    If the fit had failed, the constant fallback sigma is returned.
    """
    dx_new = np.asarray(dx_new, dtype=float)
    out = np.empty(len(dx_new))
    if not g["ok"]:
        out[:] = np.sqrt(g["sig2_next"])
        return out
    s2 = g["sig2_next"]                                  # variance forecast for dx_new[0]
    for i, x in enumerate(dx_new):
        eps = x - g["mu"]
        s2 = g["omega"] + g["a"] * eps ** 2 + g["b"] * s2
        out[i] = np.sqrt(s2)
    return out


def _garch_dict(p: dict | pd.Series) -> dict:
    return {k[len("garch_"):]: p[k] for k in p.keys() if str(k).startswith("garch_")}


# --------------------------------------------------------------------------- one window
def seasonal_effects(s: pd.Series) -> dict[int, float]:
    """Month-of-year effect inside the window: mean of (s - window mean) by calendar month.
    Months absent from the window get 0."""
    dev = s - s.mean()
    m = dev.groupby(s.index.month).mean()
    return {k: float(m.get(k, 0.0)) for k in MONTHS}


def deseasonalise(s: pd.Series, seas: dict[int, float]) -> pd.Series:
    return s - s.index.month.map(seas).to_numpy(dtype=float)


def fit_window(legs_win: pd.DataFrame, name: str, weights: dict | None = None, cfg: dict | None = None) -> dict:
    """Everything the strategy needs for spread `name` from one window of leg prices ($/bbl).

    `weights` overrides the config legs (Johansen variant). Keys: weights (w_<leg>), seasonal effects
    (seas_1..seas_12), mean, sd, ar1_b, half_life, adf_p (all on the deseasonalised spread),
    dS_sd (window sd of daily spread changes), GARCH parameters, VECM loadings and the gate flag.
    """
    cfg = cfg or load_config()
    spec, m = cfg["spreads"][name], cfg["model"]
    w = weights or spec["legs"]
    s = spread_from_legs(legs_win, w, spec["divisor"])
    seas = seasonal_effects(s) if spec["seasonal_adjust"] else {k: 0.0 for k in MONTHS}
    r = deseasonalise(s, seas)
    b, hl = ar1(r)
    ds = np.diff(s.to_numpy())
    p = {f"w_{k}": float(v) for k, v in w.items()}
    p.update({f"seas_{k}": v for k, v in seas.items()})
    p.update({"mean": float(r.mean()), "sd": float(r.std(ddof=1)), "ar1_b": b, "half_life": hl,
              "adf_p": adf_p(r), "dS_sd": float(ds.std(ddof=1)), "S_last": float(s.iloc[-1])})
    p.update({f"garch_{k}": v for k, v in fit_garch(ds).items()})
    p.update(fit_vecm_known(legs_win, w, spec["divisor"]))
    p["hl_ok"] = bool(m["half_life_min"] <= hl <= m["half_life_max"])
    p["gate"] = p["hl_ok"]                                   # strategy.entry_gate: [half_life_in_band]
    return p


# --------------------------------------------------------------------------- point-in-time path
def legs_frame(sp: pd.DataFrame) -> pd.DataFrame:
    return sp[[c for c in sp.columns if c.endswith("_bbl")]]


def rolling_fits(sp: pd.DataFrame, name: str, variant: str = "fixed", window: int | None = None,
                 cfg: dict | None = None) -> pd.DataFrame:
    """fit_window at every month start on the `window` days strictly before it. Indexed by month start.

    For the johansen variant, a month with no usable vector gets weights NaN and gate False.
    """
    cfg = cfg or load_config()
    window = window or cfg["model"]["window"]
    legs = legs_frame(sp)
    rows = [{"date": legs.index[s], **fit_at(legs, s, name, variant, window, cfg)}
            for s in month_starts(legs.index) if s >= window]
    return pd.DataFrame(rows).set_index("date")


def fit_at(legs: pd.DataFrame, pos: int, name: str, variant: str = "fixed", window: int | None = None,
           cfg: dict | None = None) -> dict:
    """The fit a trader would make at the close before row `pos`, on the `window` rows strictly before it.

    Used for the scheduled month-start fits and for refits right after an exit (backtest).
    """
    cfg = cfg or load_config()
    window = window or cfg["model"]["window"]
    win = window_before(legs, pos, window)
    if variant == "johansen":
        jw = johansen_weights(win)
        extra = {"jo_rank": jw["jo_rank"], "weights_ok": jw["weights"] is not None}
        if jw["weights"] is None:
            return {"pos": pos, "gate": False, "hl_ok": False, **extra}
        return {"pos": pos, **fit_window(win, name, jw["weights"], cfg), **extra}
    return {"pos": pos, **fit_window(win, name, None, cfg)}


def zscore(S: float | np.ndarray, month: int | np.ndarray, p: dict | pd.Series):
    """z of spread value(s) S on calendar month(s) under frozen parameters p."""
    seas = np.vectorize(lambda m: p[f"seas_{m}"])(month)
    return (S - seas - p["mean"]) / p["sd"]


def model_path(sp: pd.DataFrame, fits: pd.DataFrame, name: str, cfg: dict | None = None) -> pd.DataFrame:
    """Daily point-in-time model output for one spread, using the fit made at the start of each month.

    For each day t: S (spread under the month's frozen weights), dS = S_t - S_{t-1} (same weights),
    z, sigma_next = GARCH sigma forecast for t+1 made at close t, plus the fit's parameters and gate.
    Every value at t uses only data up to t and parameters fitted on data before the month start.
    """
    cfg = cfg or load_config()
    divisor = cfg["spreads"][name]["divisor"]
    legs = legs_frame(sp)
    starts = list(fits["pos"].astype(int))
    ends = starts[1:] + [len(legs)]
    parts = []
    for (fit_date, p), s, e in zip(fits.iterrows(), starts, ends):
        idx = legs.index[s:e]
        if not p["gate"] and not np.isfinite(p.get("sd", np.nan)):
            parts.append(pd.DataFrame({"fit_date": fit_date, "gate": False}, index=idx))
            continue
        w = {k[2:]: p[k] for k in p.index if k.startswith("w_") and np.isfinite(p[k])}
        S = spread_from_legs(legs.iloc[s - 1:e], w, divisor).to_numpy()
        dS = np.diff(S)
        parts.append(pd.DataFrame({
            "S": S[1:], "dS": dS, "z": zscore(S[1:], idx.month.to_numpy(), p),
            "sigma_next": garch_filter(_garch_dict(p), dS), "fit_date": fit_date,
            "mean": p["mean"], "sd": p["sd"], "seas": idx.month.map(lambda m: p[f"seas_{m}"]).to_numpy(dtype=float),
            "dS_sd": p["dS_sd"], "half_life": p["half_life"], "adf_p": p["adf_p"], "gate": bool(p["gate"]),
            **{f"w_{k}": v for k, v in w.items()},
        }, index=idx))
    out = pd.concat(parts)
    out["gate"] = out["gate"].fillna(False).astype(bool)
    return out


# --------------------------------------------------------------------------- volatility forecast check
def qlike(sigma: pd.Series, x: pd.Series) -> pd.Series:
    """QLIKE loss of a variance forecast (lower is better; robust to noisy proxies)."""
    s2 = sigma ** 2
    return np.log(s2) + x ** 2 / s2


def vol_forecast_test(path: pd.DataFrame, hac_lags: int = 10) -> dict:
    """Out-of-sample check that GARCH sigma beats the constant window sd of dS.

    Pairs the forecast made at close t (`sigma_next`) with the realised dS on t+1 and
    compares mean QLIKE loss. diff < 0 means GARCH is better; the t-stat uses HAC errors.
    Days that cross a re-fit (next dS is under new parameters) are dropped.
    """
    nxt = path["dS"].shift(-1)
    same_fit = path["fit_date"].shift(-1) == path["fit_date"]
    ok = nxt.notna() & same_fit & path["sigma_next"].notna()
    lg = qlike(path.loc[ok, "sigma_next"], nxt[ok])
    lc = qlike(path.loc[ok, "dS_sd"], nxt[ok])
    d = (lg - lc).to_numpy()
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {"qlike_garch": float(lg.mean()), "qlike_const": float(lc.mean()),
            "mean_diff": float(r.params[0]), "t_stat": float(r.tvalues[0]), "p": float(r.pvalues[0]),
            "n": int(ok.sum())}


def build_model(sp: pd.DataFrame, cfg: dict | None = None, window: int | None = None) -> dict:
    """Rolling fits and daily paths for every traded spread (fixed) plus the crack's johansen variant.

    Returns {(variant, spread): (fits, path)}.
    """
    cfg = cfg or load_config()
    out = {}
    for variant, vspec in cfg["variants"].items():
        for name in vspec["spreads"]:
            fits = rolling_fits(sp, name, vspec["weights"] if vspec["weights"] == "fixed" else "johansen",
                                window, cfg)
            out[(variant, name)] = (fits, model_path(sp, fits, name, cfg))
    return out

