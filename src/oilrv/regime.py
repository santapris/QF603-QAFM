"""Step 07: regime filter (pooled point-in-time logit).

Episodes
--------
One row per signal: a day t when the point-in-time |z| crosses from below `rearm_z` into
[entry_z, stop_z). Episodes do not overlap; a new one can start only after the previous one has
resolved and |z| has come back below `rearm_z`.

Label (pre-registered): 1 if |z| <= label_revert_z within label_horizon_days trading days, before
first touching label_stop_z; otherwise 0. z along the episode uses the parameters in effect on the
signal day, frozen (weights, month effects, mean, sd), exactly as a trade would. The label is known
at the close of the resolution day (the hit day, or t + horizon).

Point-in-time probability
-------------------------
One logit pooled over the traded spreads (fixed variant), with spread dummies. At each month start
m it is refit on episodes resolved strictly before m; the daily P(t) for every day in that month is
that model applied to the day's features and the spread's dummy. With fewer than `min_episodes`
resolved episodes, or if the fit fails (separation, singular design), P is NaN and no filter applies.

Features (pre-registered core set), each known at the close of t:
- ovx_pct:          percentile of today's OVX within the trailing `percentile_lookback_days`
- garch_vol_pct:    percentile of today's GARCH sigma_next within its own trailing history
- rolling_adf_p:    ADF p-value of the deseasonalised spread from the current monthly fit
- gpr:              log(1 + daily GPR index), lagged one day
- inventory_dev_5y: US stocks vs their 5-year same-week average (%), stamped on the Thursday
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import load_config
from .diagnostics import month_starts
from .model import spread_from_legs

SPREAD_DUMMIES = {"brent_wti": "d_brent_wti", "ho_rb": "d_ho_rb"}      # crack321 is the base level
CFTC_LEGS = {"crack321": {"mm_net_rb": 1, "mm_net_ho": 1}, "brent_wti": {"mm_net_cl": -1},
             "ho_rb": {"mm_net_ho": 1, "mm_net_rb": -1}}


# --------------------------------------------------------------------------- features
def trailing_percentile(x: pd.Series, lookback: int, min_periods: int = 126) -> pd.Series:
    """Share of the trailing `lookback` observations (including today) that are <= today's value."""
    return x.rolling(lookback, min_periods=min_periods).apply(lambda a: (a <= a[-1]).mean(), raw=True)


def trailing_z(x: pd.Series, lookback: int, min_periods: int = 126) -> pd.Series:
    r = x.rolling(lookback, min_periods=min_periods)
    return (x - r.mean()) / r.std()


def regime_features(name: str, path: pd.DataFrame, feats: pd.DataFrame, cal: pd.DataFrame,
                    cfg: dict | None = None) -> pd.DataFrame:
    """Daily filter features for one spread, indexed like `path`. Core and robustness features."""
    cfg = cfg or load_config()
    lb = cfg["regime"]["percentile_lookback_days"]
    f = feats.reindex(path.index)
    out = pd.DataFrame(index=path.index)
    out["ovx_pct"] = trailing_percentile(feats["ovx"], lb).reindex(path.index)
    out["garch_vol_pct"] = trailing_percentile(path["sigma_next"], lb)
    out["rolling_adf_p"] = path["adf_p"]
    out["gpr"] = np.log1p(f["gpr"])           # log1p: one day (2025-02-10) has GPR = 0
    out["inventory_dev_5y"] = f["inventory_dev_5y"]
    mm = sum(w * feats[c] for c, w in CFTC_LEGS[name].items())
    out["cftc_mm_z"] = trailing_z(mm, lb).reindex(path.index)
    c = cal.reindex(path.index)
    out["hurricane_season"] = c["hurricane_season"].astype(float)
    out["event_within_3d"] = c["event_within_3d"].astype(float)
    out["brent_m1m6_slope_pct"] = (100 * f["brent_m1m6"] / f["brent_m1"]) if "brent_m1m6" in f else np.nan
    for s, d in SPREAD_DUMMIES.items():
        out[d] = float(name == s)
    return out


# --------------------------------------------------------------------------- episodes
def label_episode(z_fwd: np.ndarray, revert_z: float, stop_z: float) -> tuple[int, int, str]:
    """Scan |z| after the signal day. Returns (label, days_to_resolution, outcome)."""
    for k, v in enumerate(np.abs(z_fwd), start=1):
        if v <= revert_z:
            return 1, k, "reverted"
        if v >= stop_z:
            return 0, k, "stopped"
    return 0, len(z_fwd), "timed_out"


def frozen_z(legs: pd.DataFrame, fit: pd.Series, divisor: float) -> np.ndarray:
    """z of `legs` rows under one fit's frozen weights, month effects, mean and sd."""
    w = {k[2:]: fit[k] for k in fit.index if k.startswith("w_") and np.isfinite(fit[k])}
    S = spread_from_legs(legs, w, divisor).to_numpy()
    seas = np.array([fit[f"seas_{m}"] for m in legs.index.month])
    return (S - seas - fit["mean"]) / fit["sd"]


def build_episodes(name: str, sp: pd.DataFrame, fits: pd.DataFrame, path: pd.DataFrame,
                   horizon: int | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Signal episodes with frozen-parameter outcomes for one spread. Index = signal date.

    Episodes whose horizon runs past the end of the data and have not resolved are dropped
    (their label is not yet known).
    """
    cfg = cfg or load_config()
    st, rg = cfg["strategy"], cfg["regime"]
    H = horizon or rg["label_horizon_days"]
    divisor = cfg["spreads"][name]["divisor"]
    legs = sp[[c for c in sp.columns if c.endswith("_bbl")]]
    pos = legs.index.get_indexer(path.index)
    z = path["z"].to_numpy()
    rows, i, armed = [], 0, False
    while i < len(z):
        az = abs(z[i])
        if not np.isfinite(az):
            i += 1
            continue
        if az < st["rearm_z"]:
            armed = True
        elif armed and st["entry_z"] <= az < st["stop_z"]:
            fit = fits.loc[path["fit_date"].iloc[i]]
            a = pos[i]
            fwd = legs.iloc[a + 1:a + 1 + H]
            label, k, outcome = label_episode(frozen_z(fwd, fit, divisor), rg["label_revert_z"], rg["label_stop_z"])
            if outcome != "timed_out" or len(fwd) == H:
                rows.append({"date": path.index[i], "spread": name, "z": z[i], "side": -int(np.sign(z[i])),
                             "reverted": label, "outcome": outcome, "days": k,
                             "resolved_on": legs.index[a + k], "gate": bool(path["gate"].iloc[i]),
                             "half_life": path["half_life"].iloc[i]})
            armed = False
            i = int(np.searchsorted(pos, a + k))           # resume after resolution
            continue
        i += 1
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["spread", "z", "side", "reverted", "outcome", "days", "resolved_on", "gate", "half_life"])


# --------------------------------------------------------------------------- logit
def fit_model(train: pd.DataFrame, features: list[str], kind: str = "logit"):
    """Logit (or probit) of `reverted` on the features that vary in `train`. None if it cannot be fit."""
    d = train.dropna(subset=features + ["reverted"])
    use = [f for f in features if d[f].nunique() > 1]
    if not use or d["reverted"].nunique() < 2 or len(d) <= len(use) + 1:
        return None
    model = sm.Logit if kind == "logit" else sm.Probit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = model(d["reverted"].astype(float), sm.add_constant(d[use].astype(float))).fit(disp=0, maxiter=200)
    except Exception:                                       # noqa: BLE001 (singular / separation)
        return None
    if not np.all(np.isfinite(r.params)) or not r.mle_retvals.get("converged", True):
        return None
    return r


def predict(res, X: pd.DataFrame) -> np.ndarray:
    cols = [c for c in res.params.index if c != "const"]
    return np.asarray(res.predict(sm.add_constant(X[cols].astype(float), has_constant="add")))


def model_features(cfg: dict | None = None, extra: list[str] | None = None) -> list[str]:
    cfg = cfg or load_config()
    return list(cfg["regime"]["core_features"]) + list(extra or []) + list(SPREAD_DUMMIES.values())


def regime_probability(episodes: pd.DataFrame, daily: dict[str, pd.DataFrame], features: list[str],
                       min_episodes: int, kind: str = "logit") -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Daily point-in-time P(revert) per spread, from one pooled model refit at each month start on
    episodes resolved strictly before it.

    `episodes` must carry the feature columns (as of each signal date). `daily` maps spread -> daily
    feature frame. Returns ({spread: DataFrame[p_hat, n_train]}, coefficient history by month).
    """
    all_days = sorted(set().union(*[set(d.index) for d in daily.values()]))
    idx = pd.DatetimeIndex(all_days)
    starts = list(month_starts(idx)) + [len(idx)]
    out = {s: pd.DataFrame({"p_hat": np.nan, "n_train": 0}, index=d.index) for s, d in daily.items()}
    coefs = []
    for s0, s1 in zip(starts[:-1], starts[1:]):
        m, m_end = idx[s0], idx[s1 - 1]
        train = episodes[episodes["resolved_on"] < m]
        n = int(train.dropna(subset=features).shape[0])
        for s, d in daily.items():
            out[s].loc[m:m_end, "n_train"] = n
        if n < min_episodes:
            continue
        res = fit_model(train, features, kind)
        if res is None:
            continue
        coefs.append(pd.Series(res.params, name=m))
        for s, d in daily.items():
            seg = d.loc[m:m_end]
            if len(seg):
                ok = seg[[c for c in res.params.index if c != "const"]].notna().all(axis=1)
                p = np.full(len(seg), np.nan)
                if ok.any():
                    p[ok.to_numpy()] = predict(res, seg[ok])
                out[s].loc[m:m_end, "p_hat"] = p
    return out, pd.DataFrame(coefs)


# --------------------------------------------------------------------------- evaluation
def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def attach_p(episodes: pd.DataFrame, p_daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """P(revert) on each episode's signal date, from the daily series keyed by the episode's `key`
    ('variant|spread')."""
    ep = episodes.copy()
    ep["p_hat"] = [p_daily[k]["p_hat"].get(d, np.nan) if k in p_daily else np.nan
                   for d, k in zip(ep.index, ep["key"])]
    return ep


def evaluate_filter(episodes: pd.DataFrame, enter_p: float) -> dict:
    """Out-of-sample usefulness of the filter on episodes where a point-in-time P existed.

    The base-rate benchmark for the Brier score is the share reverted among episodes resolved before
    each signal (also point in time).
    """
    oos = episodes[episodes["p_hat"].notna()].copy()
    if not len(oos):
        return {"episodes_total": len(episodes), "episodes_with_p": 0}
    oos["base_rate"] = [episodes.loc[episodes["resolved_on"] < d, "reverted"].mean() for d in oos.index]
    take = oos["p_hat"] >= enter_p
    return {
        "episodes_total": len(episodes), "episodes_with_p": len(oos),
        "success_all_oos": float(oos["reverted"].mean()),
        "n_pass_filter": int(take.sum()),
        "success_pass_filter": float(oos.loc[take, "reverted"].mean()) if take.any() else np.nan,
        "success_rejected": float(oos.loc[~take, "reverted"].mean()) if (~take).any() else np.nan,
        "auc": auc(oos["reverted"], oos["p_hat"]),
        "brier_model": float(((oos["p_hat"] - oos["reverted"]) ** 2).mean()),
        "brier_base_rate": float(((oos["base_rate"] - oos["reverted"]) ** 2).mean()),
    }


def calibration_table(episodes: pd.DataFrame, bins=(0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)) -> pd.DataFrame:
    oos = episodes[episodes["p_hat"].notna()]
    b = pd.cut(oos["p_hat"], bins, include_lowest=True)
    return oos.groupby(b, observed=True).agg(n=("reverted", "size"), mean_p=("p_hat", "mean"),
                                            share_reverted=("reverted", "mean"))


# --------------------------------------------------------------------------- one call
def build_regime(sp: pd.DataFrame, feats: pd.DataFrame, cal: pd.DataFrame,
                 model: dict[str, tuple[pd.DataFrame, pd.DataFrame]], cfg: dict | None = None,
                 horizon: int | None = None, extra: list[str] | None = None, kind: str = "logit"):
    """Episodes and daily point-in-time P for every model key ('variant|spread' -> (fits, path)).

    Only fixed-variant episodes train the pooled model; every key gets a daily P from it.
    Returns (episodes with p_hat, daily features by key, daily P by key, coefficient history).
    """
    cfg = cfg or load_config()
    daily, eps = {}, []
    for key, (fits, path) in model.items():
        name = key.split("|")[1]
        daily[key] = regime_features(name, path, feats, cal, cfg)
        e = build_episodes(name, sp, fits, path, horizon, cfg)
        e["key"] = key
        eps.append(e.join(daily[key], how="left"))
    episodes = pd.concat(eps).sort_index()
    train = episodes[episodes["key"].str.startswith("fixed|")]
    p_daily, coefs = regime_probability(train, daily, model_features(cfg, extra), cfg["regime"]["min_episodes"], kind)
    return attach_p(episodes, p_daily), daily, p_daily, coefs
