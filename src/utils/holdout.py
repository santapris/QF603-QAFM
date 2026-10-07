"""Holdout guard and look-ahead-safe training windows [F2][F3].

A row dated t that is used to fit or evaluate a model for target y at horizon k needs y_{t+k}.
What matters for the holdout and for training windows is **when that target becomes known**:

* exact: a per-row observation date supplied by the caller (``obs_dates``), or
* month-granular fallback: month(t) + k + ``target_observation_lag_months[target]`` from config.

Rules
-----
* ``guard``: without ``final=True``, raise if any row's target is observed after ``oos.validate_end``.
  Even with ``final=True``, raise if any target is observed after ``oos.holdout_end``.
* ``training_rows``: at forecast origin t keep only rows whose target is observed by t.
"""

from __future__ import annotations

import pandas as pd

from src.utils.dates import add_months
from src.utils.io import load_config


class HoldoutViolation(RuntimeError):
    """Raised when modelling code would touch data it is not allowed to see."""


def _row_dates(df: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if isinstance(df, pd.DataFrame) and "date" in df.columns:
        return pd.DatetimeIndex(df["date"])
    raise TypeError("df needs a DatetimeIndex or a 'date' column")


def _lag_months(target: str | None, cfg: dict) -> int:
    if target is None:
        return 0
    lags = cfg["target_observation_lag_months"]
    if target not in lags:
        raise ValueError(f"unknown target {target!r}; add it to target_observation_lag_months "
                         f"in config.yaml (known: {sorted(lags)})")
    return int(lags[target])


def target_observation_period(df, target: str | None = None, k: int = 0,
                              cfg: dict | None = None) -> pd.PeriodIndex:
    """Month in which each row's target y_{t+k} becomes known (month-granular fallback)."""
    cfg = cfg or load_config()
    return add_months(_row_dates(df), k + _lag_months(target, cfg))


def _observed_after(df, cutoff: pd.Timestamp, target, k, cfg, obs_dates) -> pd.Series:
    """Boolean mask: target observed strictly after ``cutoff``."""
    if obs_dates is not None:
        obs = pd.Series(pd.DatetimeIndex(obs_dates), index=df.index)
        if obs.isna().any():
            raise HoldoutViolation("obs_dates contains missing values; cannot verify the holdout")
        return obs > cutoff
    obs = target_observation_period(df, target, k, cfg)
    return pd.Series(obs > cutoff.to_period("M"), index=df.index)


def guard(df, target: str | None = None, k: int = 0, final: bool = False,
          cfg: dict | None = None, obs_dates=None):
    """Return ``df`` unchanged if it respects the holdout; otherwise raise HoldoutViolation.

    Parameters
    ----------
    df : DataFrame/Series indexed by forecast origin t (or with a 'date' column).
    target : name of the dependent variable (key of ``target_observation_lag_months``); None means
        the rows themselves are the observations (lag 0, k ignored unless given).
    k : forecast horizon in months.
    final : only True in the single Phase 7 holdout run.
    obs_dates : optional exact date at which each row's target is observed (overrides month rule).
    """
    cfg = cfg or load_config()
    end_key = "holdout_end" if final else "validate_end"
    cutoff = pd.Timestamp(cfg["oos"][end_key])
    bad = _observed_after(df, cutoff, target, k, cfg, obs_dates)
    if bad.any():
        first = _row_dates(df)[bad.to_numpy()].min()
        raise HoldoutViolation(
            f"{int(bad.sum())} row(s) have target {target or '(row)'} at k={k} observed after "
            f"oos.{end_key}={cutoff.date()} (first origin {first.date()}). "
            + ("Trim with holdout.restrict() or pass final=True only in the Phase 7 run."
               if not final else "No target may rely on data after the sample end."))
    return df


def restrict(df, target: str | None = None, k: int = 0, final: bool = False,
             cfg: dict | None = None, obs_dates=None):
    """Drop rows whose target is observed after the allowed cutoff (then ``guard`` passes)."""
    cfg = cfg or load_config()
    cutoff = pd.Timestamp(cfg["oos"]["holdout_end" if final else "validate_end"])
    bad = _observed_after(df, cutoff, target, k, cfg, obs_dates)
    return df.loc[~bad.to_numpy()]


def training_rows(panel, origin, target: str | None = None, k: int = 0,
                  cfg: dict | None = None, obs_dates=None):
    """Rows usable for fitting at forecast origin ``origin``: target observed on or before it."""
    cfg = cfg or load_config()
    origin = pd.Timestamp(origin)
    if obs_dates is not None:
        obs = pd.Series(pd.DatetimeIndex(obs_dates), index=panel.index)
        mask = (obs <= origin).to_numpy()
    else:
        obs = target_observation_period(panel, target, k, cfg)
        mask = obs <= origin.to_period("M")
    mask &= _row_dates(panel) <= origin
    return panel.loc[mask]
