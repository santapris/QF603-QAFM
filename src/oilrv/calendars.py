"""Event calendar: OPEC/OPEC+ meetings, Gulf Coast hurricane landfalls, stock releases, Hormuz.

Sources
-------
- data/calendars/events_manual.csv, built by hand. Columns: date, type, name, scheduled, verified, source.
  `verified` is yes (checked against the linked source), partial, inferred (from the monthly
  first-Sunday pattern of the 2025-26 core-country meetings) or no (compiled from memory; check
  against the opec.org press-release archive, which blocks automated access).
- NHC HURDAT2 (Atlantic best tracks): hurricane landfalls on the US Gulf Coast west of 85W
  (Texas to the Florida Panhandle, where the refining and offshore capacity sits). Seasons not yet in
  HURDAT2 are added by hand in events_manual.csv.

STEO and IEA Oil Market Report dates are not included: the EIA and IEA archives refuse automated
access. They only feed the `event_within_3d` robustness feature, which is not gated.

Point in time
-------------
Only events scheduled in advance (OPEC meetings with scheduled=True) and hurricane landfalls drive the
blackout. Landfall dates are actual dates, because historical landfall forecasts are not available;
this is slightly optimistic and is stated in PREREGISTRATION.md section 9.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .config import load_config, path
from .data import _download, _raw

HURDAT2_FILE = "hurdat2-1851-2025-092326.txt"
HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/" + HURDAT2_FILE
GULF_BOX = {"lat": (25.0, 31.5), "lon": (-98.0, -85.0)}      # US Gulf Coast, Texas to Florida Panhandle
EVENT_COLS = ["date", "type", "name", "scheduled", "verified", "source"]


# --------------------------------------------------------------------------- hurricanes
def _coord(s: str) -> float:
    v = float(s[:-1])
    return -v if s[-1] in "SW" else v


def parse_hurdat2(text: str) -> pd.DataFrame:
    """Every track point as a row: storm_id, name, time (UTC), record, status, lat, lon, wind_kt."""
    rows, sid, name = [], None, None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if re.match(r"^AL\d{6}$", parts[0]):
            sid, name = parts[0], parts[1]
            continue
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        rows.append({"storm_id": sid, "name": name,
                     "time": pd.Timestamp(f"{parts[0]} {parts[1][:2]}:{parts[1][2:]}"),
                     "record": parts[2], "status": parts[3],
                     "lat": _coord(parts[4]), "lon": _coord(parts[5]), "wind_kt": int(parts[6])})
    return pd.DataFrame(rows)


def gulf_hurricane_landfalls(text: str, start_year: int = 2006) -> pd.DataFrame:
    """Hurricane-strength landfalls (record 'L', status 'HU') inside GULF_BOX, one row per storm and day."""
    t = parse_hurdat2(text)
    lf = t[(t["record"] == "L") & (t["status"] == "HU") & (t["time"].dt.year >= start_year)
           & t["lat"].between(*GULF_BOX["lat"]) & t["lon"].between(*GULF_BOX["lon"])].copy()
    lf["date"] = lf["time"].dt.normalize()
    lf = lf.drop_duplicates(["storm_id", "date"])
    names = [f"{n.title()} (landfall {la:.1f}N {-lo:.1f}W, {w} kt)"
             for n, la, lo, w in zip(lf["name"], lf["lat"], lf["lon"], lf["wind_kt"])]
    return pd.DataFrame({"date": lf["date"].to_numpy(), "type": "HURRICANE", "name": names,
                         "scheduled": False, "verified": "yes", "source": "NHC HURDAT2 " + HURDAT2_FILE})


def hurricane_events(refresh: bool = False) -> pd.DataFrame:
    f = _download(HURDAT2_URL, _raw("calendars") / HURDAT2_FILE, refresh, max_age_days=np.inf)
    return gulf_hurricane_landfalls(f.read_text())


# --------------------------------------------------------------------------- calendar
def build_events(refresh: bool = False, save: bool = True) -> pd.DataFrame:
    """Hand-built events plus HURDAT2 landfalls -> data/calendars/events.csv."""
    manual = pd.read_csv(path("calendar_dir") / "events_manual.csv", parse_dates=["date"])
    ev = pd.concat([manual, hurricane_events(refresh)], ignore_index=True)[EVENT_COLS]
    ev["scheduled"] = ev["scheduled"].astype(str).str.lower().eq("true")
    ev = ev.sort_values(["date", "type"]).drop_duplicates(["date", "type"]).reset_index(drop=True)
    if save:
        ev.to_csv(path("calendar_dir") / "events.csv", index=False)
    return ev


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(path("calendar_dir") / "events.csv", parse_dates=["date"])
    ev["scheduled"] = ev["scheduled"].astype(str).str.lower().eq("true")
    return ev


def extended_trading_days(index: pd.DatetimeIndex, horizon_days: int = 400) -> pd.DatetimeIndex:
    """The observed trading days plus US business days beyond the last observation.

    Needed so that an event just after the end of the data is still counted.
    """
    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    future = pd.date_range(index[-1] + bday, periods=horizon_days, freq=bday)
    return index.append(future)


def trading_days_to_next(index: pd.DatetimeIndex, events: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """For each date t, the number of trading days until the next event strictly after t.

    1 = the event is tomorrow. NaN if no event is known after t.
    Events falling on a non-trading day are mapped to the next trading day.
    """
    days = extended_trading_days(pd.DatetimeIndex(index))
    ev = pd.DatetimeIndex(sorted(pd.to_datetime(events)))
    ev_pos = np.searchsorted(days.values, ev.values, side="left")     # trading-day position of each event
    ev_pos = np.unique(ev_pos[ev_pos < len(days)])
    t_pos = np.arange(len(index))
    k = np.searchsorted(ev_pos, t_pos, side="right")                  # first event strictly after t
    out = np.full(len(index), np.nan)
    ok = k < len(ev_pos)
    out[ok] = ev_pos[k[ok]] - t_pos[ok]
    return pd.Series(out, index=index)


def calendar_days_since(index: pd.DatetimeIndex, events: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Calendar days since the latest event on or before t (NaN before the first event)."""
    ev = pd.DatetimeIndex(sorted(pd.to_datetime(events)))
    k = np.searchsorted(ev.values, pd.DatetimeIndex(index).values, side="right") - 1
    out = np.full(len(index), np.nan)
    ok = k >= 0
    out[ok] = (pd.DatetimeIndex(index)[ok] - ev[k[ok]]).days
    return pd.Series(out, index=index)


def event_features(index: pd.DatetimeIndex, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily calendar features, each known at the close of t.

    - days_to_opec / days_to_hurricane: trading days to the next scheduled OPEC meeting / landfall
    - event_blackout: no new entry decision whose fill (fill_lag_days later) would fall within
      strategy.event_blackout_days trading days before a blackout event
    - event_within_3d: a scheduled OPEC meeting or a landfall within the next 3 trading days
    - hurricane_season: June to November
    - days_since_hormuz_closure: calendar days since the latest Hormuz closure event
    """
    cfg = load_config()["strategy"]
    ev = events if events is not None else load_events()
    opec = ev.loc[(ev["type"] == "OPEC") & ev["scheduled"], "date"]
    hurr = ev.loc[ev["type"] == "HURRICANE", "date"]
    f = pd.DataFrame(index=index)
    f["days_to_opec"] = trading_days_to_next(index, opec)
    f["days_to_hurricane"] = trading_days_to_next(index, hurr)
    by_type = {"OPEC": f["days_to_opec"], "HURRICANE": f["days_to_hurricane"]}
    # "No new entry within n trading days before an event": the entry is the FILL, which happens
    # fill_lag_days after the decision, so a decision at t is blocked when the event is within
    # n + fill_lag_days trading days (code review 2026-10-24: the decision day alone was off by one).
    n = cfg["event_blackout_days"] + cfg["fill_lag_days"]
    f["event_blackout"] = np.logical_or.reduce([by_type[t] <= n for t in cfg["blackout_event_types"]])
    f["event_within_3d"] = (f["days_to_opec"] <= 3) | (f["days_to_hurricane"] <= 3)
    f["hurricane_season"] = pd.DatetimeIndex(index).month.isin(range(6, 12))
    closures = ev.loc[(ev["type"] == "HORMUZ") & ev["name"].str.startswith("closure"), "date"]
    f["days_since_hormuz_closure"] = calendar_days_since(index, closures)
    return f
