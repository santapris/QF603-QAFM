"""Bloomberg futures curves -> continuous, same-delivery-month leg series for each spread.

Input: the Bloomberg workbook `futures.export_file` (one sheet per root: CL, CO, XB, HO), or else
data/raw/bloomberg/<ROOT>.csv. Each sheet has two header rows (ticker, field) and a date column, with
PX_SETTLE and FUT_CUR_GEN_TICKER for every generic (see data/raw/bloomberg/README.md).
FUT_CUR_GEN_TICKER names the actual contract behind each generic on each day, so the generic roll
itself is never used for prices. Bloomberg's ' .NA.' (no contract that day) counts as missing, and
settles are divided by `futures.price_divisor` (XB and HO come in cents/gal).

Contract identity. Bloomberg's live tickers use a one-digit year (CLZ6), which is ambiguous across
decades, so the year is resolved from the observation date: the first year >= obs_year - 1 that
ends in that digit. Two-digit years (CLZ16) are taken as given.

Expiries come from the exchange rules (expiry()), so no BDP pull is needed. An optional
expiries.csv (columns: contract like CLZ2016, last_tradeable_dt) overrides the rule.

Rolling. For a spread, all legs hold the SAME delivery month. On day t the held month is the k-th
nearest month that every leg quotes and whose roll date is after t, where
roll date = min over legs of (expiry - roll_bdays business days). This makes Brent and WTI match by
delivery month (Brent expires about a month earlier, so it sets the roll date).

Prices. For each leg:
- `<leg>`      unadjusted price of the held contract (the actual market level)
- `<leg>_adj`  additive back-adjusted series: its daily change is exactly the change in the contract
               held over (t-1, t], and its last value equals the unadjusted price
- `rolled`     True on the close where the position rolls into the next month

Common dates. `build_futures` keeps only the dates on which every leg of every spread settles (as the
spot data keeps only days with all four legs), so all spreads share one index. Dropping a date loses
no P&L: the adjusted series is cumulative, so the change over a gap is the sum of the daily changes.
A roll on a dropped date is carried to the next kept date.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay

from .config import ROOT, load_config, path
from .spreads import GALLON_LEGS, spread_value, to_bbl, traded_spreads

MONTH_CODES = "FGHJKMNQUVXZ"
_TICKER_RE = re.compile(r"^([A-Z]{2})([FGHJKMNQUVXZ])(\d{1,2})(?:\s+COMDTY)?$")
_NA_RE = re.compile(r"^(\.?#?N/?A\.?(\s+N/?A)?|NAN|NONE|)$")
US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


# --------------------------------------------------------------------------- contracts
def parse_contract(ticker: str, obs_date: pd.Timestamp) -> tuple[str, pd.Period]:
    """'CLZ6' seen on 2016-11-01 -> ('CL', Period('2016-12')). Raises on an unknown format."""
    m = _TICKER_RE.match(str(ticker).strip().upper())
    if not m:
        raise ValueError(f"Unrecognised contract ticker: {ticker!r}")
    root, code, yr = m.groups()
    month = MONTH_CODES.index(code) + 1
    if len(yr) == 2:
        year = 2000 + int(yr)
    else:
        year = obs_date.year - 1
        while year % 10 != int(yr):
            year += 1
    return root, pd.Period(year=year, month=month, freq="M")


def expiry(root: str, delivery: pd.Period) -> pd.Timestamp:
    """Last trading day from the exchange rules (US federal holidays approximate exchange holidays).

    - CL (NYMEX WTI): 3 business days before the 25th of the month before delivery; if the 25th is
      not a business day, 3 business days before the last business day preceding it.
    - XB, HO (NYMEX RBOB, ULSD): last business day of the month before delivery.
    - CO (ICE Brent): last business day of the second month before delivery (contracts from Mar 2016);
      before that, the business day before the 15th calendar day before the first day of delivery.
    """
    first = delivery.to_timestamp()
    if root == "CL":
        d25 = (first - pd.offsets.MonthBegin(1)) + pd.Timedelta(days=24)
        anchor = d25 if US_BDAY.is_on_offset(d25) else d25 - US_BDAY
        return anchor - 3 * US_BDAY
    if root in ("XB", "HO"):
        return first - US_BDAY
    if root == "CO":
        if delivery >= pd.Period("2016-03", "M"):
            return (first - pd.offsets.MonthBegin(1)) - BDay(1)
        return (first - pd.Timedelta(days=15)) - BDay(1)
    raise ValueError(f"No expiry rule for root {root!r}")


def load_expiry_overrides() -> dict[tuple[str, pd.Period], pd.Timestamp]:
    f = path("raw_dir") / "bloomberg" / "expiries.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f, parse_dates=["last_tradeable_dt"])
    out = {}
    for c, d in zip(df["contract"], df["last_tradeable_dt"]):
        m = re.match(r"^([A-Z]{2})([FGHJKMNQUVXZ])(\d{4})$", str(c).strip().upper())
        if m:
            out[(m.group(1), pd.Period(year=int(m.group(3)), month=MONTH_CODES.index(m.group(2)) + 1,
                                       freq="M"))] = d
    return out


# --------------------------------------------------------------------------- loading
def read_bloomberg(f: Path | str, sheet: str | None = None) -> pd.DataFrame:
    """Two-header-row BDH export (CSV, or one sheet of a workbook) -> long table (date, generic, ticker, price)."""
    f = Path(f)
    if f.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(f, sheet_name=sheet, header=[0, 1], index_col=0)
    else:
        raw = pd.read_csv(f, header=[0, 1], index_col=0)
    raw = raw[pd.notna(raw.index)]
    raw.index = pd.to_datetime(raw.index)
    raw.columns = pd.MultiIndex.from_tuples([(str(a).strip(), str(b).strip().upper()) for a, b in raw.columns])
    rows = []
    for generic in raw.columns.get_level_values(0).unique():
        if (generic, "PX_SETTLE") not in raw or (generic, "FUT_CUR_GEN_TICKER") not in raw:
            continue
        tick = raw[(generic, "FUT_CUR_GEN_TICKER")].astype("string").str.strip()
        tick = tick.mask(tick.str.upper().str.fullmatch(_NA_RE).fillna(True))
        d = pd.DataFrame({"price": pd.to_numeric(raw[(generic, "PX_SETTLE")], errors="coerce"),
                          "ticker": tick})
        d["generic"] = generic
        rows.append(d.dropna(subset=["price", "ticker"]).rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True)


def contract_panel(long: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Long table -> wide panel (date x delivery month) of settles, plus the root."""
    parsed = [parse_contract(t, d) for t, d in zip(long["ticker"], long["date"])]
    roots = {r for r, _ in parsed}
    if len(roots) != 1:
        raise ValueError(f"Mixed roots in one file: {roots}")
    df = long.assign(delivery=[p for _, p in parsed])
    panel = (df.drop_duplicates(["date", "delivery"]).pivot(index="date", columns="delivery", values="price")
             .sort_index())
    return panel, roots.pop()


# --------------------------------------------------------------------------- rolling
def held_months(panels: dict[str, pd.DataFrame], roots: dict[str, str], k: int = 1,
                roll_bdays: int = 5, overrides: dict | None = None) -> pd.Series:
    """Delivery month held on each date: the k-th nearest month quoted by every leg whose
    joint roll date (min over legs of expiry - roll_bdays business days) is after the date."""
    overrides = overrides or {}
    legs = list(panels)
    idx = panels[legs[0]].index
    for leg in legs[1:]:
        idx = idx.intersection(panels[leg].index)
    months = sorted(set.intersection(*[set(panels[leg].columns) for leg in legs]))
    roll = {m: min(overrides.get((roots[leg], m), expiry(roots[leg], m)) for leg in legs) - roll_bdays * US_BDAY
            for m in months}
    quoted = np.logical_and.reduce([panels[leg].reindex(index=idx, columns=months).notna().to_numpy()
                                    for leg in legs])
    roll_arr = np.array([roll[m] for m in months], dtype="datetime64[ns]")
    out = []
    for i, t in enumerate(idx.values):
        ok = np.flatnonzero(quoted[i] & (roll_arr > t))
        out.append(months[ok[k - 1]] if len(ok) >= k else None)
    return pd.Series(out, index=idx, name="delivery")


def continuous(panels: dict[str, pd.DataFrame], roots: dict[str, str], k: int = 1, roll_bdays: int = 5,
               overrides: dict | None = None) -> pd.DataFrame:
    """Unadjusted and additive back-adjusted series per leg, all on the same held delivery month."""
    h = held_months(panels, roots, k, roll_bdays, overrides).dropna()
    idx = h.index
    out = pd.DataFrame(index=idx)
    out["delivery"] = h.astype(str)
    out["rolled"] = h.ne(h.shift()) & h.shift().notna()
    out["stale_leg"] = False
    for leg, p in panels.items():
        p = p.reindex(idx)
        cur = np.array([p.at[t, m] for t, m in zip(idx, h)])
        prev_m = h.shift()
        # change on day t in the contract held over (t-1, t]
        held_t = np.array([p.at[t, m] if isinstance(m, pd.Period) else np.nan
                           for t, m in zip(idx, prev_m)])
        held_tm1 = np.r_[np.nan, cur[:-1]]
        ret = held_t - held_tm1
        missing = np.isnan(ret)
        missing[0] = False
        out.loc[missing, "stale_leg"] = True
        ret = np.nan_to_num(ret, nan=0.0)
        ret[0] = 0.0
        out[leg] = cur
        # last value equals cur[-1]; each earlier value subtracts all later held-contract changes
        out[f"{leg}_adj"] = cur[-1] - np.r_[ret[1:][::-1].cumsum()[::-1], 0.0]
    return out


# --------------------------------------------------------------------------- spreads
def leg_roots(cfg: dict | None = None) -> dict[str, str]:
    return {leg: g[0] for leg, g in (cfg or load_config())["futures"]["generics"].items()}


def export_file(cfg: dict | None = None) -> Path | None:
    f = (cfg or load_config())["futures"].get("export_file")
    return ROOT / f if f else None


def available(cfg: dict | None = None) -> bool:
    f, d = export_file(cfg), path("raw_dir") / "bloomberg"
    return bool(f and f.exists()) or all((d / f"{root}.csv").exists() for root in leg_roots(cfg).values())


def load_panels(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """{leg: date x delivery-month panel of settles}, in $/bbl (CL, CO) or $/gal (XB, HO)."""
    cfg = cfg or load_config()
    f, d = export_file(cfg), path("raw_dir") / "bloomberg"
    div = cfg["futures"].get("price_divisor", {})
    out = {}
    for leg, root in leg_roots(cfg).items():
        src = (f, root) if f and f.exists() else (d / f"{root}.csv", None)
        panel, r = contract_panel(read_bloomberg(*src))
        if r != root:
            raise ValueError(f"{src} holds {r} contracts, expected {root}")
        panel = panel / div.get(root, 1)
        if leg in GALLON_LEGS and panel.abs().max().max() > 20:
            raise ValueError(f"{root} settles look like cents/gal; set futures.price_divisor.{root}: 100")
        out[leg] = panel
    return out


def _to_common(flags: pd.Series, common: pd.DatetimeIndex) -> pd.Series:
    """Carry True flags on dates outside `common` to the next date in `common`."""
    pos = common.searchsorted(flags.index)
    keep = pos < len(common)
    return (pd.Series(flags.to_numpy()[keep], index=common[pos[keep]]).groupby(level=0).any()
            .reindex(common, fill_value=False))


def build_futures(panels: dict[str, pd.DataFrame] | None = None, cfg: dict | None = None,
                  save: bool = True) -> pd.DataFrame:
    """One wide table on the common dates (every leg of every traded spread settles). For each spread:
    its value from adjusted legs (`<name>`, for signals and P&L) and from unadjusted legs
    (`<name>_unadj`, the market level), its legs in $/bbl (`<name>__<leg>_bbl` adjusted,
    `<name>__<leg>_bbl_unadj`), and `rolled_<name>`, `stale_<name>`, `delivery_<name>`;
    plus `brent_m1m6` (Brent's own roll, unadjusted M1 - M6) and `brent_m1`."""
    cfg = cfg or load_config()
    panels = panels or load_panels(cfg)
    roots = leg_roots(cfg)
    rb = cfg["futures"]["roll_bdays_before_expiry"]
    ov = load_expiry_overrides()
    cont = {name: continuous({l: panels[l] for l in spec["legs"]}, {l: roots[l] for l in spec["legs"]}, 1, rb, ov)
            for name, spec in traded_spreads(cfg).items()}
    common = None
    for c in cont.values():
        common = c.index if common is None else common.intersection(c.index)
    common = common[(common >= pd.Timestamp(cfg["data"]["sample_start"]))
                    & (common <= pd.Timestamp(cfg["data"]["sample_end"] or common.max()))]
    cols = {}
    for name, spec in traded_spreads(cfg).items():
        legs, c = list(spec["legs"]), cont[name]
        adj = to_bbl(c[[f"{l}_adj" for l in legs]].rename(columns=lambda s: s[:-4]), cfg)
        raw = to_bbl(c[legs], cfg)
        cols[name] = spread_value(adj, spec).reindex(common)
        cols[f"{name}_unadj"] = spread_value(raw, spec).reindex(common)
        for l in legs:
            cols[f"{name}__{l}_bbl"] = adj[l].reindex(common)
            cols[f"{name}__{l}_bbl_unadj"] = raw[l].reindex(common)
        cols[f"rolled_{name}"] = _to_common(c["rolled"], common)
        cols[f"stale_{name}"] = _to_common(c["stale_leg"], common)
        cols[f"delivery_{name}"] = c["delivery"].reindex(common)
    b1 = continuous({"brent": panels["brent"]}, {"brent": roots["brent"]}, 1, rb, ov)
    b6 = continuous({"brent": panels["brent"]}, {"brent": roots["brent"]}, 6, rb, ov)
    cols["brent_m1m6"] = (b1["brent"] - b6["brent"]).reindex(common)
    cols["brent_m1"] = b1["brent"].reindex(common)
    out = pd.DataFrame(cols, index=common)
    out.index.name = "date"
    if save:
        out.to_csv(path("clean_dir") / "futures_spreads.csv")
    return out


def load_futures() -> pd.DataFrame:
    return pd.read_csv(path("clean_dir") / "futures_spreads.csv", index_col=0, parse_dates=True)


def spread_frames(fut: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """{spread: frame} with that spread's own legs as `<leg>_bbl` (adjusted) and `<leg>_bbl_unadj`,
    the spread `<name>` and `<name>_unadj`, and `rolled`."""
    out = {}
    for name in traded_spreads(cfg):
        pre = f"{name}__"
        f = fut[[c for c in fut.columns if c.startswith(pre)]].rename(columns=lambda c: c[len(pre):])
        f[name] = fut[name]
        f[f"{name}_unadj"] = fut[f"{name}_unadj"]
        f["rolled"] = fut[f"rolled_{name}"].astype(bool)
        out[name] = f
    return out
