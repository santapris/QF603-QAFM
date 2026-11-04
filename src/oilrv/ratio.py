"""Exploratory strategy on the Bloomberg futures: log-ratio mean reversion.

Registered 2026-11-04 (PREREGISTRATION_RATIO.md, config `exploratory_ratio`), after the binding dollar-gap verdict,
the trend and carry and pairs results and the descriptive ratio stationarity study, before any signal or P&L of
this strategy. It never changes the binding verdict.

Signal and money are kept apart on purpose:

- z reads the MARKET log-ratio of the held contracts, L = log(A / B), where A and B are the long and short sides of the
  spread in $/bbl (`*_bbl_unadj`). That is what refiners and shippers react to. The month-of-year effect is removed inside
  the window for the seasonal spreads.
- P&L, the GARCH state and costs use BACK-ADJUSTED legs, whose daily changes are exactly what a position earns.

Fit, on the window strictly before each refit:

    R = L - month effect,   z_t = (R_t - mean) / sd,   half-life = AR(1) half-life of R
    s_A * P_A = s_B * P_B   (equal market notional on both sides at the last close of the window)
    w_k = +s_A * n_k (side A),  -s_B * n_k (side B),  sum |w_k| = gross contracts per unit (2)
    S   = sum w_k * (back-adjusted leg k)

A unit therefore earns about (change in the log-ratio) x (side notional), and the registered cost of $0.05/bbl per unit
of position change is $25 per contract. Trading reuses `backtest.SpreadSim` (fills, costs, rolls, GARCH state, re-arm,
stops, trade records); `RatioSim` changes only how z is read.

Modes (the comparisons are reported, not gated):
    base           z from the market log-ratio
    dollar_market  same weights and window, z from the market DOLLAR spread sum w_k * (market leg k)
    no_gate        base, with the half-life clipped into the registered band so the gate never closes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import futures
from .backtest import Rules, SpreadSim, _ok
from .backtest import run_book as _integer_run_book
from .sizing import integer_units
from .config import load_config
from .diagnostics import ar1, window_before
from .metrics import sharpe, subperiods, window_pnl
from .model import MONTHS, deseasonalise, fit_garch, seasonal_effects

MODES = ("base", "dollar_market", "no_gate")
TRADE_EXTRAS = ("k_hat",)


# --------------------------------------------------------------------------- data
def build_data(cfg: dict | None = None, fut: pd.DataFrame | None = None, blackout: pd.Series | None = None) -> dict:
    """{'specs', 'legs', 'rolled', 'blackout', 'stores'} per spread, on the common futures dates.

    Legs: `<leg>_bbl` (back-adjusted) and `<leg>_bbl_unadj` (market), $/bbl, from data/clean/futures_spreads.csv."""
    cfg = cfg or load_config()
    fut = fut if fut is not None else futures.load_futures()
    if blackout is None:
        from .data import load_source
        blackout = load_source("futures").cal["event_blackout"]
    specs = cfg["exploratory_ratio"]["spreads"]
    legs, rolled = {}, {}
    for name, spec in specs.items():
        pre = f"{name}__"
        names = list(spec["a"]) + list(spec["b"])
        cols = {}
        for k in names:
            cols[f"{k}_bbl"] = fut[f"{pre}{k}_bbl"]
            cols[f"{k}_bbl_unadj"] = fut[f"{pre}{k}_bbl_unadj"]
        legs[name] = pd.DataFrame(cols, index=fut.index)
        rolled[name] = fut[f"rolled_{name}"].astype(bool)
    return {"specs": specs, "legs": legs, "rolled": rolled,
            "blackout": blackout.reindex(fut.index).fillna(False).astype(bool), "stores": {n: {} for n in specs}}


def ratio_cfg(name: str, cfg: dict) -> dict:
    """An in-memory config the engine can read for this spread; config.yaml's registered blocks stay as they are."""
    rx = cfg["exploratory_ratio"]
    spec = {"usd_per_1bbl_per_unit": rx["usd_per_1bbl_per_unit"], "divisor": 1}
    return {**cfg, "spreads": {**cfg["spreads"], name: spec},
            "sizing": {**cfg["sizing"], "risk_per_day_usd": rx["risk_per_day_usd"]}}


# --------------------------------------------------------------------------- fit
def side_value(win: pd.DataFrame, side: dict, suffix: str = "_bbl_unadj") -> pd.Series:
    return sum(n * win[f"{k}{suffix}"] for k, n in side.items())


def neutral_weights(a: dict, b: dict, p_a: float, p_b: float, gross: float) -> dict[str, float]:
    """Contracts per unit with equal market notional on both sides and `gross` contracts in total."""
    na, nb = sum(a.values()), sum(b.values())
    s_a = gross / (na + nb * p_a / p_b)
    s_b = s_a * p_a / p_b
    return {**{k: s_a * n for k, n in a.items()}, **{k: -s_b * n for k, n in b.items()}}


def fit_ratio(win: pd.DataFrame, spec: dict, pos: int, mode: str = "base", cfg: dict | None = None) -> dict:
    """Everything the engine reads for one spread from one window, before any gate."""
    cfg = cfg or load_config()
    rx, m = cfg["exploratory_ratio"], cfg["model"]
    a, b = spec["a"], spec["b"]
    A, B = side_value(win, a), side_value(win, b)
    w = neutral_weights(a, b, float(A.iloc[-1]), float(B.iloc[-1]), rx["gross_contracts_per_unit"])
    log_ratio = np.log(A / B)
    if mode == "dollar_market":
        x = sum(w[k] * win[f"{k}_bbl_unadj"] for k in w)
    else:
        x = log_ratio
    seas = seasonal_effects(x) if spec["seasonal_adjust"] else {k: 0.0 for k in MONTHS}
    r = deseasonalise(x, seas)
    _, hl = ar1(r)
    ds = np.diff(sum(w[k] * win[f"{k}_bbl"] for k in w).to_numpy())          # tradable changes, always back-adjusted
    half_life = hl
    if mode == "no_gate":
        half_life = float(np.clip(hl, m["half_life_min"], m["half_life_max"]))
    p = {"pos": pos, "mode": mode, "half_life": half_life, "half_life_raw": hl,
         "mean": float(r.mean()), "sd": float(r.std(ddof=1)), "dS_sd": float(ds.std(ddof=1)),
         "k_hat": float(np.exp(deseasonalise(log_ratio, seas).mean())),
         **{f"w_{k}": v for k, v in w.items()}}
    # The engine's stop-loss size limit needs the window sd of the signal in $/bbl of S. For the log-ratio, a move of d in the
    # log-ratio is a move of about d * V_A in S, V_A the market value of side A of one unit ($/bbl, last close of the window).
    v_a = float(sum(w[k] * win[f"{k}_bbl_unadj"].iloc[-1] for k in a))
    p["sd_S"] = p["sd"] if mode == "dollar_market" else p["sd"] * v_a
    p.update({f"seas_{k}": v for k, v in seas.items()})
    p.update({f"garch_{k}": v for k, v in fit_garch(ds).items()})
    return p


class RatioFitCache:
    """Same `get(pos, window)` as `backtest.FitCache`. `store` lets runs that differ only in rules share fits."""

    def __init__(self, legs: pd.DataFrame, spec: dict, mode: str, cfg: dict, store: dict | None = None):
        self.legs, self.spec, self.mode, self.cfg = legs, spec, mode, cfg
        self.store = {} if store is None else store

    def get(self, pos: int, window: int) -> dict:
        key = (self.mode, window, pos)
        if key not in self.store:
            self.store[key] = fit_ratio(window_before(self.legs, pos, window), self.spec, pos, self.mode, self.cfg)
        return self.store[key]


# --------------------------------------------------------------------------- engine
class RatioSim(SpreadSim):
    """`SpreadSim` with z read from market prices. P&L and the GARCH state still use the back-adjusted legs."""

    def __init__(self, name, legs, spec, cache, p_hat, blackout, rules, cfg, rolled=None, decision_months=None):
        super().__init__(name, legs, cache, p_hat, blackout, rules, cfg, rolled)
        self.spec = spec
        # None: decide every day. k: decide only on the first trading day of months 1, 1 + k, 1 + 2k, ... (k = 1 is every
        # month, k = 3 is January, April, July and October). Elsewhere z is not read, so nothing is entered, exited or
        # stopped; the time stop and the monthly model-break test are calendar rules and still run as registered.
        self.decide = (None if decision_months is None else np.array(
            [(i in self.ms) and ((self.month[i] - 1) % decision_months == 0) for i in range(len(self.idx))]))
        self.mkt = {c[:-len("_bbl_unadj")]: legs[c].to_numpy(dtype=float)
                    for c in legs.columns if c.endswith("_bbl_unadj")}

    def x(self, p: dict, i: int) -> float:
        """The signal variable on day i: the market log-ratio, or for `dollar_market` the market dollar spread."""
        if p.get("mode") == "dollar_market":
            return sum(p[f"w_{k}"] * self.mkt[k][i] for k in self.mkt)
        a = sum(n * self.mkt[k][i] for k, n in self.spec["a"].items())
        b = sum(n * self.mkt[k][i] for k, n in self.spec["b"].items())
        return float(np.log(a / b))

    def _units(self, sigma: float, az: float, mode: str, mult: float) -> float:
        """Research stream as registered; the integer book (PREREGISTRATION sizing: $40k stop limit, 3-unit cap) uses sd_S."""
        if mode == "integer":
            return float(integer_units(sigma, az, self.fit["sd_S"], self.M, mult, self.cfg)["units"])
        return super()._units(sigma, az, mode, mult)

    def z_raw(self, p: dict, i: int) -> float:
        """z on any day, whether or not it is a decision day (for reporting)."""
        if not _ok(p):
            return np.nan
        return (self.x(p, i) - p[f"seas_{self.month[i]}"] - p["mean"]) / p["sd"]

    def z(self, p: dict, i: int) -> float:
        """The z the engine decides on: NaN away from decision days when decisions are slowed down."""
        if self.decide is not None and not self.decide[i]:
            return np.nan
        return self.z_raw(p, i)

    def _close_trade(self, i, reasons, decided, cost_bbl, cost_usd):
        super()._close_trade(i, reasons, decided, cost_bbl, cost_usd)
        self.trades[-1].update({k: self.frozen.get(k, np.nan) for k in TRADE_EXTRAS})

    def results(self):
        daily, tr = super().results()
        if self.side != 0:
            for k in TRADE_EXTRAS:
                tr.loc[tr.index[-1], k] = self.frozen.get(k, np.nan)
        return daily, tr


def make_sim(name: str, data: dict, cfg: dict | None = None, mode: str = "base", entry_z: float | None = None,
             exit_z: float | None = None, window: int | None = None, cost_bbl: float | None = None,
             decision_months: int | None = None) -> "RatioSim":
    """One spread's simulator, not yet stepped. `decision_months` slows the decisions (None = every day, as in
    PREREGISTRATION_RATIO.md; 1 = monthly, 3 = quarterly; PREREGISTRATION_RATIO_SLOW.md)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    cfg = cfg or load_config()
    c = ratio_cfg(name, cfg)
    window = window or cfg["model"]["window"]
    legs, spec = data["legs"][name], data["specs"][name]
    cache = RatioFitCache(legs, spec, mode, c, data["stores"][name])
    rules = Rules(variant="fixed", sizing="garch", use_blackout=True, use_logit=False, entry_z=entry_z, exit_z=exit_z,
                  cost_bbl=cost_bbl, window=window, label=mode)
    return RatioSim(name, legs, spec, cache, pd.Series(dtype=float), data["blackout"], rules, c, data["rolled"][name],
                    decision_months)


def run_spread(name: str, data: dict, cfg: dict | None = None, **kw) -> dict:
    """One spread, one configuration (research stream). Returns {'daily', 'trades', 'sim'}."""
    sim = make_sim(name, data, cfg, **kw)
    for i in range(sim.first, len(sim.idx)):
        sim.step(i)
    daily, trades = sim.results()
    return {"daily": daily, "trades": trades, "sim": sim}


def run_integer_book(data: dict, cfg: dict | None = None, **kw) -> dict:
    """The registered integer book ($2m, units floored and capped at 3, $40k stop limit, halve size after a 10% drawdown, stop at
    20%) across the three spreads, with the same rules as the research stream. Reported next to it, never gated."""
    cfg = cfg or load_config()
    sims = {n: make_sim(n, data, cfg, **kw) for n in data["specs"]}
    book, res = _integer_run_book(sims, cfg)
    return {"book": book, "spreads": {n: {"daily": r[0], "trades": r[1], "sim": sims[n]} for n, r in res.items()}}


def round_trips(trades: pd.DataFrame) -> int:
    return int((trades["exit_reason"] != "open_at_end").sum()) if len(trades) else 0


def run_book(data: dict, cfg: dict | None = None, **kw) -> dict:
    """Every spread under one configuration and their equal-risk average.

    Returns {'spreads': {name: run}, 'book': daily P&L, 'round_trips': closed trades summed over spreads}."""
    cfg = cfg or load_config()
    runs = {n: run_spread(n, data, cfg, **kw) for n in data["specs"]}
    book = pd.DataFrame({n: r["daily"]["pnl_usd"] for n, r in runs.items()}).mean(axis=1)
    return {"spreads": runs, "book": book, "round_trips": sum(round_trips(r["trades"]) for r in runs.values())}


# --------------------------------------------------------------------------- evaluation
def verdict(pnl: pd.Series, n_round_trips: int, grid: dict[str, float], cfg: dict | None = None) -> dict:
    """G1-G5 of PREREGISTRATION_RATIO.md section 3 for one P&L stream."""
    cfg = cfg or load_config()
    g = cfg["exploratory_ratio"]["gates"]
    frame = pnl.to_frame("pnl")
    sub = subperiods(frame, "pnl", cfg["evaluation"]["sub_periods"])
    stress = {w: window_pnl(frame, "pnl", *cfg["evaluation"]["stress_windows"][w]) for w in g["stress_windows"]}
    sr = sharpe(pnl)
    out = {
        "G1_sharpe": bool(np.isfinite(sr) and sr > g["min_net_sharpe"]), "G1_value": sr,
        "G2_round_trips": bool(n_round_trips >= g["min_round_trips"]), "G2_value": n_round_trips,
        "G3_subperiods": bool((sub["net"] > 0).sum() >= g["min_positive_subperiods"]),
        "G3_value": int((sub["net"] > 0).sum()),
        "G4_grid": bool(all(np.isfinite(v) and v > g["grid_min_sharpe"] for v in grid.values())), "G4_value": grid,
        "G5_stress": bool(all(v > -g["stress_max_loss_usd"] for v in stress.values())), "G5_value": stress,
    }
    out["PASS"] = all(out[k] for k in ["G1_sharpe", "G2_round_trips", "G3_subperiods", "G4_grid", "G5_stress"])
    return out


def single_spread_evidence(spread_verdict: dict, t_stat: float, cfg: dict | None = None) -> bool:
    """One spread counts only if it passes every gate and clears the three-test Bonferroni bar."""
    return bool(spread_verdict["PASS"] and t_stat > (cfg or load_config())["exploratory_ratio"]["single_spread_min_t"])


def roll_attribution(run: dict, rolled: pd.Series) -> pd.DataFrame:
    """Per closed trade: gross P&L over the hold and the part of it earned on roll days (the roll jump in the signal ratio
    is not earned by a holder, but a held position can still gain or lose on a roll close)."""
    daily, trades = run["daily"], run["trades"]
    if not len(trades):
        return trades.assign(roll_day_gross_usd=[])
    gross = daily["pnl_usd"] + daily["cost_usd"]
    roll = rolled.reindex(daily.index).fillna(False).to_numpy(dtype=bool)
    out = trades[trades["exit_reason"] != "open_at_end"].copy()
    rd = []
    for _, t in out.iterrows():
        sel = (daily.index > t["fill_date"]) & (daily.index <= t["exit_fill_date"])
        rd.append(float(gross[sel & roll].sum()))
    out["roll_day_gross_usd"] = rd
    return out


def daily_paths(spreads: dict[str, dict]) -> pd.DataFrame:
    parts = [r["daily"].assign(spread=n).rename_axis("date").reset_index() for n, r in spreads.items()]
    cols = ["date", "spread", "side", "units", "z", "gate_open", "sigma_next", "pnl_bbl", "pnl_usd", "cost_usd"]
    return pd.concat(parts, ignore_index=True)[cols]
