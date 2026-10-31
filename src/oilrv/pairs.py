"""Exploratory strategy on the Bloomberg futures: cointegration pairs.

Registered 2026-10-31 (PREREGISTRATION_PAIRS.md, config `exploratory_pairs`), after the binding
mean-reversion verdict and the trend and carry result, before any signal or P&L of this strategy.
It never changes that verdict.

Instruments. Each pair holds one delivery month in both legs (`futures.continuous`, k = 1, rolled 5
business days before the earlier expiry). Levels are additive back-adjusted $/bbl, so daily changes are
what a position earns. All series sit on the common futures dates.

Fit, on the window strictly before each refit:

    y = a + beta * x + c * t + e                  (Engle-Granger: coint(y, x, trend="ct"))
    w_y = gross / (1 + |beta|),   w_x = -beta * w_y,   S = w_y * y + w_x * x
    R_t = S_t - c_S * t - seas[month(t)],   c_S = w_y * c,   z_t = (R_t - mean) / sd

The trend absorbs average roll carry and keeps extrapolating until the next refit; t counts rows from the
window start. A failed gate sets `half_life` to NaN, which closes `backtest.gate_open`; the fitted value
stays in `half_life_raw`.

Trading reuses `backtest.SpreadSim`, which supplies the registered fills, costs, rolls, GARCH state,
re-arm, stops and trade records. `PairSim` changes only how z is read.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from . import futures
from .backtest import Rules, SpreadSim, _ok
from .config import load_config
from .diagnostics import ar1, window_before
from .metrics import sharpe, subperiods, window_pnl
from .model import MONTHS, deseasonalise, fit_garch, seasonal_effects
from .sizing import research_units
from .spreads import to_bbl

# mode: (fit signal, Engle-Granger gate on)
MODES = {"base": ("adjusted_with_trend", True), "no_gate": ("adjusted_with_trend", False),
         "adjusted": ("adjusted", True), "market": ("market", True)}
TRADE_EXTRAS = ("beta", "trend_c", "eg_p")


# --------------------------------------------------------------------------- data
def pair_name(spec: dict) -> str:
    return f"{spec['y']}/{spec['x']}"


def pair_legs(y: str, x: str, panels: dict[str, pd.DataFrame], common: pd.DatetimeIndex,
              cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """(legs, rolled) on `common`. Legs: `<leg>_bbl` (adjusted) and `<leg>_bbl_unadj`, $/bbl, one delivery month."""
    cfg = cfg or load_config()
    roots = futures.leg_roots(cfg)
    c = futures.continuous({y: panels[y], x: panels[x]}, {y: roots[y], x: roots[x]}, 1,
                           cfg["futures"]["roll_bdays_before_expiry"], futures.load_expiry_overrides())
    adj = to_bbl(c[[f"{y}_adj", f"{x}_adj"]].rename(columns=lambda s: s[:-4]), cfg).reindex(common)
    raw = to_bbl(c[[y, x]], cfg).reindex(common)
    legs = pd.concat([adj.add_suffix("_bbl"), raw.add_suffix("_bbl_unadj")], axis=1)
    return legs, futures._to_common(c["rolled"], common)


def build_data(cfg: dict | None = None, panels: dict[str, pd.DataFrame] | None = None,
               common: pd.DatetimeIndex | None = None, blackout: pd.Series | None = None) -> dict:
    """{'specs', 'legs', 'rolled', 'blackout', 'stores'} per pair name, on the common futures dates."""
    from .data import load_source
    cfg = cfg or load_config()
    panels = panels or futures.load_panels(cfg)
    if common is None or blackout is None:
        src = load_source("futures")
        common = src.index if common is None else common
        blackout = src.cal["event_blackout"] if blackout is None else blackout
    specs = {pair_name(s): s for s in cfg["exploratory_pairs"]["pairs"]}
    built = {n: pair_legs(s["y"], s["x"], panels, common, cfg) for n, s in specs.items()}
    return {"specs": specs, "legs": {n: b[0] for n, b in built.items()}, "rolled": {n: b[1] for n, b in built.items()},
            "blackout": blackout.reindex(common).fillna(False).astype(bool), "stores": {n: {} for n in specs}}


def pair_cfg(name: str, cfg: dict) -> dict:
    """An in-memory config the engine can read for this pair; config.yaml's registered blocks stay as they are."""
    px = cfg["exploratory_pairs"]
    spec = {"usd_per_1bbl_per_unit": px["usd_per_1bbl_per_unit"], "divisor": 1}
    return {**cfg, "spreads": {**cfg["spreads"], name: spec},
            "sizing": {**cfg["sizing"], "risk_per_day_usd": px["risk_per_day_usd"]}}


# --------------------------------------------------------------------------- fit
def fit_pair(win: pd.DataFrame, y: str, x: str, seasonal: bool, pos: int, window: int,
             signal: str = "adjusted_with_trend", cfg: dict | None = None) -> dict:
    """Everything the engine reads for one pair from one window, before the Engle-Granger gate is applied."""
    cfg = cfg or load_config()
    gross = cfg["exploratory_pairs"]["gross_contracts_per_unit"]
    level = "_bbl_unadj" if signal == "market" else "_bbl"
    with_trend = signal == "adjusted_with_trend"
    ys, xs = win[f"{y}{level}"], win[f"{x}{level}"]
    t = np.arange(len(win), dtype=float)
    X = sm.add_constant(np.column_stack([xs.to_numpy(), t] if with_trend else [xs.to_numpy()]))
    params = sm.OLS(ys.to_numpy(), X).fit().params
    beta, c = float(params[1]), float(params[2]) if with_trend else 0.0
    w_y = gross / (1.0 + abs(beta))
    w_x = -beta * w_y
    c_s = w_y * c
    res = w_y * ys + w_x * xs - c_s * t
    seas = seasonal_effects(res) if seasonal else {k: 0.0 for k in MONTHS}
    r = deseasonalise(res, seas)
    _, hl = ar1(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg_p = float(coint(ys, xs, trend="ct" if with_trend else "c", autolag="aic")[1])
    ds = np.diff((w_y * win[f"{y}_bbl"] + w_x * win[f"{x}_bbl"]).to_numpy())      # tradable changes, always adjusted
    p = {"pos": pos, f"w_{y}": w_y, f"w_{x}": w_x, "beta": beta, "trend_c": c_s, "trend_i0": pos - window,
         "market_signal": signal == "market", "eg_p": eg_p, "half_life": hl, "half_life_raw": hl,
         "mean": float(r.mean()), "sd": float(r.std(ddof=1)), "dS_sd": float(ds.std(ddof=1))}
    p.update({f"seas_{k}": v for k, v in seas.items()})
    p.update({f"garch_{k}": v for k, v in fit_garch(ds).items()})
    return p


def with_gate(fit: dict, eg_p_max: float | None) -> dict:
    """The fit as the engine should see it: a failed Engle-Granger test closes the half-life gate."""
    if eg_p_max is None or fit["eg_p"] < eg_p_max:
        return fit
    return {**fit, "half_life": np.nan}


class PairFitCache:
    """Same `get(pos, window)` as `backtest.FitCache`. `store` lets runs that differ only in rules share fits."""

    def __init__(self, legs: pd.DataFrame, spec: dict, signal: str, eg_p_max: float | None, cfg: dict,
                 store: dict | None = None):
        self.legs, self.spec, self.signal, self.eg_p_max, self.cfg = legs, spec, signal, eg_p_max, cfg
        self.store = {} if store is None else store

    def get(self, pos: int, window: int) -> dict:
        key = (self.signal, window, pos)
        if key not in self.store:
            self.store[key] = fit_pair(window_before(self.legs, pos, window), self.spec["y"], self.spec["x"],
                                       self.spec["seasonal_adjust"], pos, window, self.signal, self.cfg)
        return with_gate(self.store[key], self.eg_p_max)


# --------------------------------------------------------------------------- engine
class PairSim(SpreadSim):
    """`SpreadSim` with z read net of the fitted trend and, for the `market` comparison, from unadjusted legs.
    P&L and the GARCH state still use the adjusted legs."""

    def __init__(self, name, legs, cache, p_hat, blackout, rules, cfg, rolled=None):
        super().__init__(name, legs, cache, p_hat, blackout, rules, cfg, rolled)
        self.sig_leg = {c[:-len("_bbl_unadj")]: legs[c].to_numpy(dtype=float)
                        for c in legs.columns if c.endswith("_bbl_unadj")}

    def z(self, p: dict, i: int) -> float:
        if not _ok(p):
            return np.nan
        level = self.sig_leg if p.get("market_signal") else self.leg
        s = sum(p[f"w_{k}"] * level[k][i] for k in level) / self.div
        trend = p.get("trend_c", 0.0) * (i - p.get("trend_i0", 0))
        return (s - trend - p[f"seas_{self.month[i]}"] - p["mean"]) / p["sd"]

    def _close_trade(self, i, reasons, decided, cost_bbl, cost_usd):
        super()._close_trade(i, reasons, decided, cost_bbl, cost_usd)
        self.trades[-1].update({k: self.frozen.get(k, np.nan) for k in TRADE_EXTRAS})

    def results(self):
        daily, tr = super().results()
        if self.side != 0:                     # the engine's open_at_end row has no fit fields
            for k in TRADE_EXTRAS:
                tr.loc[tr.index[-1], k] = self.frozen.get(k, np.nan)
        return daily, tr


def run_pair(name: str, data: dict, cfg: dict | None = None, mode: str = "base", eg_p_max: float | None = None,
             entry_z: float | None = None, window: int | None = None, cost_bbl: float | None = None) -> dict:
    """One pair, one configuration. Returns {'daily', 'trades', 'sim'}."""
    cfg = cfg or load_config()
    c = pair_cfg(name, cfg)
    signal, gated = MODES[mode]
    eg = (eg_p_max if eg_p_max is not None else cfg["exploratory_pairs"]["eg_p_max"]) if gated else None
    window = window or cfg["model"]["window"]
    legs = data["legs"][name]
    cache = PairFitCache(legs, data["specs"][name], signal, eg, c, data["stores"][name])
    rules = Rules(variant="fixed", sizing="garch", use_blackout=True, use_logit=False, entry_z=entry_z,
                  cost_bbl=cost_bbl, window=window, label=mode)
    sim = PairSim(name, legs, cache, pd.Series(dtype=float), data["blackout"], rules, c, data["rolled"][name])
    for i in range(sim.first, len(legs)):
        sim.step(i)
    daily, trades = sim.results()
    return {"daily": daily, "trades": trades, "sim": sim}


def run_book(data: dict, cfg: dict | None = None, **kw) -> dict:
    """Every pair under one configuration and their equal-risk average.

    Returns {'pairs': {name: run}, 'book': daily P&L, 'round_trips': closed trades summed over pairs}."""
    cfg = cfg or load_config()
    pairs = {n: run_pair(n, data, cfg, **kw) for n in data["specs"]}
    book = pd.DataFrame({n: r["daily"]["pnl_usd"] for n, r in pairs.items()}).mean(axis=1)
    return {"pairs": pairs, "book": book, "round_trips": sum(round_trips(r["trades"]) for r in pairs.values())}


def round_trips(trades: pd.DataFrame) -> int:
    return int((trades["exit_reason"] != "open_at_end").sum()) if len(trades) else 0


# --------------------------------------------------------------------------- evaluation
def verdict(pnl: pd.Series, n_round_trips: int, grid: dict[str, float], cfg: dict | None = None) -> dict:
    """G1-G5 of PREREGISTRATION_PAIRS.md section 3 for one P&L stream."""
    cfg = cfg or load_config()
    g = cfg["exploratory_pairs"]["gates"]
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


def single_pair_evidence(name: str, pair_verdict: dict, t_stat: float, cfg: dict | None = None) -> bool:
    """One pair counts only if it passes every gate, clears Bonferroni and is not a registered spread's legs."""
    px = (cfg or load_config())["exploratory_pairs"]
    return bool(pair_verdict["PASS"] and t_stat > px["single_pair_min_t"] and name in px["new_pairs"])


# --------------------------------------------------------------------------- descriptive (not gated)
def carry_split(trades: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Per trade: the fitted carry earned over the hold, and the rest of the gross P&L (convergence)."""
    M = (cfg or load_config())["exploratory_pairs"]["usd_per_1bbl_per_unit"]
    out = trades.copy()
    if not len(out):
        return out.assign(carry_usd=[], convergence_usd=[])
    out["carry_usd"] = out["side"] * out["units"] * M * out["trend_c"] * out["hold_days"]
    out["convergence_usd"] = out["gross_usd"] - out["carry_usd"]
    return out


def by_side(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades, net P&L and win rate by direction (+1 long the spread, -1 short)."""
    if not len(trades):
        return pd.DataFrame(columns=["trades", "net", "avg", "win_rate", "worst", "avg_hold"])
    g = trades.groupby("side")["net_usd"]
    return pd.DataFrame({"trades": g.size(), "net": g.sum(), "avg": g.mean(), "win_rate": g.apply(lambda s: (s > 0).mean()),
                         "worst": g.min(), "avg_hold": trades.groupby("side")["hold_days"].mean()})


def daily_paths(pairs: dict[str, dict]) -> pd.DataFrame:
    """Long frame of every pair's daily path."""
    parts = [r["daily"].assign(pair=n).rename_axis("date").reset_index() for n, r in pairs.items()]
    cols = ["date", "pair", "side", "units", "z", "gate_open", "sigma_next", "pnl_bbl", "pnl_usd", "cost_usd"]
    return pd.concat(parts, ignore_index=True)[cols]


def latest_state(pairs: dict[str, dict], data: dict, cfg: dict | None = None) -> pd.DataFrame:
    """Latest close per pair: z, gate, position, and the contracts per leg a new position would hold."""
    cfg = cfg or load_config()
    rows = {}
    for n, r in pairs.items():
        sim, last = r["sim"], r["daily"].iloc[-1]
        p = sim.frozen if sim.side != 0 else sim.fit
        spec = data["specs"][n]
        units = research_units(last["sigma_next"], sim.M, sim.cfg)
        rows[n] = {"date": r["daily"].index[-1].date(), "z": last["z"], "gate_open": bool(last["gate_open"]),
                   "side": int(last["side"]), "held_units": last["units"], "beta": p["beta"], "eg_p": p["eg_p"],
                   "half_life": p["half_life_raw"], "sigma_usd_bbl": last["sigma_next"],
                   "implied_contracts": ", ".join(f"{k.upper()} {p[f'w_{k}'] * units:+.2f}" for k in (spec["y"], spec["x"]))}
    return pd.DataFrame(rows).T
