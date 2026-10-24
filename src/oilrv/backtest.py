"""Steps 08-09: walk-forward backtest of the pre-registered rules.

Daily sequence at each close i (as in Project 2):
  1. P&L for (close i-1, close i] on the position held at close i-1, under the FROZEN entry
     weights: pnl = units * M * side * (S_i - S_{i-1}); side = -1 sells the spread, +1 buys it.
  2. Fill yesterday's decision at today's close (fill_lag_days = 1) and charge the cost.
  3. If flat with nothing pending: refit at a month start, or right after an exit, on the
     `window` rows strictly before i.
  4. Decide at today's close:
       flat     -> enter if every entry condition holds
       in trade -> exit if any exit condition holds. The monthly re-test runs while in a trade
                   (model-break exit) but never swaps the frozen parameters.

P&L streams:
  pnl_bbl : one unit, in $/bbl of spread (cost `cost_usd_per_bbl` per side)
  pnl_usd : research stream (gated): units = R / (sigma * M), unrounded, fixed at entry;
            or the integer book (size_mode='integer'): the plan's floored, capped size
On futures data a roll cost of `roll_cost_ticks_per_leg` ticks per contract leg is charged on each
close where the held position rolls (spot data has no rolls).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import load_config
from .diagnostics import month_starts
from .model import fit_at
from .sizing import drawdown_multiplier, integer_units, research_units


@dataclass
class Rules:
    """One backtest configuration. Defaults = the full registered rules (the gated ladder step)."""
    variant: str = "fixed"
    sizing: str = "garch"                  # "window_sd" (ladder: core) or "garch"
    use_blackout: bool = True
    use_logit: bool = True
    entry_z: float | None = None           # None -> config
    exit_z: float | None = None
    stop_z: float | None = None
    rearm_z: float | None = None
    cost_bbl: float | None = None
    window: int | None = None
    label: str = ""

    @classmethod
    def ladder(cls, step: str, cfg: dict | None = None, **kw) -> "Rules":
        s = (cfg or load_config())["ladder"][step]
        return cls(sizing=s["sizing"], use_blackout=s["blackout"], use_logit=s["logit"], label=step, **kw)

    def resolved(self, cfg: dict) -> dict:
        st, m = cfg["strategy"], cfg["model"]
        r = {
            "variant": self.variant, "sizing": self.sizing,
            "use_blackout": self.use_blackout, "use_logit": self.use_logit,
            "entry_z": self.entry_z if self.entry_z is not None else st["entry_z"],
            "exit_z": self.exit_z if self.exit_z is not None else st["exit_z"],
            "stop_z": self.stop_z if self.stop_z is not None else st["stop_z"],
            "cost_bbl": self.cost_bbl if self.cost_bbl is not None else st["cost_usd_per_bbl"],
            "window": self.window or m["window"],
            "time_stop_hl": st["time_stop_halflives"],
            "enter_p": st["regime_enter_p"], "exit_p": st["regime_exit_p"],
            "hl_min": m["half_life_min"], "hl_max": m["half_life_max"],
        }
        r["rearm_z"] = self.rearm_z if self.rearm_z is not None else (
            st["rearm_z"] if self.entry_z is None else r["entry_z"])
        if r["variant"] not in cfg["variants"] or r["sizing"] not in ("window_sd", "garch"):
            raise ValueError(f"bad rules {r}")
        return r


class FitCache:
    """fit_at results keyed by (window, row position) for one spread and variant."""

    def __init__(self, legs: pd.DataFrame, name: str, variant: str, cfg: dict):
        self.legs, self.name, self.variant, self.cfg, self._d = legs, name, variant, cfg, {}

    def seed(self, fits: pd.DataFrame, window: int) -> None:
        for _, row in fits.iterrows():
            self._d[(window, int(row["pos"]))] = row.to_dict()

    def get(self, pos: int, window: int) -> dict:
        key = (window, pos)
        if key not in self._d:
            self._d[key] = fit_at(self.legs, pos, self.name, self.variant, window, self.cfg)
        return self._d[key]


def _ok(fit: dict) -> bool:
    return np.isfinite(fit.get("sd", np.nan))


def gate_open(fit: dict, r: dict) -> bool:
    """Registered entry gate: half-life in band (and, for johansen, a usable vector)."""
    return bool(_ok(fit) and r["hl_min"] <= fit["half_life"] <= r["hl_max"])


class SpreadSim:
    """State machine for one spread; `step(i)` advances one close."""

    def __init__(self, name: str, legs: pd.DataFrame, cache: FitCache, p_hat: pd.Series, blackout: pd.Series,
                 rules: Rules, cfg: dict, rolled: pd.Series | None = None):
        self.name, self.cfg, self.cache = name, cfg, cache
        self.r = r = rules.resolved(cfg)
        spec = cfg["spreads"][name]
        self.M, self.div = float(spec["usd_per_1bbl_per_unit"]), float(spec["divisor"])
        self.idx = legs.index
        self.leg = {c[:-4]: legs[c].to_numpy(dtype=float) for c in legs.columns if c.endswith("_bbl")}
        self.month = self.idx.month.to_numpy()
        self.ms = set(month_starts(self.idx).tolist())
        self.first = min(s for s in self.ms if s >= r["window"])
        self.ph = p_hat.reindex(self.idx).to_numpy(dtype=float) if r["use_logit"] else np.full(len(self.idx), np.nan)
        self.bo = (blackout.reindex(self.idx).fillna(False).to_numpy(dtype=bool) if r["use_blackout"]
                   else np.zeros(len(self.idx), bool))
        self.rolled = rolled.reindex(self.idx).fillna(False).to_numpy(dtype=bool) if rolled is not None else None
        st = cfg["strategy"]
        self.roll_tick = {k: st["roll_cost_ticks_per_leg"] * v for k, v in st["tick_usd"].items()}
        self.fit, self.sig2, self.need_refit = None, np.nan, True
        self.side, self.units, self.frozen, self.fill_i = 0, 0.0, None, -1
        self.pending, self.armed, self.halted = None, False, False
        self.entry, self.trades, self.rows = {}, [], []
        self.blocked_by_size = 0

    # ------------------------------------------------------------------ helpers
    def S(self, p: dict, i: int) -> float:
        return sum(p[f"w_{k}"] * self.leg[k][i] for k in self.leg if f"w_{k}" in p and np.isfinite(p[f"w_{k}"])) / self.div

    def z(self, p: dict, i: int) -> float:
        if not _ok(p):
            return np.nan
        return (self.S(p, i) - p[f"seas_{self.month[i]}"] - p["mean"]) / p["sd"]

    def _units(self, sigma: float, az: float, mode: str, mult: float) -> float:
        if mode == "integer":
            return float(integer_units(sigma, az, self.fit["sd"], self.M, mult, self.cfg)["units"])
        return research_units(sigma, self.M, self.cfg)

    def _close_trade(self, i: int, reasons: list[str], decided: pd.Timestamp, cost_bbl: float, cost_usd: float):
        e, f = self.entry, self.frozen
        self.trades.append({
            "decision_date": e["decision_date"], "fill_date": e["fill_date"], "exit_decision_date": decided,
            "exit_fill_date": self.idx[i], "side": self.side, "z_entry": e["z"], "exit_reason": reasons[0],
            "all_exit_reasons": "+".join(reasons), "hold_days": i - self.fill_i, "half_life": f["half_life"],
            "units": self.units, "p_hat_entry": e["p_hat"], "sigma_entry": e["sigma"],
            "gross_bbl": e["gross_bbl"], "net_bbl": e["gross_bbl"] - e["cost_bbl"] - cost_bbl,
            "gross_usd": e["gross_usd"], "net_usd": e["gross_usd"] - e["cost_usd"] - cost_usd,
            **{k: f[k] for k in f if k.startswith("w_")},
        })

    # ------------------------------------------------------------------ one close
    def step(self, i: int, mode: str = "research", mult: float = 1.0, halt: bool = False) -> None:
        r = self.r
        pnl_bbl = pnl_usd = cost_bbl = cost_usd = 0.0
        # 1. P&L on the position held over (i-1, i]
        if self.side != 0:
            dS = self.S(self.frozen, i) - self.S(self.frozen, i - 1)
            pnl_bbl = self.side * dS
            pnl_usd = self.units * self.M * self.side * dS
            self.entry["gross_bbl"] += pnl_bbl
            self.entry["gross_usd"] += pnl_usd
            if self.rolled is not None and self.rolled[i]:
                roll_usd = self.units * sum(abs(self.frozen.get(f"w_{k}", 0.0)) * t for k, t in self.roll_tick.items())
                roll_bbl = roll_usd / (self.units * self.M)
                cost_usd += roll_usd
                cost_bbl += roll_bbl
                self.entry["cost_usd"] += roll_usd
                self.entry["cost_bbl"] += roll_bbl
        # 2. Fill yesterday's decision at today's close
        if self.pending is not None:
            kind = self.pending[0]
            if kind == "enter":
                _, self.side, self.units, info = self.pending
                self.frozen, self.fill_i = info["fit"], i
                c_bbl, c_usd = r["cost_bbl"], r["cost_bbl"] * self.units * self.M
                cost_bbl += c_bbl
                cost_usd += c_usd
                self.entry = {**info, "fill_date": self.idx[i], "gross_bbl": 0.0, "gross_usd": 0.0,
                              "cost_bbl": c_bbl, "cost_usd": c_usd}
            else:
                c_bbl, c_usd = r["cost_bbl"], r["cost_bbl"] * self.units * self.M
                cost_bbl += c_bbl
                cost_usd += c_usd
                self._close_trade(i, self.pending[1], self.pending[2], c_bbl, c_usd)
                self.side, self.units, self.frozen = 0, 0.0, None
                self.armed, self.need_refit = False, True
            self.pending = None
        # 3. Refit when flat
        if self.side == 0 and self.pending is None and (self.need_refit or i in self.ms):
            self.fit = self.cache.get(i, r["window"])
            self.sig2 = self.fit.get("garch_sig2_next", np.nan)        # forecast for dS_i
            self.need_refit = False
        fit = self.fit
        # GARCH state under the flat fit: observe dS_i, forecast for i+1
        sigma_next = np.nan
        if _ok(fit):
            dS_flat = self.S(fit, i) - self.S(fit, i - 1)
            if fit.get("garch_ok"):
                self.sig2 = fit["garch_omega"] + fit["garch_a"] * (dS_flat - fit["garch_mu"]) ** 2 + fit["garch_b"] * self.sig2
            sigma_next = float(np.sqrt(self.sig2))
        # 4. Decide
        params = self.frozen if self.side != 0 else fit
        z = self.z(params, i)
        if i < len(self.idx) - 1 and self.pending is None:
            if self.side == 0 and not halt and np.isfinite(z):
                if abs(z) < r["rearm_z"]:
                    self.armed = True
                p_ok = np.isnan(self.ph[i]) or self.ph[i] >= r["enter_p"]
                if (self.armed and gate_open(fit, r) and r["entry_z"] <= abs(z) < r["stop_z"]
                        and not self.bo[i] and p_ok):
                    sigma = sigma_next if r["sizing"] == "garch" else fit["dS_sd"]
                    u = self._units(sigma, abs(z), mode, mult)
                    if u > 0:
                        self.pending = ("enter", -int(np.sign(z)), u,
                                        {"fit": fit, "decision_date": self.idx[i], "z": z, "p_hat": self.ph[i],
                                         "sigma": sigma})
                    else:
                        self.blocked_by_size += 1
            elif self.side != 0:
                reasons = []
                if halt:
                    reasons.append("book_stop")
                if abs(z) >= r["stop_z"]:
                    reasons.append("stop_loss")
                if abs(z) <= r["exit_z"]:
                    reasons.append("take_profit")
                if i in self.ms and not gate_open(self.cache.get(i, r["window"]), r):
                    reasons.append("model_break")
                if not np.isnan(self.ph[i]) and self.ph[i] < r["exit_p"]:
                    reasons.append("regime_break")
                if (i - self.fill_i) > r["time_stop_hl"] * self.frozen["half_life"]:
                    reasons.append("time_stop")
                if reasons:
                    self.pending = ("exit", reasons, self.idx[i])
        self.rows.append((self.idx[i], self.side, z, pnl_bbl - cost_bbl, pnl_usd - cost_usd, cost_usd,
                          sigma_next, gate_open(fit, r), self.ph[i], self.units))

    def results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        daily = pd.DataFrame(self.rows, columns=["date", "side", "z", "pnl_bbl", "pnl_usd", "cost_usd",
                                                 "sigma_next", "gate_open", "p_hat", "units"]).set_index("date")
        tr = pd.DataFrame(self.trades)
        if self.side != 0:                     # still open at the end: marked to market, flagged
            e = self.entry
            tr = pd.concat([tr, pd.DataFrame([{
                "decision_date": e["decision_date"], "fill_date": e["fill_date"], "exit_reason": "open_at_end",
                "side": self.side, "z_entry": e["z"], "hold_days": len(self.idx) - 1 - self.fill_i,
                "half_life": self.frozen["half_life"], "units": self.units, "p_hat_entry": e["p_hat"],
                "sigma_entry": e["sigma"], "gross_bbl": e["gross_bbl"], "net_bbl": e["gross_bbl"] - e["cost_bbl"],
                "gross_usd": e["gross_usd"], "net_usd": e["gross_usd"] - e["cost_usd"]}])], ignore_index=True)
        daily.attrs["rules"] = self.r
        daily.attrs["blocked_by_size"] = self.blocked_by_size
        return daily, tr


def run_backtest(name: str, legs: pd.DataFrame, cache: FitCache, p_hat: pd.Series, blackout: pd.Series,
                 rules: Rules | None = None, cfg: dict | None = None, mode: str = "research",
                 rolled: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One spread, one configuration. Returns (daily, trades)."""
    cfg = cfg or load_config()
    sim = SpreadSim(name, legs, cache, p_hat, blackout, rules or Rules(), cfg, rolled)
    for i in range(sim.first, len(legs)):
        sim.step(i, mode)
    return sim.results()


def run_book(sims: dict[str, SpreadSim], cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Integer book across spreads with the pre-registered drawdown rules (shared equity).

    Before each close the book's drawdown from its peak sets the size multiplier for new entries
    (1, 0.5, 0). At the stop level every position is closed and the book stops for good ('review').
    """
    cfg = cfg or load_config()
    s = cfg["sizing"]
    names = list(sims)
    idx = sims[names[0]].idx
    first = max(sim.first for sim in sims.values())
    equity, peak, halted = float(s["book_usd"]), float(s["book_usd"]), False
    rows = []
    for i in range(first, len(idx)):
        dd = (peak - equity) / peak
        mult = drawdown_multiplier(dd, cfg)
        halted = halted or mult == 0.0
        day = 0.0
        for n in names:
            sims[n].step(i, "integer", mult, halted)
            day += sims[n].rows[-1][4]
        equity += day
        peak = max(peak, equity)
        rows.append((idx[i], day, equity, dd, mult, halted))
    book = pd.DataFrame(rows, columns=["date", "pnl_usd", "equity", "drawdown", "size_mult", "halted"]).set_index("date")
    return book, {n: sim.results() for n, sim in sims.items()}
