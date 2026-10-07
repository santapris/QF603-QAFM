"""Phase 6.5 cost model (all parameters from ``config.strategy``; logged in DECISIONS.md).

* Options: fill at mid ∓ θ × half-spread (sell below mid, buy above), θ = ``option_spread_theta``; plus
  ``option_fee_per_contract`` dollars per contract (SPX multiplier 100 → fee / 100 index points per unit).
  Held to cash settlement: no exit spread, no exit fee.
* Hedge: index futures, ``hedge_cost_bp`` basis points of traded notional per side.
A zero-cost model reproduces the gross P&L exactly (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

MULTIPLIER = 100


@dataclass(frozen=True)
class CostModel:
    theta: float
    fee_per_contract: float
    hedge_bp: float
    label: str = "custom"

    @classmethod
    def from_config(cls, cfg: dict, scenario: str = "base") -> "CostModel":
        s = cfg["strategy"]
        fee = s["option_fee_per_contract"]
        if fee is None:
            raise ValueError("strategy.option_fee_per_contract is not set — 🧑 choose and log it (PLAN 6.5)")
        fee = fee[scenario] if isinstance(fee, dict) else fee
        return cls(theta=s["option_spread_theta"][scenario], fee_per_contract=float(fee),
                   hedge_bp=s["hedge_cost_bp"][scenario], label=scenario)

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(0.0, 0.0, 0.0, "zero")

    def scaled(self, k: float) -> "CostModel":
        return CostModel(self.theta * k, self.fee_per_contract * k, self.hedge_bp * k, f"{self.label}×{k:g}")

    def option_entry_cost(self, half_spreads: list[float]) -> float:
        """Index points lost at entry versus mid, for one contract of each leg."""
        return self.theta * sum(half_spreads) + len(half_spreads) * self.fee_per_contract / MULTIPLIER

    def hedge_cost(self, traded_units: float, index_level: float) -> float:
        return self.hedge_bp * 1e-4 * abs(traded_units) * index_level
