"""Guards the pre-registered gates, rules and variants (fixed 2026-10-24, before any data was downloaded).

If this test fails, someone edited a pre-committed value. Robustness variants belong in
config['robustness'], not here. Changing these numbers needs an explicit, documented decision.
"""
from oilrv.config import load_config

REGISTERED_ON = "2026-10-24"

REGISTERED_GATES = {
    "min_net_sharpe": 0.5,
    "min_round_trips": 20,
    "min_positive_subperiods": 2,
    "entry_grid_stability": {"min_sharpe_all_grid_points": 0.0},
    "stress_filter": {"windows": ["ukraine", "hormuz"], "filtered_ge_unfiltered": True, "max_loss_usd": 40000},
}
REGISTERED_STRATEGY = {
    "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "time_stop_halflives": 3.0, "rearm_z": 2.0,
    "entry_gate": ["half_life_in_band"], "model_break_exit": "half_life_out_of_band",
    "regime_enter_p": 0.6, "regime_exit_p": 0.4,
    "event_blackout_days": 1, "blackout_event_types": ["OPEC", "HURRICANE"],
    "fill_lag_days": 1, "cost_usd_per_bbl": 0.05, "roll_cost_ticks_per_leg": 1,
}
REGISTERED_MODEL = {
    "window": 756, "reestimate": "monthly", "seasonal": "month_of_year_in_window",
    "half_life_min": 5, "half_life_max": 120, "rolling_adf_window": 756,
    "garch": {"p": 1, "q": 1, "dist": "t"},
}
REGISTERED_SPREADS = {
    "crack321": {"legs": {"cl": -3, "rb": 2, "ho": 1}, "divisor": 3, "usd_per_1bbl_per_unit": 3000,
                 "seasonal_adjust": True, "traded": True},
    "brent_wti": {"legs": {"brent": 1, "cl": -1}, "divisor": 1, "usd_per_1bbl_per_unit": 1000,
                  "seasonal_adjust": False, "traded": True},
    "ho_rb": {"legs": {"ho": 1, "rb": -1}, "divisor": 1, "usd_per_1bbl_per_unit": 1000,
              "seasonal_adjust": True, "traded": True},
    "brent_m1m6": {"traded": False},
}
REGISTERED_REGIME = {
    "label_revert_z": 1.0, "label_stop_z": 4.0, "label_horizon_days": 20, "pooled": True,
    "refit": "monthly_expanding", "min_episodes": 20,
    "core_features": ["ovx_pct", "garch_vol_pct", "rolling_adf_p", "gpr", "inventory_dev_5y"],
    "robustness_features": ["cftc_mm_z", "hurricane_season", "event_within_3d", "brent_m1m6_slope_pct"],
}
REGISTERED_SIZING = {
    "risk_per_day_usd": 6000, "max_loss_per_trade_usd": 40000, "max_units": 3,
    "drawdown_halve": 0.10, "drawdown_stop": 0.20, "research_stream": "unrounded_vol_target",
}
REGISTERED_BINDING = {"binding_source": "futures", "proxy_source": "spot",
                      "fallback_if_no_futures": "spot_proxy_verdict_labelled"}
REGISTERED_VARIANTS = {
    "fixed": {"weights": "fixed", "spreads": ["crack321", "brent_wti", "ho_rb"], "gated": True},
    "johansen": {"weights": "johansen_first_vector", "spreads": ["crack321"], "gated": False, "overrides": {}},
}


def _subset(block: dict, registered: dict) -> dict:
    return {k: block.get(k) for k in registered}


def test_gates_unchanged():
    cfg = load_config()
    assert cfg["registered_on"] == REGISTERED_ON
    assert cfg["gates"] == REGISTERED_GATES
    assert cfg["evaluation"]["sub_periods"] == [            # G3
        ["2007-01-01", "2013-12-31"], ["2014-01-01", "2019-12-31"], ["2020-01-01", None]]
    assert cfg["robustness"]["entry_z"] == [1.5, 2.0, 2.5]  # G4


def test_base_rules_unchanged():
    cfg = load_config()
    assert _subset(cfg["strategy"], REGISTERED_STRATEGY) == REGISTERED_STRATEGY
    assert _subset(cfg["model"], REGISTERED_MODEL) == REGISTERED_MODEL
    assert _subset(cfg["sizing"], REGISTERED_SIZING) == REGISTERED_SIZING


def test_spreads_unchanged():
    assert load_config()["spreads"] == REGISTERED_SPREADS


def test_regime_filter_unchanged():
    """Features are frozen: none may be added or dropped after seeing 2022 or 2026 results."""
    assert _subset(load_config()["regime"], REGISTERED_REGIME) == REGISTERED_REGIME


def test_binding_data_rule_unchanged():
    assert load_config()["binding_data"] == REGISTERED_BINDING


def test_variants_unchanged():
    """Exactly two pre-registered variants; only `fixed` is gated (see PREREGISTRATION.md)."""
    v = load_config()["variants"]
    assert set(v) == set(REGISTERED_VARIANTS)
    for name, spec in REGISTERED_VARIANTS.items():
        assert v[name]["registered_on"] == REGISTERED_ON
        assert _subset(v[name], spec) == spec


def test_gate_windows_exist():
    """G5 refers to stress windows by name; they must be defined in evaluation.stress_windows."""
    cfg = load_config()
    assert set(cfg["gates"]["stress_filter"]["windows"]) <= set(cfg["evaluation"]["stress_windows"])
    assert cfg["evaluation"]["stress_windows"]["ukraine"] == ["2022-01-01", "2022-12-31"]
    assert cfg["evaluation"]["stress_windows"]["hormuz"] == ["2026-01-01", None]
