"""Part 1 tests: parsing, units, point-in-time stamping, futures rolls, calendars. Offline except the
last block, which checks the built data files and is skipped if they are missing."""
import io

import numpy as np
import pandas as pd
import pytest

from oilrv import futures
from oilrv.calendars import (calendar_days_since, event_features, gulf_hurricane_landfalls,
                             trading_days_to_next)
from oilrv.config import load_config, path
from oilrv.data import (_parse_cftc, _parse_eia_xls, _parse_fred_csv, asof_daily, five_year_deviation,
                        flag_spikes)
from oilrv.spreads import build_spreads, usd_per_bbl


# ------------------------------------------------------------------ parsing
def test_fred_csv_missing_markers():
    txt = "observation_date,DCOILWTICO\n2020-04-17,18.31\n2020-04-20,-36.98\n2020-04-21,.\n2020-04-22,\n"
    s = _parse_fred_csv(txt, "DCOILWTICO")
    assert s.isna().tolist() == [False, False, True, True]
    assert s.iloc[1] == -36.98                      # negative prices survive parsing


def test_eia_xls_layout():
    buf = io.BytesIO()
    rows = [["Back to Contents", "Data 1: NYH ULSD ($/gal)"], ["Sourcekey", "EER_X"], ["Date", "NYH ULSD ($/gal)"],
            [pd.Timestamp("2026-10-19"), 5.173], [pd.Timestamp("2026-10-20"), 5.265]]
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame([["contents"]]).to_excel(w, sheet_name="Contents", header=False, index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="Data 1", header=False, index=False)
    buf.seek(0)
    s = _parse_eia_xls(buf, "ho")
    assert s.index.tolist() == [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20")]
    assert s.tolist() == [5.173, 5.265] and s.name == "ho"


def test_cftc_managed_money_net():
    f = pd.DataFrame({
        "CFTC_Contract_Market_Code": ["067651", "111659", "022651", "999999"],
        "Report_Date_as_YYYY-MM-DD": ["2026-10-20"] * 4,
        "Open_Interest_All": [100, 200, 300, 1], "M_Money_Positions_Long_All": [50, 40, 30, 1],
        "M_Money_Positions_Short_All": [10, 60, 5, 1]})
    c = _parse_cftc([f])
    row = c.loc[pd.Timestamp("2026-10-20")]
    assert (row["mm_net_cl"], row["mm_net_rb"], row["mm_net_ho"]) == (40, -20, 25)


# ------------------------------------------------------------------ spreads and units
def test_spreads_match_plan_example():
    """EIA spot on 20 Oct 2026 (oil-rv-plan.md): crack about $47, diesel crack about $113."""
    px = pd.DataFrame({"cl": [107.89], "brent": [115.22], "rb": [2.897], "ho": [5.265]},
                      index=[pd.Timestamp("2026-10-20")])
    sp = build_spreads(px).iloc[0]
    assert sp["crack321"] == pytest.approx((2 * 42 * 2.897 + 42 * 5.265 - 3 * 107.89) / 3)
    assert sp["crack321"] == pytest.approx(46.94, abs=0.01)
    assert sp["brent_wti"] == pytest.approx(7.33)
    assert sp["ho_rb"] == pytest.approx(42 * (5.265 - 2.897))
    assert sp["ho_bbl"] - sp["cl_bbl"] == pytest.approx(113.24, abs=0.01)


def test_dollar_multipliers_match_contract_pnl():
    """1 unit = the contracts in `legs`, 1,000 bbl each, barrel-neutral. Its P&L summed leg by leg
    must equal M x (change in the spread), with M = usd_per_1bbl_per_unit."""
    cfg = load_config()
    bbl = cfg["futures"]["contract_bbl"]
    before = pd.DataFrame({"cl": [80.0], "brent": [84.0], "rb": [2.40], "ho": [2.90]})
    after = pd.DataFrame({"cl": [81.3], "brent": [84.9], "rb": [2.47], "ho": [2.85]})
    s0, s1 = build_spreads(before).iloc[0], build_spreads(after).iloc[0]
    for name, spec in cfg["spreads"].items():
        if not spec.get("traded"):
            continue
        assert sum(spec["legs"].values()) == 0                       # barrel-neutral
        leg_pnl = sum(w * bbl * (s1[f"{leg}_bbl"] - s0[f"{leg}_bbl"]) for leg, w in spec["legs"].items())
        assert leg_pnl == pytest.approx(usd_per_bbl(name) * (s1[name] - s0[name]))


# ------------------------------------------------------------------ point in time
def test_weekly_stocks_available_thursday_after_week_end():
    wk = pd.Series([100.0, 200.0], index=pd.DatetimeIndex(["2026-10-09", "2026-10-16"]))    # Fridays
    days = pd.bdate_range("2026-10-19", "2026-10-23")
    lag = load_config()["data"]["weekly_stock_lag_days"]
    x = asof_daily(wk, wk.index + pd.Timedelta(days=lag), days)
    assert x.loc["2026-10-21"] == 100.0             # Wednesday: release day, not yet usable
    assert x.loc["2026-10-22"] == 200.0             # Thursday


def test_cot_available_monday_after_friday_release():
    cot = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-10-13", "2026-10-20"]))     # Tuesdays
    days = pd.bdate_range("2026-10-19", "2026-10-27")
    x = asof_daily(cot, cot.index + pd.Timedelta(days=load_config()["data"]["cftc_release_lag_days"]), days)
    assert x.loc["2026-10-23"] == 1.0                       # Friday: released after the settle
    assert x.loc["2026-10-26"] == 2.0                       # Monday close


def test_five_year_deviation_uses_past_only():
    idx = pd.date_range("2010-01-01", periods=52 * 8, freq="7D")
    w = pd.Series(100.0, index=idx, name="stocks")
    w.iloc[-1] = 110.0
    dev = five_year_deviation(w)
    assert dev.iloc[: 52 * 5].isna().all()          # needs five prior years
    assert dev.iloc[-2] == pytest.approx(0.0) and dev.iloc[-1] == pytest.approx(10.0)
    w2 = w.copy()
    w2.iloc[-10:] = 999.0                            # perturb the future
    assert five_year_deviation(w2).iloc[: -10].equals(dev.iloc[: -10])


def test_spike_flag_catches_reversing_outlier_only():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
    x.iloc[200] += 30                               # one-day spike that reverses
    x.iloc[250:] += 30                              # a permanent jump: not a spike
    flags = flag_spikes(x, 5.0, 0.8, 60)
    assert flags.iloc[200] and not flags.iloc[250] and flags.sum() == 1


# ------------------------------------------------------------------ futures
def test_parse_contract_resolves_decade():
    assert futures.parse_contract("CLZ6", pd.Timestamp("2016-11-01")) == ("CL", pd.Period("2016-12", "M"))
    assert futures.parse_contract("CLZ6 Comdty", pd.Timestamp("2026-11-01")) == ("CL", pd.Period("2026-12", "M"))
    assert futures.parse_contract("CLF7", pd.Timestamp("2026-12-15")) == ("CL", pd.Period("2027-01", "M"))
    assert futures.parse_contract("COH16", pd.Timestamp("2015-12-01")) == ("CO", pd.Period("2016-03", "M"))
    with pytest.raises(ValueError):
        futures.parse_contract("CL1 Comdty", pd.Timestamp("2016-01-01"))


@pytest.mark.parametrize("root,month,expected", [
    ("CL", "2020-05", "2020-04-21"),    # the contract that settled at -$37.63
    ("CL", "2026-12", "2026-11-20"),
    ("XB", "2026-12", "2026-11-30"),
    ("HO", "2026-12", "2026-11-30"),
    ("CO", "2026-12", "2026-10-30"),    # last business day of the second month before delivery
    ("CO", "2015-12", "2015-11-13"),    # pre-2016 rule
])
def test_expiry_rules(root, month, expected):
    assert futures.expiry(root, pd.Period(month, "M")) == pd.Timestamp(expected)


def _synthetic_panels():
    dates = pd.bdate_range("2025-11-03", "2026-03-31")
    months = pd.period_range("2026-01", "2026-08", freq="M")
    out = {}
    for leg, root, base in [("brent", "CO", 65.0), ("cl", "CL", 60.0)]:
        p = pd.DataFrame({m: base + 1.5 * i + 0.1 * np.arange(len(dates)) + np.sin(np.arange(len(dates)) + i)
                          for i, m in enumerate(months)}, index=dates)
        for m in months:
            p.loc[p.index > futures.expiry(root, m), m] = np.nan
        out[leg] = p
    return out, {"brent": "CO", "cl": "CL"}


def test_continuous_same_delivery_month_and_back_adjustment():
    panels, roots = _synthetic_panels()
    c = futures.continuous(panels, roots, k=1, roll_bdays=5)
    h = c["delivery"].map(lambda s: pd.Period(s, "M"))
    # Brent expires first, so it sets the joint roll date for the Feb contract
    roll_feb = futures.expiry("CO", pd.Period("2026-02", "M")) - 5 * futures.US_BDAY
    assert (h[c.index < roll_feb] <= pd.Period("2026-02", "M")).all()
    assert h.loc[roll_feb] == pd.Period("2026-03", "M") and c.loc[roll_feb, "rolled"]
    for leg in ["brent", "cl"]:
        p = panels[leg].reindex(c.index)
        adj = c[f"{leg}_adj"]
        assert adj.iloc[-1] == pytest.approx(c[leg].iloc[-1])
        for i in range(1, len(c)):
            m = h.iloc[i - 1]
            held_change = p[m].iloc[i] - p[m].iloc[i - 1]
            assert adj.iloc[i] - adj.iloc[i - 1] == pytest.approx(held_change)
        assert (c[leg] == [p.at[t, m] for t, m in zip(c.index, h)]).all()
    assert not c["stale_leg"].any()


def test_continuous_k2_is_next_month():
    panels, roots = _synthetic_panels()
    h1 = futures.held_months(panels, roots, k=1)
    h2 = futures.held_months(panels, roots, k=2)
    assert ((h2 - h1).map(lambda d: d.n) == 1).all()


def test_read_bloomberg_two_row_header(tmp_path):
    f = tmp_path / "CL.csv"
    f.write_text(
        ",CL1 Comdty,CL1 Comdty,CL2 Comdty,CL2 Comdty\n"
        "Dates,PX_SETTLE,FUT_CUR_GEN_TICKER,PX_SETTLE,FUT_CUR_GEN_TICKER\n"
        "2026-10-19,110.01,CLX6,108.50,CLZ6\n"
        "2026-10-20,107.89,CLX6,106.70,CLZ6\n"
        "2026-10-21,105.00,CLZ6,,\n")
    panel, root = futures.contract_panel(futures.read_bloomberg(f))
    assert root == "CL"
    assert panel.loc["2026-10-20", pd.Period("2026-12", "M")] == 106.70
    assert panel.loc["2026-10-21", pd.Period("2026-12", "M")] == 105.00   # same contract, now CL1


# ------------------------------------------------------------------ calendars
HURDAT_SNIPPET = """AL092008,                IKE,     3,
20080913, 0000,  , HU, 29.0N,  94.5W,  95,  952,
20080913, 0700, L, HU, 29.3N,  94.7W,  95,  950,
20080913, 1200,  , HU, 30.3N,  95.2W,  75,  959,
AL102022,                IAN,     2,
20220928, 1905, L, HU, 26.7N,  82.2W, 130,  941,
20220930, 1805, L, HU, 33.2N,  79.2W,  70,  977,
AL052026,            EDOUARD,     1,
20260901, 1920, L, TS, 29.8N,  93.6W,  50,  996,
"""


def test_gulf_landfalls_keep_gulf_hurricanes_only():
    lf = gulf_hurricane_landfalls(HURDAT_SNIPPET)
    assert lf["date"].tolist() == [pd.Timestamp("2008-09-13")]        # Ian (FL west coast, SC) and TS excluded
    assert lf["name"].iloc[0].startswith("Ike")


def test_trading_days_to_next_and_since():
    idx = pd.bdate_range("2026-10-26", "2026-11-06")
    d = trading_days_to_next(idx, pd.DatetimeIndex(["2026-11-01"]))   # Sunday -> Monday 2 Nov
    assert d.loc["2026-10-30"] == 1 and d.loc["2026-10-29"] == 2 and np.isnan(d.loc["2026-11-02"])
    s = calendar_days_since(idx, pd.DatetimeIndex(["2026-10-25"]))
    assert s.loc["2026-10-26"] == 1 and s.loc["2026-11-06"] == 12


def test_blackout_uses_scheduled_opec_and_landfalls_only():
    idx = pd.bdate_range("2026-10-01", "2026-10-30")
    ev = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-04", "2026-10-15", "2026-10-21", "2026-10-28"]),
        "type": ["OPEC", "OPEC", "HURRICANE", "HORMUZ"],
        "name": ["meeting", "surprise", "landfall", "closure: x"],
        "scheduled": [True, False, False, False]})
    f = event_features(idx, ev)
    assert f.loc["2026-10-02", "event_blackout"]           # Friday before Sunday's scheduled meeting
    assert f.loc["2026-10-01", "event_blackout"]           # Thursday decision would fill Friday, the day before
    assert not f.loc["2026-10-14", "event_blackout"]       # unscheduled OPEC: no blackout
    assert f.loc["2026-10-20", "event_blackout"]           # day before landfall
    assert f.loc["2026-10-19", "event_blackout"]           # would fill the day before landfall
    assert not f["event_blackout"].loc["2026-10-22":].any()
    assert f.loc["2026-10-29", "days_since_hormuz_closure"] == 1


# ------------------------------------------------------------------ built data (skipped if not built)
@pytest.fixture(scope="module")
def built():
    f = path("clean_dir") / "spreads.csv"
    if not f.exists():
        pytest.skip("run scripts/build_data.py first")
    sp = pd.read_csv(f, index_col=0, parse_dates=True)
    feats = pd.read_csv(path("clean_dir") / "features.csv", index_col=0, parse_dates=True)
    return sp, feats


def test_built_spreads_clean(built):
    sp, _ = built
    assert sp[["crack321", "brent_wti", "ho_rb"]].notna().all().all()
    assert sp.index.is_monotonic_increasing and not sp.index.duplicated().any()
    assert sp.index[0] >= pd.Timestamp(load_config()["data"]["sample_start"])
    assert 15 < sp["crack321"].median() < 25                        # plan: long-run median about $19
    assert sp.loc["2020-04-20", "flag_apr2020"] and sp.loc["2020-04-20", "cl_bbl"] < 0


def test_built_features_complete_after_2010(built):
    sp, feats = built
    assert feats.index.equals(sp.index)
    core = ["ovx", "gpr", "inventory_dev_5y", "mm_net_cl", "mm_net_rb", "mm_net_ho"]
    assert feats.loc["2010":, core].notna().all().all()
