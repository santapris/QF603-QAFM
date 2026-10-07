import numpy as np
import pandas as pd
import pytest

from src.data import load_bloomberg, load_hkm
from src.measures.vrp import FORWARD_COLUMNS, assemble
from src.utils.dates import month_end_trading_days, trading_days
from src.utils.io import load_config


@pytest.fixture
def cfg():
    c = load_config()
    c["har"]["headline"] = "log"
    return c


def synthetic_inputs(cfg, seed=0):
    rng = np.random.default_rng(seed)
    days = trading_days("2007-06-01", cfg["sample"]["end"])
    me = month_end_trading_days("2007-12-01", cfg["sample"]["end"])
    rv = pd.Series(1e-4 * rng.chisquare(3, len(days)) / 3, index=days)
    iv = pd.Series(0.04 + 0.01 * rng.standard_normal(len(me)) ** 2, index=me)
    har = pd.DataFrame({f"har_{v}": 0.03 + 0.005 * rng.random(len(me)) for v in ["level", "log", "iv"]}, index=me)
    daily = lambda lvl: pd.Series(lvl * np.exp(np.cumsum(0.01 * rng.standard_normal(len(days)))), index=days)  # noqa: E731
    bbg = pd.DataFrame({"vix3m": daily(22), "vvix": daily(90), "skew": daily(130), "spxt": daily(3000),
                        "put": daily(1000), "bxm": daily(2000)})
    fred = pd.DataFrame({c: rng.random(len(me)) * 0.01 for c in ["fund_spread", "fund_spread_nonfin", "credit_spread", "rf"]}, index=me)
    cftc = pd.DataFrame({"cftc_pos": rng.standard_normal(len(me)), "cftc_pos_lev": rng.standard_normal(len(me))}, index=me)
    hkm = pd.DataFrame({"hkm": rng.random(len(me)), "hkm_lag1": rng.random(len(me))}, index=me)
    return dict(iv=iv, har=har, rv=rv, vix_daily=daily(20), bbg=bbg, fred=fred, cftc=cftc, hkm=hkm)


def perturb_after(obj, t):
    obj = obj.copy()
    obj.loc[obj.index > t] = obj.loc[obj.index > t] * 3 + 1
    return obj


def test_panel_has_no_lookahead(cfg):
    inputs = synthetic_inputs(cfg)
    base = assemble(cfg, **inputs)
    t = pd.Timestamp("2012-06-29")
    changed = assemble(cfg, **{k: perturb_after(v, t) for k, v in inputs.items()})
    info_cols = [c for c in base.columns if c not in FORWARD_COLUMNS]
    pd.testing.assert_frame_equal(base.loc[:t, info_cols], changed.loc[:t, info_cols])
    # forward-looking columns do change at t (they are targets, known later)
    assert not np.isclose(base.loc[t, "exret_1"], changed.loc[t, "exret_1"])


def test_panel_units_and_forward_returns(cfg):
    inputs = synthetic_inputs(cfg, seed=1)
    p = assemble(cfg, **inputs)
    assert p.index.min() == pd.Timestamp(cfg["sample"]["start"])
    assert np.allclose(p["vrp_exante"], p["iv"] - p["har_fcst"], equal_nan=True)
    # 3-month excess return = sum of three 1-month excess returns
    s = p["exret_1"] + p["exret_1"].shift(-1) + p["exret_1"].shift(-2)
    assert np.allclose(p["exret_3"].dropna(), s.loc[p["exret_3"].dropna().index])


def test_bloomberg_loader_validates(tmp_path, cfg):
    days = pd.bdate_range("2007-01-01", cfg["sample"]["end"])
    for name, ticker in load_bloomberg.REQUIRED.items():
        pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "px_last": 10.0}).to_csv(
            tmp_path / load_bloomberg.filename(ticker), index=False)
    df, missing = load_bloomberg.load(cfg, directory=tmp_path)
    assert set(load_bloomberg.REQUIRED) <= set(df.columns) and not missing
    pd.DataFrame({"Date": ["2020-01-02"], "PX_LAST": [-1]}).to_csv(tmp_path / "VIX_Index.csv", index=False)
    with pytest.raises(ValueError):
        load_bloomberg.load(cfg, directory=tmp_path)
    (tmp_path / "VIX_Index.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_bloomberg.load(cfg, directory=tmp_path)


def test_hkm_loader_applies_lag(tmp_path, cfg):
    pd.DataFrame({"yyyymm": [201712, 201801, 201802], "intermediary_capital_ratio": [0.05, 0.06, 0.07],
                  "intermediary_capital_risk_factor": [0, 0, 0]}).to_csv(tmp_path / "HKM_monthly.csv", index=False)
    df = load_hkm.build(cfg, directory=tmp_path)
    assert df["hkm"].dropna().index[0] == pd.Timestamp("2018-03-29")      # Dec 2017 value usable end-Mar 2018
    assert df["hkm_lag1"].dropna().index[0] == pd.Timestamp("2018-01-31")
