"""Part 5 tests: episode labelling with frozen parameters, re-arming, censoring, point-in-time
logit refits, and the evaluation helpers."""
import numpy as np
import pandas as pd
import pytest

from oilrv import regime as R
from oilrv.config import load_config


# ------------------------------------------------------------------ labels
@pytest.mark.parametrize("z_fwd,expected", [
    ([1.8, 1.4, 0.9], (1, 3, "reverted")),
    ([2.5, 3.9, 4.0], (0, 3, "stopped")),
    ([-1.5, -1.0], (1, 2, "reverted")),        # |z| <= 1 counts, either side
    ([1.5, 1.6, 1.7], (0, 3, "timed_out")),
])
def test_label_episode(z_fwd, expected):
    assert R.label_episode(np.array(z_fwd), 1.0, 4.0) == expected


# ------------------------------------------------------------------ episodes on a hand-built path
def _setup(S):
    """Brent - WTI = S exactly; a single frozen fit with mean 0, sd 1, no month effects -> z = S."""
    idx = pd.bdate_range("2020-01-01", periods=len(S))
    sp = pd.DataFrame({"brent_bbl": 50 + np.asarray(S, float), "cl_bbl": 50.0}, index=idx)
    fit = pd.Series({"w_brent": 1.0, "w_cl": -1.0, "mean": 0.0, "sd": 1.0,
                     **{f"seas_{m}": 0.0 for m in range(1, 13)}}, name=idx[0])
    fits = pd.DataFrame([fit])
    path = pd.DataFrame({"z": np.asarray(S, float), "fit_date": idx[0], "gate": True, "half_life": 10.0}, index=idx)
    return sp, fits, path


def test_episodes_rearm_no_overlap_and_frozen_labels():
    cfg = load_config()
    S = [0, 0, 2.5, 1.5, 0.8, 0, 3.0, 3.5, 4.2, 3.0, 2.5, 1.5, 2.2, 1.9, 0.5] + [0] * 25
    sp, fits, path = _setup(S)
    ep = R.build_episodes("brent_wti", sp, fits, path, horizon=20, cfg=cfg)
    d = sp.index
    assert list(ep.index) == [d[2], d[6], d[12]]
    assert ep.loc[d[2], ["reverted", "outcome", "days"]].tolist() == [1, "reverted", 2]
    assert ep.loc[d[2], "resolved_on"] == d[4]
    assert ep.loc[d[6], ["reverted", "outcome", "days"]].tolist() == [0, "stopped", 2]
    # after the stop on d[8], |z| stays >= 2 until d[11] (1.5): no new episode at d[10] (2.5)
    assert d[10] not in ep.index
    assert ep.loc[d[12], ["reverted", "days"]].tolist() == [1, 2]
    assert (ep["side"] == -1).all()


def test_unresolved_episode_at_the_end_is_dropped():
    cfg = load_config()
    sp, fits, path = _setup([0, 0, 0, 2.5, 2.4, 2.3])
    assert R.build_episodes("brent_wti", sp, fits, path, horizon=20, cfg=cfg).empty
    sp, fits, path = _setup([0, 0, 0, 2.5, 2.4, 0.5])
    assert len(R.build_episodes("brent_wti", sp, fits, path, horizon=20, cfg=cfg)) == 1


def test_frozen_z_uses_signal_day_fit_not_later_fits():
    """Labels must use the parameters in force on the signal day, even if the path refits later."""
    cfg = load_config()
    S = [0, 0, 2.5] + [1.5] * 30
    sp, fits, path = _setup(S)
    later = fits.iloc[0].copy()
    later.name = sp.index[3]
    later["mean"] = 1.2                                    # under the later fit, 1.5 would be z = 0.3
    fits2 = pd.concat([fits, later.to_frame().T])
    path2 = path.copy()
    path2.loc[sp.index[3]:, "fit_date"] = sp.index[3]
    ep = R.build_episodes("brent_wti", sp, fits2, path2, horizon=20, cfg=cfg)
    assert ep.iloc[0]["outcome"] == "timed_out"            # frozen z stays at 1.5


# ------------------------------------------------------------------ features
def test_trailing_percentile():
    x = pd.Series([1.0, 2, 3, 4, 2])
    p = R.trailing_percentile(x, 4, min_periods=4)
    assert np.isnan(p.iloc[2]) and p.iloc[3] == 1.0 and p.iloc[4] == pytest.approx(0.5)


# ------------------------------------------------------------------ point-in-time refits
def _synthetic_episodes(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n * 10)[::10]
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * x))).astype(int)
    ep = pd.DataFrame({"reverted": y, "x": x, "d_brent_wti": 0.0, "d_ho_rb": 0.0,
                       "resolved_on": dates + pd.Timedelta(days=15), "spread": "crack321",
                       "key": "fixed|crack321"}, index=dates)
    days = pd.bdate_range(dates[0], dates[-1] + pd.Timedelta(days=60))
    daily = pd.DataFrame({"x": np.sin(np.arange(len(days)) / 30), "d_brent_wti": 0.0, "d_ho_rb": 0.0}, index=days)
    return ep, {"fixed|crack321": daily}


def test_regime_probability_uses_only_resolved_episodes():
    ep, daily = _synthetic_episodes()
    feats = ["x", "d_brent_wti", "d_ho_rb"]
    p, coefs = R.regime_probability(ep, daily, feats, min_episodes=20)
    m = pd.Timestamp("2014-03-03")
    # flip every label resolved on or after m: P before and in month m must not change
    ep2 = ep.copy()
    late = ep2["resolved_on"] >= m
    ep2.loc[late, "reverted"] = 1 - ep2.loc[late, "reverted"]
    p2, _ = R.regime_probability(ep2, daily, feats, min_episodes=20)
    a, b = p["fixed|crack321"], p2["fixed|crack321"]
    m_end = a.index[a.index.to_period("M") == m.to_period("M")][-1]
    pd.testing.assert_series_equal(a.loc[:m_end, "p_hat"], b.loc[:m_end, "p_hat"])
    assert not a["p_hat"].loc[m_end:].iloc[30:].equals(b["p_hat"].loc[m_end:].iloc[30:])
    # no filter until 20 resolved episodes; the coefficient on x is positive once estimated
    assert a.loc[a["n_train"] < 20, "p_hat"].isna().all()
    assert a.loc[a["p_hat"].first_valid_index(), "n_train"] >= 20
    assert coefs["x"].iloc[-1] > 0


def test_fit_model_returns_none_when_unfittable():
    ep, _ = _synthetic_episodes(30)
    ep["reverted"] = 1
    assert R.fit_model(ep, ["x"]) is None


# ------------------------------------------------------------------ evaluation
def test_auc_and_evaluate_filter():
    y = np.array([0, 0, 1, 1])
    assert R.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert R.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    idx = pd.bdate_range("2020-01-01", periods=4)
    ep = pd.DataFrame({"reverted": [1, 0, 1, 0], "p_hat": [np.nan, 0.7, 0.8, 0.3],
                       "resolved_on": idx + pd.Timedelta(days=1)}, index=idx)
    r = R.evaluate_filter(ep, 0.6)
    assert r["episodes_with_p"] == 3 and r["n_pass_filter"] == 2
    assert r["success_pass_filter"] == 0.5 and r["success_rejected"] == 0.0
