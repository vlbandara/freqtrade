"""
Tests for the sessionkit feature / signal library.

Run from the repo root:
    pytest user_data/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "strategies"))
sys.path.insert(0, str(HERE))

from sessionkit.features import (  # noqa: E402
    anchored_vwap,
    atr,
    cme_weekend_anchor,
    detect_sweep,
    equal_levels,
    hourly_seasonality,
    in_window,
    opening_range,
    prior_day_levels,
    prior_week_levels,
    rel_volume_tod,
    return_z,
    session_flags,
    session_range,
    swing_points,
)
from sessionkit.signals import (  # noqa: E402
    ALL_MODULES,
    Params,
    build_features,
    compute_signals,
)
from synthetic import make_funding, make_minute_paths, resample  # noqa: E402


@pytest.fixture(scope="module")
def market():
    # 60 days from Feb 3: spans the US DST switch (Mar 9) and leaves 4h EMA / percentile warm-up
    m = make_minute_paths(start="2025-02-03", days=60, seed=3)
    eth5 = resample(m["ETH/USDT:USDT"], "5min")
    return {
        "eth5": eth5,
        "eth1h": resample(m["ETH/USDT:USDT"], "1h"),
        "eth4h": resample(m["ETH/USDT:USDT"], "4h"),
        "btc5": resample(m["BTC/USDT:USDT"], "5min"),
        "funding": make_funding(eth5["date"]),
    }


def _all_on() -> Params:
    p = Params()
    p.enabled = {k: True for k in ALL_MODULES}
    return p


def _pipeline(mk, k=None):
    """Run features + signals. With ``k``, every input is cut to what was KNOWN when the
    k-th 5m candle closed: informative candles must have closed, funding must be settled."""
    df5, df1h, df4h, btc5, fund = mk["eth5"], mk["eth1h"], mk["eth4h"], mk["btc5"], mk["funding"]
    if k is not None:
        last_open = df5["date"].iloc[k - 1]
        t_close = last_open + pd.Timedelta(minutes=5)
        df5 = df5.iloc[:k].copy()
        btc5 = btc5[btc5["date"] <= last_open]
        df1h = df1h[df1h["date"] + pd.Timedelta(hours=1) <= t_close]
        df4h = df4h[df4h["date"] + pd.Timedelta(hours=4) <= t_close]
        fund = fund[fund["date"] <= last_open]
    p = _all_on()
    f = build_features(df5, 5, df1h, fund, btc5, p, df4h=df4h)
    return compute_signals(f, p)


# ---------------------------------------------------------------------------
# Look-ahead invariance: the core guarantee of the library.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cut",
    [
        "2025-03-20 03:35",  # inside the Asia session
        "2025-03-20 09:40",  # London window, Asia range just finalized
        "2025-03-20 13:45",  # inside the NY opening range (EDT)
        "2025-03-22 14:00",  # weekend
        "2025-03-31 23:55",  # last candle before the UTC day roll
        "2025-03-23 21:55",  # CME reopen candle (17:00 CDT)
        "2025-03-24 08:55",  # top of the hour: seasonality lookup + 4h boundary
        "2025-03-26 11:50",  # mid-week: prior-week levels, equal highs/lows
    ],
)
def test_no_lookahead(market, cut):
    full = _pipeline(market)
    k = int((market["eth5"]["date"] <= pd.Timestamp(cut, tz="UTC")).sum())
    part = _pipeline(market, k)
    assert len(part) == k
    cols = [c for c in part.columns if c != "date"]
    a = full.iloc[:k][cols].reset_index(drop=True)
    b = part[cols].reset_index(drop=True)
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            ok = np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9)
        else:
            ok = (x == y) | (x.isna() & y.isna())
        assert bool(np.all(ok)), f"column {c} differs between full and truncated data at {cut}"


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, label: str):
    for c in b.columns:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            ok = np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9)
        else:
            ok = (x == y) | (x.isna() & y.isna())
        assert bool(np.all(ok)), f"{label}: column {c} differs"


def test_feature_functions_are_causal_at_many_cuts(market):
    """Dense check (40 random cuts) straight on the feature functions: sparse events such
    as swing pivots or equal-high pools would rarely sit on one of the pipeline cuts."""
    df5, df1h = market["eth5"], market["eth1h"]
    a5 = atr(df5)
    funcs = {
        "swing_points": lambda d, h: swing_points(d, 3),
        "equal_levels": lambda d, h: equal_levels(d, atr(d)),
        "return_z": lambda d, h: return_z(d["close"]).to_frame(),
        "rel_volume_tod": lambda d, h: rel_volume_tod(d).to_frame(),
        "cme_weekend_anchor": lambda d, h: cme_weekend_anchor(d),
        "prior_week_levels": lambda d, h: prior_week_levels(d, h),
        "hourly_seasonality": lambda d, h: hourly_seasonality(d, h, min_obs=3),
    }
    full = {name: f(df5, df1h) for name, f in funcs.items()}
    rng = np.random.default_rng(0)
    for k in sorted(rng.integers(3000, len(df5), 40)):
        t_close = df5["date"].iloc[k - 1] + pd.Timedelta(minutes=5)
        h_k = df1h[df1h["date"] + pd.Timedelta(hours=1) <= t_close]
        for name, f in funcs.items():
            _assert_same(full[name].iloc[:k], f(df5.iloc[:k].copy(), h_k), f"{name} @k={k}")
    assert a5.notna().any()


def test_all_modules_fire_on_synthetic(market):
    out = _pipeline(market)
    tags = set(out["enter_tag"].dropna())
    # sweeps / squeezes are rare by design; these must fire on 60 days of data
    assert {"ny_orb", "late_mom", "wknd_vwap", "fund_window", "eq_sweep", "trend_4h"} <= tags
    trend = out[out["enter_tag"] == "trend_4h"]
    assert trend["sig_trail"].gt(0).all()
    assert out.loc[out["enter_tag"] != "trend_4h", "sig_trail"].isna().all()
    sig = out[out["enter_tag"].notna()]
    long_ = sig["enter_long"] == 1
    assert (sig.loc[long_, "sig_stop"] < sig.loc[long_, "close"]).all()
    assert (sig.loc[~long_, "sig_stop"] > sig.loc[~long_, "close"]).all()
    assert (sig["sig_deadline"] > sig["date"].astype("int64") // 10**9).all()
    assert not ((out["enter_long"] == 1) & (out["enter_short"] == 1)).any()


# ---------------------------------------------------------------------------
# Sessions & DST
# ---------------------------------------------------------------------------
def test_ny_window_follows_dst():
    dates = pd.Series(
        pd.to_datetime(["2025-03-07 14:30", "2025-03-10 13:30", "2025-03-10 14:30"], utc=True)
    )
    # 09:30 New York is 14:30 UTC in EST (Mar 7) and 13:30 UTC in EDT (Mar 10)
    assert in_window(dates, "America/New_York", "09:30", "10:00").tolist() == [True, True, False]


def test_session_flags_weekend_and_asia():
    dates = pd.Series(
        pd.to_datetime(["2025-03-08 03:00", "2025-03-10 06:55", "2025-03-10 07:00"], utc=True)
    )
    f = session_flags(pd.DataFrame({"date": dates}))
    assert f["is_weekend"].tolist() == [True, False, False]
    assert f["sess_asia"].tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------
def _frame(dates, o, h, lo, c, v=None):
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates, utc=True),
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": v if v is not None else np.ones(n),
        }
    )


def test_session_range_only_after_session():
    dates = pd.date_range("2025-03-10 05:00", "2025-03-10 09:00", freq="1h", tz="UTC")
    df = _frame(dates, [1] * 5, [2, 5, 3, 9, 9], [0.5, 0.2, 1, 1, 1], [1] * 5)
    asia = session_flags(df)["sess_asia"]
    day = df["date"].dt.tz_localize(None).dt.floor("D")
    r = session_range(df, asia, day, min_bars=2)
    assert r["high"].isna().tolist() == [True, True, False, False, False]
    assert r["high"].iloc[2] == 5 and r["low"].iloc[2] == 0.2  # 07:00 high (3) not included


def test_prior_day_levels_need_complete_previous_day():
    dates = pd.date_range("2025-03-10 00:00", "2025-03-12 23:00", freq="1h", tz="UTC")
    n = len(dates)
    h = np.arange(n, dtype=float) + 1
    df = _frame(dates, h - 0.5, h, h - 1, h - 0.2)
    lv = prior_day_levels(df, tf_minutes=60)
    assert lv["pdh"].iloc[:24].isna().all()
    assert lv["pdh"].iloc[24] == 24.0 and lv["pdl"].iloc[24] == 0.0
    assert lv["day_open"].iloc[30] == df["open"].iloc[24]
    # drop half of day 2 -> day 3 has no PDH
    lv2 = prior_day_levels(df.drop(index=range(30, 42)).reset_index(drop=True), tf_minutes=60)
    assert lv2["pdh"].iloc[-1] != lv2["pdh"].iloc[-1]  # NaN


def test_anchored_vwap_resets_and_masks():
    dates = pd.date_range("2025-03-10 22:00", periods=4, freq="1h", tz="UTC")
    df = _frame(
        dates, [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [1, 3, 1, 1]
    )
    day = df["date"].dt.tz_localize(None).dt.floor("D")
    v = anchored_vwap(df, day)
    assert v["vwap"].tolist() == [10, 10, 20, 20]
    vm = anchored_vwap(df, day, mask=pd.Series([True, False, True, True]))
    assert np.isnan(vm["vwap"].iloc[1]) and vm["vwap"].iloc[3] == 20


def test_rel_volume_tod_uses_only_past_same_slot():
    dates = pd.date_range("2025-03-01", periods=24 * 12, freq="1h", tz="UTC")
    vol = np.where(dates.hour == 14, 10.0, 1.0)
    vol[-1] = 30.0  # last candle is 23:00 -> normal slot, but 30x busier
    df = _frame(dates, 1.0, 1.0, 1.0, 1.0, vol)
    rv = rel_volume_tod(df, lookback_days=5, min_days=3)
    assert rv.iloc[-1] == pytest.approx(30.0)
    assert rv[(dates.hour == 14)].dropna().eq(1.0).all()  # busy slot is normal for that slot


def test_opening_range_valid_window():
    dates = pd.date_range("2025-03-10 13:30", "2025-03-10 20:30", freq="5min", tz="UTC")
    n = len(dates)
    df = _frame(dates, np.ones(n), np.ones(n) * 2, np.ones(n) * 0.5, np.ones(n))
    df.loc[2, "high"] = 3.0  # inside the 09:30-10:00 EDT window
    df.loc[20, "high"] = 9.0  # after the window, must not change the range
    r = opening_range(df, tf_minutes=5)
    assert r["high"].iloc[:6].isna().all()
    assert (r["high"].dropna() == 3.0).all()
    assert r["high"].iloc[-1] != r["high"].iloc[-1]  # after 16:00 EDT -> NaN


# ---------------------------------------------------------------------------
# Sweep detection
# ---------------------------------------------------------------------------
def test_detect_sweep_wick_and_reclaim():
    dates = pd.date_range("2025-03-10 08:00", periods=8, freq="5min", tz="UTC")
    o = [99.0, 99.5, 99.6, 99.4, 99.0, 98.0, 98.0, 99.0]
    h = [99.8, 99.9, 101.0, 99.8, 99.3, 98.5, 98.5, 102.0]
    lo = [98.8, 99.2, 99.3, 98.9, 98.1, 97.5, 97.5, 98.8]
    c = [99.5, 99.6, 99.4, 99.0, 98.2, 98.0, 98.0, 99.0]
    df = _frame(dates, o, h, lo, c)
    level_hi = pd.Series(100.0, index=df.index)
    level_lo = pd.Series(90.0, index=df.index)
    atr_ = pd.Series(1.0, index=df.index)
    day = df["date"].dt.tz_localize(None).dt.floor("D")
    sw = detect_sweep(df, level_hi, level_lo, atr_, day)
    # bar 2 trades to 101 and closes back below 100 with a long upper wick -> short
    assert sw["sweep_short"].tolist() == [False, False, True, False, False, False, False, False]
    assert sw["sweep_ext_high"].iloc[2] == 101.0
    # bar 7 breaches again, but only the first sweep of the day counts
    assert not sw["sweep_long"].any()


def test_detect_sweep_rejects_breakouts():
    dates = pd.date_range("2025-03-10 08:00", periods=4, freq="5min", tz="UTC")
    # breaks 100 by 3 ATR then comes back: too far -> breakout, not a sweep
    df = _frame(
        dates,
        [99, 101, 103, 101],
        [99.5, 103, 103.2, 101.5],
        [98.5, 100.5, 101, 99],
        [99, 102.5, 101, 99.5],
    )
    day = df["date"].dt.tz_localize(None).dt.floor("D")
    one = pd.Series(1.0, index=df.index)
    sw = detect_sweep(df, one * 100, one * 90, one, day)
    assert not sw["sweep_short"].any()


# ---------------------------------------------------------------------------
# Second-wave features
# ---------------------------------------------------------------------------
def test_swing_points_confirmed_k_bars_later():
    dates = pd.date_range("2025-03-10", periods=9, freq="5min", tz="UTC")
    h = [1, 2, 3, 9, 3, 2, 1, 1, 1]
    df = _frame(dates, h, h, [x - 0.5 for x in h], h)
    sw = swing_points(df, k=3)
    assert sw["swing_high"].notna().tolist() == [False] * 6 + [True, False, False]
    assert sw["swing_high"].iloc[6] == 9  # pivot at bar 3, known at bar 6


def test_equal_highs_pool_detected_and_untouched_rule():
    # two swing highs at 10.00 and 10.02 (within 0.1 ATR of 1.0), nothing above in between
    h = [8, 9, 10.0, 9, 8, 7, 8, 9, 10.02, 9, 8, 7, 7, 7]
    dates = pd.date_range("2025-03-10", periods=len(h), freq="5min", tz="UTC")
    df = _frame(dates, h, h, [x - 0.5 for x in h], h)
    eq = equal_levels(df, pd.Series(1.0, index=df.index), k=2, lookback=50, tol_atr=0.1)
    assert eq["eqh"].iloc[10] == pytest.approx(10.02)  # second pivot (bar 8) confirmed at bar 10
    assert eq["eqh"].iloc[:10].isna().all()
    # a higher high between the pivots means the pool was already taken -> no level
    h2 = h.copy()
    h2[5] = 11.0
    df2 = _frame(dates, h2, h2, [x - 0.5 for x in h2], h2)
    eq2 = equal_levels(df2, pd.Series(1.0, index=df.index), k=2, lookback=50, tol_atr=0.1)
    assert eq2["eqh"].isna().all()


def test_cme_anchor_follows_chicago_dst():
    # Friday close 16:00 CT = 22:00 UTC in CST (Mar 7), Sunday reopen 17:00 CT = 22:00 UTC
    # in CDT (Mar 9, the day the US switched)
    dates = pd.date_range("2025-03-07 21:00", "2025-03-09 23:00", freq="5min", tz="UTC")
    n = len(dates)
    close = np.arange(n, dtype=float)
    df = _frame(dates, close, close + 1, close - 1, close)
    cme = cme_weekend_anchor(df)
    fri_idx = int(np.flatnonzero(dates == pd.Timestamp("2025-03-07 21:55", tz="UTC"))[0])
    assert cme["cme_fri_close"].iloc[fri_idx] == close[fri_idx]
    assert cme["cme_fri_close"].iloc[:fri_idx].isna().all()
    reopen = dates[cme["cme_reopen"].to_numpy()]
    assert list(reopen) == [pd.Timestamp("2025-03-09 21:55", tz="UTC")]
    assert cme["cme_fri_close"].iloc[-1] == close[fri_idx]
