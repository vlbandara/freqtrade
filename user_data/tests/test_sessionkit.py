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
    detect_sweep,
    in_window,
    opening_range,
    prior_day_levels,
    rel_volume_tod,
    session_flags,
    session_range,
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
    m = make_minute_paths(start="2025-02-24", days=40, seed=3)  # spans the US DST switch (Mar 9)
    return {
        "eth5": resample(m["ETH/USDT:USDT"], "5min"),
        "eth1h": resample(m["ETH/USDT:USDT"], "1h"),
        "btc5": resample(m["BTC/USDT:USDT"], "5min"),
    }


def _all_on() -> Params:
    p = Params()
    p.enabled = {k: True for k in ALL_MODULES}
    return p


def _pipeline(df5, df1h, btc5):
    p = _all_on()
    f = build_features(df5, 5, df1h, make_funding(btc5["date"]), btc5, p)
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
    ],
)
def test_no_lookahead(market, cut):
    full = _pipeline(market["eth5"], market["eth1h"], market["btc5"])
    k = int((market["eth5"]["date"] <= pd.Timestamp(cut, tz="UTC")).sum())
    part = _pipeline(market["eth5"].iloc[:k].copy(), market["eth1h"], market["btc5"])
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


def test_all_modules_fire_on_synthetic(market):
    out = _pipeline(market["eth5"], market["eth1h"], market["btc5"])
    tags = set(out["enter_tag"].dropna())
    # sweeps are rare by design; the remaining modules must fire on 40 days of data
    assert {"ny_orb", "late_mom", "wknd_vwap", "fund_window"} <= tags
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
