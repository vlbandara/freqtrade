"""Round 4 (modern regime) weight builders. Hourly weights, decided on closed candles only.

E  pullback in uptrend  : long majors on deep dips below the 20d EMA while above the 100d EMA
F  majors vs BTC trend  : Donchian on each coin/BTC ratio, long/short the ratio (market-neutral)
G  daily shock reversal : fade a >k-sigma daily move that goes against the 50d trend
H  broad majors trend   : top-10 by volume, Donchian 20/55 long/flat, vol-targeted (C extended)
I  young-coin short     : short coins aged 30..N days, BTC hedge (A extended)
"""

import edge_lab as L
import numba
import numpy as np
import pandas as pd
from modern_lab import BTC, listing_age_hours


MAJORS = ("BTC", "ETH", "SOL", "BNB", "XRP")
RATIO_COINS = ("ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK")


def _sym(s):
    return f"{s}_USDT_USDT"


def _rv(px, days=30):
    return px.pct_change(fill_method=None).rolling(24 * days, min_periods=24 * 10).std() * np.sqrt(
        24 * 365
    )


@numba.njit(cache=True)
def _pullback(c, ema20, trend, atr, k, max_h, stop_k):
    n = len(c)
    st = np.zeros(n)
    pos, entry, held = 0, 0.0, 0
    for t in range(n):
        if pos == 1:
            held += 1
            if c[t] >= ema20[t] or held >= max_h or c[t] < entry - stop_k * atr[t]:
                pos = 0
        elif trend[t] and atr[t] > 0 and c[t] < ema20[t] - k * atr[t]:
            pos, entry, held = 1, c[t], 0
        st[t] = pos
    return st


def pullback(p, k=1.5, max_days=10, stop_k=3.0, target_vol=0.15, assets=MAJORS):
    close = p["close"]
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in map(_sym, assets):
        px = close[a].ffill()
        ema20 = px.ewm(span=24 * 20, min_periods=24 * 20).mean()
        trend = (px > px.ewm(span=24 * 100, min_periods=24 * 100).mean()).to_numpy()
        rng = px.rolling(24).max() - px.rolling(24).min()
        atr = rng.rolling(24 * 14, min_periods=24 * 7).mean().fillna(0)
        st = _pullback(
            px.to_numpy(),
            ema20.fillna(np.inf).to_numpy(),
            trend,
            atr.to_numpy(),
            k,
            24 * max_days,
            stop_k,
        )
        size = (target_vol / _rv(px)).clip(upper=1.0) / len(assets)
        w[a] = (pd.Series(st, index=px.index) * size).fillna(0).where(close[a].notna(), 0)
    return w


@numba.njit(cache=True)
def _donchian_ls(x, hi_s, lo_s, hi_f, lo_f):
    n = len(x)
    st = np.zeros(n)
    pos = 0
    for t in range(n):
        if pos == 0:
            if x[t] > hi_s[t]:
                pos = 1
            elif x[t] < lo_s[t]:
                pos = -1
        elif pos == 1 and x[t] < lo_f[t]:
            pos = -1 if x[t] < lo_s[t] else 0
        elif pos == -1 and x[t] > hi_f[t]:
            pos = 1 if x[t] > hi_s[t] else 0
        st[t] = pos
    return st


def ratio_trend(p, fast=20, slow=55, target_vol=0.15, coins=RATIO_COINS):
    close = p["close"].ffill()
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    live = [c for c in map(_sym, coins) if c in close]
    for c in live:
        r = np.log(close[c] / close[BTC])
        hs, ls_ = r.rolling(24 * slow).max().shift(1), r.rolling(24 * slow).min().shift(1)
        hf, lf = r.rolling(24 * fast).max().shift(1), r.rolling(24 * fast).min().shift(1)
        st = _donchian_ls(
            r.fillna(0).to_numpy(),
            hs.fillna(np.inf).to_numpy(),
            ls_.fillna(-np.inf).to_numpy(),
            hf.fillna(np.inf).to_numpy(),
            lf.fillna(-np.inf).to_numpy(),
        )
        vol = r.diff().rolling(24 * 30, min_periods=24 * 10).std() * np.sqrt(24 * 365)
        size = (target_vol / vol).clip(upper=0.5) / len(live)
        leg = (pd.Series(st, index=r.index) * size).fillna(0).where(p["close"][c].notna(), 0)
        w[c] += leg
        w[BTC] -= leg
    return L.staggered(w, 24, 4)


def shock_reversal(p, k=2.0, hold_days=3, size=0.1, assets=MAJORS):
    """Daily decision at 00:00 UTC from the completed previous day."""
    close = p["close"]
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in map(_sym, assets):
        d = close[a].resample("1D").last()
        r = d.pct_change(fill_method=None)
        sig = r.rolling(30, min_periods=20).std()
        up = d > d.rolling(50, min_periods=50).mean()
        ev = pd.Series(np.nan, index=d.index)
        ev[(r < -k * sig) & up] = 1.0  # sharp dip inside an uptrend -> buy
        ev[(r > k * sig) & ~up] = -1.0  # sharp pop inside a downtrend -> sell
        held = ev.ffill(limit=hold_days - 1) if hold_days > 1 else ev
        pos = held.fillna(0) * size
        # day d's close is known at 00:00 of d+1 -> trade from d+1
        hourly = pos.shift(1).reindex(close.index, method="ffill").fillna(0)
        w[a] = hourly.where(close[a].notna(), 0)
    return w


def broad_trend(p, n=10, fast=20, slow=55, target_vol=0.15):
    close = p["close"]
    U = L.universe(p, n)
    hi = close.rolling(24 * slow).max().shift(1)
    lo = close.rolling(24 * fast).min().shift(1)
    st = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    st[close > hi] = 1.0
    st[close < lo] = 0.0
    sig = st.ffill().fillna(0).where(U, 0)
    rv = close.pct_change(fill_method=None).rolling(24 * 30, min_periods=24 * 10).std() * np.sqrt(
        24 * 365
    )
    w = (sig * (target_vol / rv) / n).clip(upper=0.2).fillna(0)
    return L.staggered(w, 24, 4)


def young_short(p, min_days=30, max_days=180, short_gross=0.25, cap=0.05, min_qv_day=2e6):
    close = p["close"]
    age = listing_age_hours(close)
    qv = p["quote_volume"].rolling(24, min_periods=12).sum()
    elig = (age > 24 * min_days) & (age <= 24 * max_days) & (qv > min_qv_day)
    elig = elig.drop(columns=[c for c in L.EXCLUDE if c in elig.columns], errors="ignore").reindex(
        columns=close.columns, fill_value=False
    )
    vol = close.pct_change(fill_method=None).rolling(72, min_periods=24).std()
    inv = (1 / vol).where(elig)
    w = inv.div(inv.sum(axis=1), axis=0).clip(upper=cap / short_gross).fillna(0)
    w = -(w.div(np.maximum(w.sum(axis=1), 1.0), axis=0) * short_gross)
    w[BTC] = w[BTC] + w.abs().sum(axis=1)
    return L.staggered(w, 24, 4)


# ---------------------------------------------------------------- batch 2 (never run on holdout)
def weekend(p, mode="btc_long", size=0.5):
    """J: hold only Saturday+Sunday (entered at Fri 24:00 UTC, exited Mon 00:00).
    btc_long: long BTC; alt_short: short EW top-30 alts vs long BTC."""
    close = p["close"]
    nxt_dow = pd.Series(close.index.dayofweek, index=close.index).shift(-1).fillna(0)
    on = (nxt_dow >= 5).astype(float)  # weight at t applies to candle t+1
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if mode == "btc_long":
        w[BTC] = on * size
    else:
        U = (
            L.universe(p, 32)
            .drop(columns=[BTC, "ETH_USDT_USDT"])
            .reindex(columns=close.columns, fill_value=False)
        )
        alt = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        w = -alt.mul(on * size, axis=0)
        w[BTC] += on * size
    return w


def turn_of_month(p, days_before=1, days_after=3, size=0.5, asset=BTC):
    """K: long from `days_before` days before month end through day `days_after` of the month."""
    close = p["close"]
    t = close.index + pd.Timedelta("1h")  # weight at t applies to candle t+1
    dim = t.days_in_month
    on = ((t.day > dim - days_before) | (t.day <= days_after)).astype(float)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[asset] = np.asarray(on) * size
    return w


def funding_capitulation(p, pct=0.2, min_hold_days=3, size=0.5, assets=(BTC, "ETH_USDT_USDT")):
    """L: market-wide median 3d funding (top-50) in its bottom `pct` of the trailing year ->
    long BTC/ETH; hold at least `min_hold_days` after the last signal."""
    close = p["close"]
    U = L.universe(p, 50)
    agg = p["funding_known"].rolling(72, min_periods=8).mean().where(U).median(axis=1)
    rank = agg.rolling(24 * 365, min_periods=24 * 90).rank(pct=True)
    sig = pd.Series(np.where(rank < pct, 1.0, np.nan), index=close.index)
    on = sig.ffill(limit=24 * min_hold_days).fillna(0)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in assets:
        w[a] = on * size / len(assets)
    return w


def pump_short(p, jump=0.3, hold_days=3, size=0.05, n_univ=100, max_gross=0.5):
    """M: short coins (top-100 by volume, ex BTC/ETH) after a >`jump` 24h rise; BTC hedge."""
    close = p["close"]
    U = (
        L.universe(p, n_univ)
        .drop(columns=[BTC, "ETH_USDT_USDT"])
        .reindex(columns=close.columns, fill_value=False)
    )
    r24 = close.pct_change(24, fill_method=None)
    ev = pd.DataFrame(
        np.where((r24 > jump) & U, 1.0, np.nan), index=close.index, columns=close.columns
    )
    pos = ev.ffill(limit=24 * hold_days).fillna(0) * size
    pos = pos.div(np.maximum(pos.sum(axis=1) / max_gross, 1.0), axis=0)
    w = -pos
    w[BTC] = w[BTC] + pos.sum(axis=1)
    return w
