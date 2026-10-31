"""Round 3 (modern regime) weight builders. 2020-2021 are never used for design or headline.

A  dilution short   : short perps in their first N days after listing, BTC long hedge
B  BTC vs alts      : long BTC / short EW alt basket while ALT/BTC is below its moving average
C  majors trend     : BTC + ETH long/flat, trend filter, volatility-targeted
All weights at bar t use data up to the close of t and are applied over t+1 (edge_lab.run).
"""

import edge_lab as L
import numpy as np
import pandas as pd


SPLITS_MODERN = {
    "DESIGN": ("2022-01-01", "2023-12-31 23:59"),
    "HOLDOUT": ("2024-01-01", "2030-01-01"),
}
BTC, ETH = "BTC_USDT_USDT", "ETH_USDT_USDT"


def listing_age_hours(close: pd.DataFrame) -> pd.DataFrame:
    """Hours since a perp's first candle; NaN for pairs already trading when the data starts
    (their true listing date is unknown)."""
    age = close.notna().cumsum()
    first = close.apply(lambda s: s.first_valid_index())
    known = first > close.index[0] + pd.Timedelta("7D")
    return age.where(known.to_numpy()[None, :] & close.notna().to_numpy())


def dilution_short(p, n_days=60, skip_days=1, short_gross=0.5, cap=0.1, min_qv_day=2e6):
    close = p["close"]
    age = listing_age_hours(close)
    qv = p["quote_volume"].rolling(24, min_periods=12).sum()
    elig = (age > 24 * skip_days) & (age <= 24 * n_days) & (qv > min_qv_day)
    elig = elig.drop(columns=[c for c in L.EXCLUDE if c in elig.columns], errors="ignore").reindex(
        columns=close.columns, fill_value=False
    )
    vol = close.pct_change(fill_method=None).rolling(72, min_periods=24).std()
    inv = (1 / vol).where(elig)
    w = inv.div(inv.sum(axis=1), axis=0).clip(upper=cap / short_gross).fillna(0)
    w = w.div(np.maximum(w.sum(axis=1), 1.0), axis=0) * short_gross
    w = -w
    w[BTC] = w[BTC] + w.abs().sum(axis=1)  # beta-1 hedge: long BTC for the same notional
    return L.staggered(w, 24, 4)


def btc_vs_alts(p, ma_days=20, gross=1.0, n_alts=30, always=False):
    close = p["close"]
    U = L.universe(p, n_alts + 2)
    U = U.drop(columns=[BTC, ETH]).reindex(columns=close.columns, fill_value=False)
    ret = close.pct_change(fill_method=None)
    alt_ret = ret.where(U).mean(axis=1).fillna(0)
    rel = (1 + alt_ret).cumprod() / (close[BTC] / close[BTC].dropna().iloc[0])
    on = (
        pd.Series(1.0, index=close.index)
        if always
        else (rel < rel.rolling(24 * ma_days).mean()).astype(float)
    )
    alt_w = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = -alt_w.mul(on * gross / 2, axis=0)
    w[BTC] = w[BTC] + on * gross / 2
    return L.staggered(w, 24, 4)


def majors_trend(p, rule="ema", fast_days=10, slow_days=50, target_vol=0.15, assets=(BTC, ETH)):
    close = p["close"]
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in assets:
        px = close[a]
        if rule == "ema":
            sig = (px.ewm(span=24 * fast_days).mean() > px.ewm(span=24 * slow_days).mean()).astype(
                float
            )
        else:  # donchian: enter above the slow-day high, exit below the fast-day low
            hi = px.rolling(24 * slow_days).max().shift(1)
            lo = px.rolling(24 * fast_days).min().shift(1)
            st = pd.Series(np.nan, index=px.index)
            st[px > hi] = 1.0
            st[px < lo] = 0.0
            sig = st.ffill().fillna(0)
        rv = px.pct_change(fill_method=None).rolling(24 * 30, min_periods=24 * 10).std() * np.sqrt(
            24 * 365
        )
        size = (target_vol / rv).clip(upper=1.0) / len(assets)
        w[a] = (sig * size).fillna(0)
    return L.staggered(w, 24, 4)


def yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def dilution_short_confirmed(p, n_days=60, ema_hours=72, short_gross=0.5, cap=0.1, min_qv_day=2e6):
    """A2: like dilution_short, but only short a new listing while it trades below its own
    `ema_hours` EMA (the decay has started), which avoids shorting into early squeezes."""
    close = p["close"]
    age = listing_age_hours(close)
    qv = p["quote_volume"].rolling(24, min_periods=12).sum()
    below = close < close.ewm(span=ema_hours, min_periods=24).mean()
    elig = (age > 24) & (age <= 24 * n_days) & (qv > min_qv_day) & below
    elig = elig.drop(columns=[c for c in L.EXCLUDE if c in elig.columns], errors="ignore").reindex(
        columns=close.columns, fill_value=False
    )
    vol = close.pct_change(fill_method=None).rolling(72, min_periods=24).std()
    inv = (1 / vol).where(elig)
    w = inv.div(inv.sum(axis=1), axis=0).clip(upper=cap / short_gross).fillna(0)
    w = w.div(np.maximum(w.sum(axis=1), 1.0), axis=0) * short_gross
    w = -w
    w[BTC] = w[BTC] + w.abs().sum(axis=1)
    return L.staggered(w, 24, 4)


def inv_vol_mix(streams: dict, window_days=90, gross_scale=1.0):
    """Daily inverse-vol mix of daily return streams; weights use data up to the prior day."""
    D = pd.DataFrame(streams).fillna(0)
    iv = 1 / D.rolling(window_days, min_periods=30).std().shift(1)
    wts = iv.div(iv.sum(axis=1), axis=0).fillna(1 / D.shape[1])
    return (wts * D).sum(axis=1) * gross_scale, wts
