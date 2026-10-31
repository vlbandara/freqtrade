"""Round 6: daily point-in-time feature panel for the cross-sectional ML model.

Timing: the row for day D uses data up to the close of the last 1h candle of D, i.e. it is
known at D+1 00:00 UTC. Trades decided on row D are held from D+1 00:00 (edge_lab.run
convention: weights set on the hourly row with open time D 23:00).
"""

from __future__ import annotations

import os

import carry_lab as C
import edge_lab as L
import numpy as np
import pandas as pd
import r6_common as C6


BTC = "BTC_USDT_USDT"
CACHE = L.CACHE


def _daily_last(df):
    return df.resample("D").last()


def build(p: dict, top_n: int = 100, horizon: int = 3):
    cache = CACHE / f"r6_features_top{top_n}_h{horizon}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    close_h = p["close"]
    U_h = L.universe(p, top_n)
    close = _daily_last(close_h)
    high = p["high"].resample("D").max()
    low = p["low"].resample("D").min()
    qv = p["quote_volume"].resample("D").sum(min_count=1)
    fund_paid = p["funding"].resample("D").sum()  # funding rate paid by longs during day D
    fk = _daily_last(p["funding_known"])
    U = _daily_last(U_h.astype(float)).fillna(0) > 0
    spot = _daily_last(C.load_spot(close_h))
    age = _daily_last(C6.listing_age_hours_clean(close_h)) / 24

    r1 = close.pct_change(fill_method=None)
    btc = r1[BTC]
    feats = {}
    for n in (1, 3, 7, 14, 30, 60):
        feats[f"ret_{n}d"] = close.pct_change(n, fill_method=None)
    var_b = btc.rolling(30, min_periods=20).var()
    beta = r1.rolling(30, min_periods=20).cov(btc).div(var_b, axis=0)
    feats["beta_30d"] = beta
    feats["corr_btc_30d"] = r1.rolling(30, min_periods=20).corr(btc)
    resid = r1.sub(beta.mul(btc, axis=0))
    for n in (7, 30):
        feats[f"resid_mom_{n}d"] = resid.rolling(n, min_periods=max(3, n // 2)).sum()
    feats["vol_7d"] = r1.rolling(7, min_periods=5).std()
    feats["vol_30d"] = r1.rolling(30, min_periods=20).std()
    feats["vol_ratio"] = feats["vol_7d"] / feats["vol_30d"]
    feats["downvol_30d"] = r1.where(r1 < 0).rolling(30, min_periods=10).std()
    feats["skew_30d"] = r1.rolling(30, min_periods=20).skew()
    feats["max_30d"] = r1.rolling(30, min_periods=20).max()
    feats["min_30d"] = r1.rolling(30, min_periods=20).min()
    feats["range_7d"] = ((high - low) / close).rolling(7, min_periods=5).mean()
    feats["dist_high_30d"] = close / close.rolling(30, min_periods=20).max() - 1
    feats["dist_low_30d"] = close / close.rolling(30, min_periods=20).min() - 1
    feats["log_qv_30d"] = np.log1p(qv.rolling(30, min_periods=20).mean())
    feats["qv_shock_7_30"] = np.log(qv.rolling(7, min_periods=5).mean() / qv.rolling(30, min_periods=20).mean())
    feats["qv_shock_1_7"] = np.log(qv / qv.rolling(7, min_periods=5).mean())
    feats["amihud_30d"] = (r1.abs() / qv).rolling(30, min_periods=20).mean() * 1e9
    # funding: per-day paid rate (interval-agnostic) and settlement-level known rate
    feats["fund_paid_1d"] = fund_paid
    feats["fund_paid_3d"] = fund_paid.rolling(3, min_periods=2).mean()
    feats["fund_paid_14d"] = fund_paid.rolling(14, min_periods=7).mean()
    feats["fund_chg_3_14"] = feats["fund_paid_3d"] - feats["fund_paid_14d"]
    fs = fund_paid.rolling(60, min_periods=20)
    feats["fund_z_60d"] = (fund_paid - fs.mean()) / fs.std()
    feats["fund_last"] = fk
    basis = close / spot - 1
    feats["basis"] = basis
    feats["basis_3d"] = basis.rolling(3, min_periods=2).mean()
    feats["has_spot"] = spot.notna().astype(float).where(close.notna())
    feats["log_age"] = np.log1p(age)

    # market-state features (same value for every coin on a day; trees use them to condition)
    mkt = {
        "mkt_btc_ret_7d": close[BTC].pct_change(7, fill_method=None),
        "mkt_btc_vol_30d": btc.rolling(30, min_periods=20).std(),
        "mkt_breadth_7d": (feats["ret_7d"].where(U) > 0).sum(1) / U.sum(1),
        "mkt_fund_avg": fund_paid.where(U).mean(1),
        "mkt_dispersion_7d": feats["ret_7d"].where(U).std(1),
    }

    # targets: forward H-day return incl. funding paid by a long, held from D+1 00:00
    fwd = close.shift(-horizon) / close - 1
    fwd_fund = fund_paid[::-1].rolling(horizon, min_periods=1).sum()[::-1].shift(-1)
    tgt = (fwd - fwd_fund).where(U)
    tgt_xs = tgt.sub(tgt.mean(1), axis=0)

    stack = {k: v.where(U).stack(future_stack=True) for k, v in feats.items()}
    X = pd.DataFrame(stack)
    X = X[X["ret_1d"].notna()]
    # cross-sectional ranks (robust to regime-level scale shifts); keep raw market features
    Xr = X.groupby(level=0).rank(pct=True)
    days = Xr.index.get_level_values(0)
    for k, v in mkt.items():
        Xr[k] = v.reindex(days).to_numpy()
    y = tgt_xs.stack(future_stack=True).reindex(Xr.index)
    y_rank = y.groupby(level=0).rank(pct=True) - 0.5
    out = {"X": Xr, "y": y, "y_rank": y_rank, "U": U, "close": close, "horizon": horizon}
    pd.to_pickle(out, cache)
    return out
