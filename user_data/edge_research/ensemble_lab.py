"""Trend sleeve + carry sleeve ensemble (shared implementation)."""

import carry_lab as C
import edge_lab as L
import numpy as np


def trend_weights(p, majors=30, lb_days=20, mode="ls", fund_filter=0.0003, book_vol=0.4):
    """Cross-sectional universe, per-coin EMA trend, inverse-vol sized, funding-filtered."""
    close = p["close"]
    fk = p["funding_known"]
    Um = L.universe(p, majors)
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(24 * 20, min_periods=24 * 10).std() * np.sqrt(24 * 365)
    lb = 24 * lb_days
    ema_f = close.ewm(span=lb // 4, min_periods=lb // 4).mean()
    ema_s = close.ewm(span=lb, min_periods=lb).mean()
    sig = np.sign(ema_f - ema_s)
    if mode == "long":
        sig = sig.clip(lower=0)
    w = (sig.where(Um) * (book_vol / vol) / majors).clip(-0.25, 0.25).fillna(0)
    gross = w.abs().sum(axis=1)
    w = w.div(np.maximum(gross, 2.0) / 2.0, axis=0)
    if fund_filter is not None:
        f3 = fk.rolling(72, min_periods=8).mean()
        w = w.where(~((np.sign(w) * f3) > fund_filter), 0.0)
    return L.staggered(w, 24, 4)


def ensemble(p, spot, w_trend=0.5, carry_kwargs=None, trend_kwargs=None, fee_mult=1.0):
    carry_kwargs = carry_kwargs or {}
    trend_kwargs = trend_kwargs or {}
    U50 = L.universe(p, 50)
    cp, cw, comp = C.carry(
        p, spot, U50, perp_fee=L.FEE * fee_mult, spot_fee=C.SPOT_FEE * fee_mult, **carry_kwargs
    )
    tw = trend_weights(p, **trend_kwargs)
    tr = L.run("trend", tw, p, fee=L.FEE * fee_mult)
    pnl = w_trend * tr.pnl + (1 - w_trend) * cp
    return pnl, tr.pnl, cp
