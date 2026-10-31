"""S2: liquidation-cascade reversion (perp-only, beta-hedged with BTC).

When the equal-weight top-50 basket drops more than `flush` in one hour (forced selling),
buy the `nworst` most oversold coins of that hour (0.5 / nworst each), short BTC for the same
beta, hold `hold` hours from the last trigger. Signals use only the closed candle.
"""

import edge_lab as L
import numpy as np
import pandas as pd


def cascade_weights(p, U, flush=-0.03, hold=6, nworst=5, gross_long=0.5):
    close = p["close"]
    ret = close.pct_change(fill_method=None)
    ew = ret.where(U).mean(axis=1)
    btc = ret["BTC_USDT_USDT"]
    beta = (
        ret.rolling(24 * 30, min_periods=24 * 10)
        .cov(btc)
        .div(btc.rolling(24 * 30, min_periods=240).var(), axis=0)
    )
    trig = ew < flush
    worst = ret.where(U).rank(axis=1, ascending=True) <= nworst
    ent = worst & trig.to_numpy()[:, None]
    pos = pd.DataFrame(np.where(ent, 1.0, np.nan), index=close.index, columns=close.columns)
    pos = pos.ffill(limit=hold - 1).fillna(0)
    wl = pos / nworst * gross_long
    hedge = -(wl * beta.fillna(1).clip(0, 3)).sum(axis=1)
    w = wl.copy()
    w["BTC_USDT_USDT"] = w["BTC_USDT_USDT"] + hedge
    return w, trig


def cascade(p, U, fee=L.FEE, **kw):
    w, trig = cascade_weights(p, U, **kw)
    return L.run("cascade", w, p, fee=fee).pnl, w, trig
