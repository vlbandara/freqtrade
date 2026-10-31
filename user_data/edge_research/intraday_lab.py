"""D: intraday mean reversion on major perps (5m candles) with maker (post-only) execution.

Signal at the close of bar t: z = (close - EMA_n) / rolling std_n.  When |z| > k and the market
is calm (24h realised vol below its 30-day median), post a limit order at close[t] against the
move.  It fills on bar t+1 only if price trades *through* it by `touch` (queue-position haircut).
Exit: limit at the previous bar's EMA (maker) when touched through, else a time stop after
`max_bars` or a hard stop at |z| > stop_k, both at close as taker.  Funding is ignored (holds are
minutes to hours).  Equity is marked to market every bar.
"""

from pathlib import Path

import numba
import numpy as np
import pandas as pd


FUT = Path(__file__).resolve().parents[1] / "data" / "binance" / "futures"
MAKER, TAKER = 0.0002, 0.0005 + 0.0002


def load5m(sym: str) -> pd.DataFrame:
    d = pd.read_feather(FUT / f"{sym}_USDT_USDT-5m-futures.feather")
    return d.drop_duplicates("date").set_index("date")


@numba.njit(cache=True)
def _sim(o, h, l, c, ema, z, calm, k, stop_k, max_bars, touch, maker, taker):
    n = len(c)
    ret = np.zeros(n)
    pos = 0  # +1 long, -1 short
    entry = 0.0
    ref = 0.0
    held = 0
    pend = 0
    plim = 0.0
    trades = 0
    for t in range(1, n):
        r = 0.0
        # pending entry from bar t-1
        if pend != 0 and pos == 0:
            if (pend == 1 and l[t] < plim * (1 - touch)) or (
                pend == -1 and h[t] > plim * (1 + touch)
            ):
                pos = pend
                entry = plim
                ref = plim
                held = 0
                r -= maker
                trades += 1
                # fill bar: unknown intrabar order, so no exit on the same bar; mark to close
                r += pos * (c[t] - ref) / entry
                ref = c[t]
                pend = 0
                ret[t] = r
                continue
            pend = 0
        if pos != 0:
            held += 1
            tgt = ema[t - 1]
            hit = (pos == 1 and h[t] > tgt * (1 + touch) and tgt > entry) or (
                pos == -1 and l[t] < tgt * (1 - touch) and tgt < entry
            )
            if hit:
                r += pos * (tgt - ref) / entry - maker
                pos = 0
            elif held >= max_bars or abs(z[t]) > stop_k:
                r += pos * (c[t] - ref) / entry - taker
                pos = 0
            else:
                r += pos * (c[t] - ref) / entry
                ref = c[t]
        if pos == 0 and calm[t]:
            if z[t] < -k:
                pend = 1
                plim = c[t]
            elif z[t] > k:
                pend = -1
                plim = c[t]
        ret[t] = r
    return ret, trades


def reversion(sym, n=48, k=2.5, stop_k=5.0, max_bars=36, touch=0.0002, maker=MAKER, taker=TAKER):
    d = load5m(sym)
    c = d["close"]
    ema = c.ewm(span=n, adjust=False).mean()
    sd = c.rolling(n).std()
    z = ((c - ema) / sd).fillna(0)
    rv = np.log(c).diff().rolling(288).std()
    calm = (rv < rv.rolling(288 * 30, min_periods=288 * 10).median()).fillna(False)
    r, trades = _sim(
        d["open"].to_numpy(),
        d["high"].to_numpy(),
        d["low"].to_numpy(),
        c.to_numpy(),
        ema.to_numpy(),
        z.to_numpy(),
        calm.to_numpy(),
        k,
        stop_k,
        max_bars,
        touch,
        maker,
        taker,
    )
    return pd.Series(r, index=d.index), trades


def basket(syms=("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"), lev=1.0, **kw):
    """Equal capital per coin; returns hourly portfolio returns and total trades."""
    rs, tot = [], 0
    for s in syms:
        r, t = reversion(s, **kw)
        rs.append(r.rename(s))
        tot += t
    R = pd.concat(rs, axis=1).fillna(0)
    port = R.mean(axis=1) * lev
    hourly = (1 + port).groupby(port.index.floor("h")).prod() - 1
    return hourly, tot, R
