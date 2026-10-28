"""Round 5: market timing with the aggregate-leverage gauge x trend (IS + VAL only).

Aggregate funding = median 3d funding across the top-50 universe ("froth gauge").
Instruments: BTC, ETH, and an equal-weight top-10 basket.
"""

import edge_lab as L
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
p = L.load_panel()
close = p["close"]
fk = p["funding_known"]
U50 = L.universe(p, 50)
U10 = L.universe(p, 10)
rows = []

agg = fk.rolling(72, min_periods=8).mean().where(U50).median(axis=1)
# percentile of the gauge vs its own trailing year (no lookahead)
agg_pct = agg.rolling(24 * 365, min_periods=24 * 90).rank(pct=True)


def rec(name, w, every=24):
    w = L.staggered(w, every, 4) if every > 1 else w
    r = L.run(name, w, p)
    df = L.report(r, splits=("IS", "VAL"), show=False)
    row = {"name": name}
    for s in ("IS", "VAL"):
        for m in ("sharpe", "cagr", "maxdd"):
            row[f"{s}_{m}"] = df.loc[s, m]
    row["expo"] = df["expo"].mean()
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return r


def instrument(name):
    if name == "EW10":
        return U10.astype(float).div(U10.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[f"{name}_USDT_USDT"] = 1.0
    return w


for inst in ("BTC", "ETH", "EW10"):
    base = instrument(inst)
    px = close[f"{inst}_USDT_USDT"] if inst != "EW10" else None
    if inst == "EW10":
        r = close.pct_change(fill_method=None).where(U10).mean(axis=1).fillna(0)
        px = (1 + r).cumprod()
    rec(f"{inst}_buyhold", base, 1)
    for lb_days in (20, 50):
        trend = (px.ewm(span=24 * lb_days // 4).mean() > px.ewm(span=24 * lb_days).mean()).astype(
            float
        )
        rec(f"{inst}_trend{lb_days}_long", base.mul(trend, axis=0))
        rec(f"{inst}_trend{lb_days}_ls", base.mul(2 * trend - 1, axis=0))
        for cap in (0.8, 0.9):
            cool = (agg_pct < cap).astype(float)
            rec(f"{inst}_trend{lb_days}_long_notfroth{cap}", base.mul(trend * cool, axis=0))
    # contrarian capitulation: long when the market-wide funding is in its bottom decile
    for lo in (0.1, 0.2):
        cap_sig = (agg_pct < lo).astype(float)
        rec(f"{inst}_capitulation_long_p{lo}", base.mul(cap_sig, axis=0))
    # froth short: short when funding is in its top decile and trend has rolled over
    trend20 = (px.ewm(span=120).mean() > px.ewm(span=480).mean()).astype(float)
    froth_short = ((agg_pct > 0.9) & (trend20 == 0)).astype(float)
    rec(f"{inst}_froth_rollover_short", base.mul(-froth_short, axis=0))

res = pd.DataFrame(rows).set_index("name")
print(res.round(3).to_string())
res.to_csv("round5.csv")
