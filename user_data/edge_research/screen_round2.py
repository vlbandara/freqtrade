"""Round 2: dissect the funding-contrarian family + weird combos (IS + VAL only)."""

import edge_lab as L
import numpy as np
import pandas as pd


p = L.load_panel()
close = p["close"]
fk = p["funding_known"]
ret1h = close.pct_change(fill_method=None)
rows = []


def rec(name, w, every=8, top_n=50, **kw):
    w = L.staggered(w, every, n=min(every, 4)) if every > 1 else w
    r = L.run(name, w, p, **kw)
    df = L.report(r, splits=("IS", "VAL"), show=False)
    g = L.run(name, w, p, fee=0.0, funding=False)
    gdf = L.report(g, splits=("IS", "VAL"), show=False)
    row = {"name": name}
    for s in ("IS", "VAL"):
        row[f"{s}_sh"] = df.loc[s, "sharpe"]
        row[f"{s}_cagr"] = df.loc[s, "cagr"]
        row[f"{s}_dd"] = df.loc[s, "maxdd"]
        row[f"{s}_priceonly_cagr"] = gdf.loc[s, "cagr"]
    row["turn/d"] = df["turn/d"].mean()
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return r


def fund_avg(k):
    return fk.rolling(8 * k, min_periods=8).mean()


# --- plateau of the base funding-contrarian signal
for top_n in (30, 50, 100):
    U = L.universe(p, top_n)
    for k in (6, 9, 14):
        for q in (0.1, 0.2, 0.33):
            rec(f"A_fc_top{top_n}_k{k}_q{q}", L.long_short_from_rank(L.xs_rank(-fund_avg(k), U), q))

U = L.universe(p, 50)
f9 = fund_avg(9)
# --- legs: is it the short leg (crowded longs) or the long leg?
rec("A_fc_longleg_only", L.long_short_from_rank(L.xs_rank(-f9, U), 0.2, short=False))
rec("A_fc_shortleg_only", L.long_short_from_rank(L.xs_rank(-f9, U), 0.2, long=False))
# --- rebalance cadence
for every in (24, 72):
    rec(f"A_fc_every{every}", L.long_short_from_rank(L.xs_rank(-f9, U), 0.2), every)

# --- funding *change*: rising funding = leverage piling in
for k in (3, 9):
    dfund = fund_avg(k) - fund_avg(k).shift(8 * k)
    rec(f"J_dfund_contra_k{k}", L.long_short_from_rank(L.xs_rank(-dfund, U), 0.2))

# --- funding relative to the coin's own history (z-score), not cross-sectional level
for k in (9,):
    z = (fund_avg(k) - fk.rolling(24 * 60, min_periods=24 * 20).mean()) / fk.rolling(
        24 * 60, min_periods=24 * 20
    ).std()
    rec("J_fundz_contra", L.long_short_from_rank(L.xs_rank(-z, U), 0.2))

# --- composite: momentum (14d) + funding contrarian
mom14 = close.pct_change(336, fill_method=None)
mom7 = close.pct_change(168, fill_method=None)
for wf in (0.5, 0.67):
    sc = wf * L.xs_rank(-f9, U) + (1 - wf) * L.xs_rank(mom14, U)
    rec(f"K_fc{wf}_mom14", L.long_short_from_rank(L.xs_rank(sc, U), 0.2))
# --- "unlevered rally": price momentum residualised on funding (mom per unit funding)
sc = L.xs_rank(mom7, U) * (1 - L.xs_rank(f9, U)) - L.xs_rank(-mom7, U) * L.xs_rank(f9, U)
rec("K_unlevered_rally", L.long_short_from_rank(L.xs_rank(sc, U), 0.2))

# --- New-listing decay: short coins in their first N days, hedged with BTC long
age = close.notna().cumsum()
first_date = close.apply(lambda s: s.first_valid_index())
new_since_2020 = first_date > pd.Timestamp("2020-01-15", tz="UTC")
for n_days in (14, 30, 60):
    young = (age > 24) & (age <= n_days * 24) & new_since_2020.values[None, :]
    young = young & close.notna()
    w = -young.astype(float).div(young.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
    w["BTC_USDT_USDT"] = w["BTC_USDT_USDT"] + 0.5 * (young.sum(axis=1) > 0)
    rec(f"C_newlisting_short_{n_days}d", w, 24)

# --- Funding-settlement clock: long-short (low vs high funding) only in the hours around
# settlement. If crowded longs de-risk *into* settlement, spread should be concentrated there.
hours = close.index.hour
base = L.long_short_from_rank(L.xs_rank(-f9, U), 0.2)
for name, hrs in {
    "pre_settle": [7, 15, 23],  # candle 07:00-08:00 etc ends at settlement
    "post_settle": [0, 8, 16],
    "far": [3, 4, 11, 12, 19, 20],
}.items():
    mask = pd.Series(np.isin(hours, hrs), index=close.index)
    # weights decided at t apply to candle t+1 -> select on the *next* candle's hour
    sel = mask.shift(-1, fill_value=False)
    rec(f"L_clock_{name}", base.mul(sel, axis=0), 1, funding=True)

# --- Weekend effect on the XS funding spread
dow = pd.Series(close.index.dayofweek, index=close.index).shift(-1).fillna(0)
rec("L_weekend_only", base.mul(dow >= 5, axis=0), 1)
rec("L_weekday_only", base.mul(dow < 5, axis=0), 1)

res = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(res.round(3))
res.to_csv("round2.csv")
