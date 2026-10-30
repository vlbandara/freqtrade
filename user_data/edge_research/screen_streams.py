"""Round 2 stream screen (IS + VAL only). Kill rule: IS & VAL Sharpe >= 1.5 and corr to carry < 0.3.

S1 basis-spike carry   : perp premium over spot spikes -> long spot / short perp until it converges
S2 cascade reversion   : after market-wide 1h flush, buy the most oversold alts, short BTC (beta)
S3 large-cap pairs     : log-ratio z-score mean reversion between top-15 perps, dollar-neutral
"""

import itertools

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
p = L.load_panel()
close = p["close"]
spot = C.load_spot(close)
U50 = L.universe(p, 50)
U15 = L.universe(p, 15)
carry = pd.read_parquet("carry_pnl.parquet")["carry"]
cd = (1 + carry).groupby(carry.index.floor("D")).prod() - 1
rows = []


def rec(name, ret):
    d = (1 + ret).groupby(ret.index.floor("D")).prod() - 1
    row = {"name": name}
    for s in ("IS", "VAL"):
        m = L.metrics(ret, *L.SPLITS[s])
        row.update({f"{s}_sh": m["sharpe"], f"{s}_cagr": m["cagr"], f"{s}_dd": m["maxdd"]})
    row["corr_carry"] = d.loc[:"2023"].corr(cd.loc[:"2023"])
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


# ---------------------------------------------------------------- S1 basis spike
prem = close / spot - 1
for x in (0.003, 0.006, 0.01):
    for hold in (24, 72):
        ent = (prem > x) & U50
        state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        state[ent] = 1.0
        state[prem < x / 4] = 0.0
        pos = state.ffill(limit=hold).fillna(0)  # time stop: entry signal must refresh
        w = pos.div(np.maximum(pos.sum(axis=1), 10), axis=0)
        r, _, _ = C2.simulate(p, spot, w)
        rec(f"S1_basis_spike_x{x}_h{hold}", r)

# ---------------------------------------------------------------- S2 cascade reversion
ret = close.pct_change(fill_method=None)
ew = ret.where(U50).mean(axis=1)
btc = ret["BTC_USDT_USDT"]
beta = (
    ret.rolling(24 * 30, min_periods=24 * 10)
    .cov(btc)
    .div(btc.rolling(24 * 30, min_periods=240).var(), axis=0)
)
for flush in (-0.03, -0.05):
    for hold in (6, 24):
        for nworst in (5, 10):
            trig = ew < flush
            r1 = ret.where(U50)
            worst = r1.rank(axis=1, ascending=True) <= nworst
            ent = worst & trig.values[:, None]
            pos = pd.DataFrame(np.where(ent, 1.0, np.nan), index=close.index, columns=close.columns)
            pos = pos.ffill(limit=hold - 1).fillna(0)
            # restart clock: positions are only kept `hold` bars after the last trigger
            wl = pos / nworst * 0.5
            hedge = -(wl * beta.fillna(1)).sum(axis=1)
            w = wl.copy()
            w["BTC_USDT_USDT"] = w["BTC_USDT_USDT"] + hedge
            rec(f"S2_cascade_f{flush}_h{hold}_n{nworst}", L.run("", w, p).pnl)

# ---------------------------------------------------------------- S3 pairs stat-arb (daily)
cl = close.resample("1D").last()
u15 = U15.resample("1D").last().fillna(False)
lc = np.log(cl)
for lb in (10, 30):
    for zin in (1.5, 2.0):
        wd = pd.DataFrame(0.0, index=cl.index, columns=cl.columns)
        cands = [c for c in cl.columns if u15[c].any()]
        for a, b in itertools.combinations(cands, 2):
            both = u15[a] & u15[b]
            if both.sum() < 120:
                continue
            spr = lc[a] - lc[b]
            z = (spr - spr.rolling(lb).mean()) / spr.rolling(lb).std()
            st = pd.Series(np.nan, index=cl.index)
            st[z > zin] = -1.0
            st[z < -zin] = 1.0
            st[z.abs() < 0.5] = 0.0
            st[~both] = 0.0
            st = st.ffill().fillna(0)
            wd[a] += st
            wd[b] -= st
        gross = wd.abs().sum(axis=1)
        wd = wd.div(np.maximum(gross, 1), axis=0)  # gross <= 1
        # decision uses day d's close -> applies from day d+1 00:00 onward
        w = wd.shift(1).reindex(close.index, method="ffill").fillna(0)
        rec(f"S3_pairs_lb{lb}_z{zin}", L.run("", w, p).pnl)

res = pd.DataFrame(rows).set_index("name")
print(res.round(3).to_string())
res.round(4).to_csv("streams_screen.csv")
