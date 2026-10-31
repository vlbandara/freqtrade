"""Final, one-shot evaluation of the locked candidate incl. the blind OOS period.

Locked before this run (chosen on IS+VAL only):
  universe top 50 by 30d median quote volume, signal = 3-day mean of settled funding,
  long lowest 20% / short highest 20%, rebalanced on an 8h schedule (4 staggered sleeves).
  freqtrade replica: enter at 20%, exit at 30% (hysteresis), 5% of equity per position.
"""

import edge_lab as L
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
rng = np.random.default_rng(7)

p = L.load_panel()
close = p["close"]
fk = p["funding_known"]
U50 = L.universe(p, 50)


def signal(k_days=3):
    return fk.rolling(int(round(24 * k_days)), min_periods=8).mean()


def base_weights(U=U50, k_days=3, q=0.2):
    return L.staggered(L.long_short_from_rank(L.xs_rank(-signal(k_days), U), q), 8, 4)


print("\n#### 1. Locked candidate, all splits")
w = base_weights()
res = L.run("FundingContrarian_top50_3d_q0.2", w, p)
tab = L.report(res)
L.report(L.run("  ...same at 2x costs", w, p, fee=2 * L.FEE))
L.report(L.run("  ...price only (no fees, no funding)", w, p, fee=0, funding=False))

print("\n#### 2. freqtrade replica (hysteresis 0.20/0.30, 5%/position)")
wh = L.hysteresis_weights(L.xs_rank(-signal(3), U50), 0.2, 0.3, 0.05)
res_h = L.run("Hysteresis replica", wh, p)
L.report(res_h)

print("\n#### 3. Parameter plateau on OOS (all 27 neighbours)")
rows = []
for top_n in (30, 50, 100):
    U = U50 if top_n == 50 else L.universe(p, top_n)
    for k in (2, 3, 14 / 3):
        for q in (0.1, 0.2, 0.33):
            r = L.run("", base_weights(U, k, q), p)
            m = L.metrics(r.pnl, *L.SPLITS["OOS"])
            rows.append({"top_n": top_n, "fund_days": round(k, 2), "q": q, **m})
plateau = pd.DataFrame(rows)
print(plateau.round(3).to_string())
print(
    f"OOS: {(plateau.ret > 0).mean():.0%} of neighbours profitable, "
    f"median Sharpe {plateau.sharpe.median():.2f}"
)

print("\n#### 4. Quarterly returns (locked candidate)")
qr = L.quarterly(res)
print(qr.loc["2021Q1":].round(3).to_string())
print(f"positive quarters: {(qr.loc['2021Q1':] > 0).mean():.0%}")

print("\n#### 5. Market exposure")
daily = (1 + res.pnl).groupby(res.pnl.index.floor("D")).prod() - 1
btc = close["BTC_USDT_USDT"].resample("1D").last().pct_change()
ew = close.pct_change(fill_method=None).where(U50).mean(axis=1)
ewd = (1 + ew).groupby(ew.index.floor("D")).prod() - 1
for nm, bench in (("BTC", btc), ("EW top50 alts", ewd)):
    j = pd.concat([daily, bench], axis=1).dropna().loc["2021":]
    beta = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()
    print(f"corr to {nm}: {j.corr().iloc[0, 1]:.2f}  beta: {beta:.2f}")

print("\n#### 6. Permutation test (OOS): shuffle funding across coins each day")
oos = slice(*L.SPLITS["OOS"])
f = signal(3)
days = f.index.floor("D")
obs = L.metrics(res.pnl, *L.SPLITS["OOS"])["sharpe"]
null = []
fv = f.loc[oos]
uv = U50.loc[oos]
for i in range(40):
    # same permutation of columns for all hours of a day -> keeps turnover realistic
    out = fv.copy()
    for d, idx in fv.groupby(days[(f.index >= oos.start) & (f.index < oos.stop)]).groups.items():
        perm = rng.permutation(fv.shape[1])
        out.loc[idx] = fv.loc[idx].values[:, perm]
    wr = L.staggered(L.long_short_from_rank(L.xs_rank(-out, uv), 0.2), 8, 4)
    pr = {k: v.loc[oos] for k, v in p.items()}
    null.append(L.metrics(L.run("", wr, pr).pnl)["sharpe"])
null = np.array(null)
print(
    f"observed OOS Sharpe {obs:.2f}; null mean {null.mean():.2f} sd {null.std():.2f}; "
    f"p = {(np.sum(null >= obs) + 1) / (len(null) + 1):.3f}"
)

print("\n#### 7. Bootstrap 95% CI of OOS Sharpe (stationary block bootstrap, 10d blocks)")
d_oos = daily.loc[oos]
sh = []
n = len(d_oos)
for _ in range(2000):
    idx = []
    while len(idx) < n:
        s = rng.integers(0, n)
        idx.extend(range(s, min(s + 10, n)))
    x = d_oos.values[idx[:n]]
    sh.append(x.mean() / x.std() * np.sqrt(365))
print(f"OOS Sharpe 95% CI: [{np.percentile(sh, 2.5):.2f}, {np.percentile(sh, 97.5):.2f}]")

print("\n#### 8. Survivorship stress: delisted coins are absent from the dataset")
for drag in (0.10, 0.20, 0.30):
    # assume the long leg loses an extra `drag` per year (unobserved blow-ups)
    long_expo = w.clip(lower=0).sum(axis=1).shift(1).fillna(0)
    stressed = res.pnl - long_expo * drag / (365 * 24)
    m = L.metrics(stressed, *L.SPLITS["OOS"])
    print(f"long-leg drag {drag:.0%}/yr -> OOS CAGR {m['cagr']:.2%}, Sharpe {m['sharpe']:.2f}")

print("\n#### 9. Return decomposition per split (price / fees / funding)")
for s, (a, b) in L.SPLITS.items():
    gross = res.gross.loc[a:b].sum()
    cost = (res.turnover.shift(1).fillna(0) * L.FEE).loc[a:b].sum()
    fund = -(w.shift(1).fillna(0) * p["funding"]).sum(axis=1).loc[a:b].sum()
    print(f"{s}: price {gross:+.3f}  fees {-cost:+.3f}  funding {fund:+.3f}  (sum of bar returns)")

res.pnl.to_frame("pnl").to_parquet("candidate_pnl.parquet")
res_h.pnl.to_frame("pnl").to_parquet("replica_pnl.parquet")
