"""One-shot OOS evaluation #2: trend + carry ensemble (locked on IS+VAL).

Locked: w_trend = 0.4
  trend: top-30 universe, EMA(5d) vs EMA(20d) sign, inverse-vol sized (40% book-vol target,
         gross <= 2x), drop legs whose 3d funding is > 0.03%/8h against us, daily rebalance x4 sleeves
  carry: top-50 universe w/ spot market, 3d avg funding > 0.01%/8h, top 10, exit < 0.005%/8h,
         long spot + short perp, capital = notional x 1.2 (portfolio-margin style)
"""

import carry_lab as C
import edge_lab as L
import ensemble_lab as E
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
rng = np.random.default_rng(11)
p = L.load_panel()
spot = C.load_spot(p["close"])
CK = dict(k_days=3, thr_in=1e-4, thr_out=5e-5, top_k=10, perp_margin=0.2)
WT = 0.4


def rep(name, x, splits=("IS", "VAL", "OOS")):
    return L.report(L.Result(name, x, x, x * 0, x * 0, 0), splits=splits)


print("#### 1. Locked ensemble and sleeves")
pnl, tp, cp = E.ensemble(p, spot, WT, CK)
rep("ENSEMBLE (0.4 trend / 0.6 carry)", pnl)
rep("  trend sleeve", tp)
rep("  carry sleeve", cp)
pnl2, _, _ = E.ensemble(p, spot, WT, CK, fee_mult=2.0)
rep("ENSEMBLE at 2x fees+slippage", pnl2)

print("\n#### 2. Calendar years")
yr = pd.DataFrame(
    {
        k: (1 + v).groupby(v.index.year).prod() - 1
        for k, v in {"ensemble": pnl, "trend": tp, "carry": cp, "ens_2xfee": pnl2}.items()
    }
)
print(yr.loc[2021:].round(3).to_string())

print("\n#### 3. Quarterly (ensemble)")
q = (1 + pnl).groupby(pnl.index.tz_localize(None).to_period("Q")).prod() - 1
print(q.loc["2021Q1":].round(3).to_string())
print(f"positive quarters: {(q.loc['2021Q1':] > 0).mean():.0%}")

print("\n#### 4. Plateau: neighbours of every locked parameter (OOS)")
rows = []
for lb in (10, 20, 30):
    for majors in (20, 30, 50):
        for thr in (1e-4, 2e-4):
            for wt in (0.3, 0.4, 0.5):
                ck = dict(CK, thr_in=thr, thr_out=thr / 2)
                x, _, _ = E.ensemble(p, spot, wt, ck, trend_kwargs=dict(majors=majors, lb_days=lb))
                m = L.metrics(x, *L.SPLITS["OOS"])
                rows.append(dict(lb=lb, majors=majors, thr=thr, w_trend=wt, **m))
pl = pd.DataFrame(rows)
print(pl.round(3).to_string())
print(
    f"OOS neighbours profitable: {(pl.ret > 0).mean():.0%}; Sharpe median {pl.sharpe.median():.2f}"
    f", min {pl.sharpe.min():.2f}, max {pl.sharpe.max():.2f}"
)

print("\n#### 5. Bootstrap 95% CI of OOS Sharpe (10d blocks)")
d = (1 + pnl).groupby(pnl.index.floor("D")).prod() - 1
do = d.loc[L.SPLITS["OOS"][0] :].values
sh = []
for _ in range(3000):
    idx = []
    while len(idx) < len(do):
        s = rng.integers(0, len(do))
        idx.extend(range(s, min(s + 10, len(do))))
    x = do[idx[: len(do)]]
    sh.append(x.mean() / x.std() * np.sqrt(365))
print(
    f"OOS Sharpe 95% CI [{np.percentile(sh, 2.5):.2f}, {np.percentile(sh, 97.5):.2f}];"
    f" P(Sharpe<=0) = {(np.array(sh) <= 0).mean():.3f}"
)

print("\n#### 6. Market exposure (daily, 2021+)")
btc = p["close"]["BTC_USDT_USDT"].resample("1D").last().pct_change()
j = pd.concat([d, btc], axis=1).dropna().loc["2021":]
beta = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()
print(f"corr to BTC {j.corr().iloc[0, 1]:.2f}, beta {beta:.3f}")
dt = (1 + tp).groupby(tp.index.floor("D")).prod() - 1
dc = (1 + cp).groupby(cp.index.floor("D")).prod() - 1
print(f"trend/carry daily corr OOS: {dt.loc['2024':].corr(dc.loc['2024':]):.2f}")

pd.DataFrame({"ensemble": pnl, "trend": tp, "carry": cp}).to_parquet("ensemble_pnl.parquet")
