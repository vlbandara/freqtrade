"""Round 3 final evaluation - opens the 2024-01..2025-05 holdout ONCE for the locked portfolio.

Locked on the design window (2022-2023) before this run:
  A dilution short  : n_days=30, skip_days=1   (short new listings, BTC hedge)
  B BTC vs alts     : ma_days=20               (long BTC / short EW alts while ALT/BTC < MA)
  C majors trend    : donchian 20/55, 15% vol target, BTC + ETH long/flat
  Portfolio         : daily inverse-vol mix, 90-day window, no leverage
Pass bar: holdout Sharpe >= 1.5; every year 2022..2025 positive; max DD <= 10%; >= 100 trades;
positive at 2x costs; positive without the best month; plateau; no lookahead.
"""

import itertools

import edge_lab as L
import modern_lab as M
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
rng = np.random.default_rng(3)
p = L.load_panel()
HA, HB = M.SPLITS_MODERN["HOLDOUT"]


def daily(r):
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


def streams(a=(30, 1), b=20, c=("donchian", 20, 55, 0.15), fee=L.FEE):
    wA = M.dilution_short(p, *a)
    wB = M.btc_vs_alts(p, b)
    wC = M.majors_trend(p, c[0], c[1], c[2], c[3])
    out = {
        k: daily(L.run(k, w, p, fee=fee).pnl).loc["2022":]
        for k, w in (("A", wA), ("B", wB), ("C", wC))
    }
    return out, {"A": wA, "B": wB, "C": wC}


def stats(d):
    eq = (1 + d).cumprod()
    return {
        "sharpe": d.mean() / d.std() * np.sqrt(365),
        "cagr": eq.iloc[-1] ** (365 / len(d)) - 1,
        "maxdd": (eq / eq.cummax() - 1).min(),
    }


S, W = streams()
port, wts = M.inv_vol_mix(S)
S2, _ = streams(fee=2 * L.FEE)
port2, _ = M.inv_vol_mix(S2)

print("#### 1. Holdout 2024-01 .. 2025-05 (opened once) and design window, per stream")
rows = []
for name, d in {**S, "PORTFOLIO": port, "PORTFOLIO 2x costs": port2}.items():
    for split, (a, b) in M.SPLITS_MODERN.items():
        rows.append({"stream": name, "split": split, **stats(d.loc[a:b])})
print(pd.DataFrame(rows).set_index(["stream", "split"]).round(3).to_string())

print("\n#### 2. Calendar years")
print(
    pd.DataFrame({k: M.yearly(v) for k, v in {**S, "PORTFOLIO": port, "PORT_2x": port2}.items()})
    .round(4)
    .to_string()
)

print("\n#### 3. Monthly (portfolio), best-month removal, average weights")
m = port.groupby(port.index.to_period("M")).apply(lambda v: (1 + v).prod() - 1)
print(m.round(4).to_string())
hm = m.loc["2024-01":]
print(f"holdout months positive {(hm > 0).mean():.0%}; worst {hm.min():.2%}; best {hm.max():.2%}")
print(
    f"holdout total {(1 + hm).prod() - 1:.2%}; without best month {(1 + hm.drop(hm.idxmax())).prod() - 1:.2%}"
)
print("avg stream weights (holdout):", wts.loc[HA:].mean().round(3).to_dict())

print("\n#### 4. Trade counts in holdout")
for k, w in W.items():
    nz = w.drop(columns=[M.BTC]) if k != "C" else w
    ent = ((nz != 0) & (nz.shift(1).fillna(0) == 0)).loc[HA:].sum().sum()
    print(f"  {k}: {int(ent)} position openings")

print("\n#### 5. Plateau: all neighbours of the locked settings (holdout portfolio)")
prow = []
for a, b, c in itertools.product(
    ((30, 1), (60, 1), (30, 7)),
    (10, 20, 50),
    (("donchian", 20, 55, 0.15), ("donchian", 10, 20, 0.15), ("ema", 5, 20, 0.15)),
):
    s_, _ = streams(a, b, c)
    pm, _ = M.inv_vol_mix(s_)
    prow.append(
        {
            "A": a,
            "B": b,
            "C": c[0] + f"{c[1]}/{c[2]}",
            **stats(pm.loc[HA:HB]),
            "design_sharpe": stats(
                pm.loc[M.SPLITS_MODERN["DESIGN"][0] : M.SPLITS_MODERN["DESIGN"][1]]
            )["sharpe"],
        }
    )
pl = pd.DataFrame(prow)
print(pl.round(3).to_string())
print(
    f"holdout: {(pl.cagr > 0).mean():.0%} of 27 neighbours profitable; Sharpe median {pl.sharpe.median():.2f}, min {pl.sharpe.min():.2f}"
)

print("\n#### 6. Bootstrap max drawdown, full 2022..2025 portfolio (10-day blocks)")
v = port.to_numpy()
dds = []
for _ in range(4000):
    idx = []
    while len(idx) < len(v):
        s0 = rng.integers(0, len(v))
        idx.extend(range(s0, min(s0 + 10, len(v))))
    eq = np.cumprod(1 + v[idx[: len(v)]])
    dds.append((eq / np.maximum.accumulate(eq) - 1).min())
print(
    f"median {np.median(dds):.1%}, 95th pct {np.percentile(dds, 5):.1%}, 99th pct {np.percentile(dds, 1):.1%}"
)

btc = p["close"][M.BTC].resample("1D").last().pct_change()
print(f"\ncorr to BTC (daily 2022+): {port.corr(btc.reindex(port.index)):.2f}")
pd.DataFrame({"portfolio": port, "portfolio_2x": port2, **S}).to_parquet("modern_pnl.parquet")
