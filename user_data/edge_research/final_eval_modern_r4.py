"""Round 4 final evaluation - opens the 2024-01..2025-05 holdout ONCE for the new stream M.

Locked on the design window before this run (screen_modern_r4c.py):
  M pump short: short top-100 coins (ex BTC/ETH) after a >50% 24h rise, hold 2 days,
                5% of equity each (book <= 50%), BTC long hedge.
New portfolio = round-3 streams A, B, C + M, daily inverse-vol mix (same as round 3).
Rejected at the design gate (never opened on holdout): E, F, G (5 trades), H, I, J, K, L.
"""

import itertools

import edge_lab as L
import modern2_lab as M2
import modern_lab as M
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
rng = np.random.default_rng(4)
p = L.load_panel()
DA, DB = M.SPLITS_MODERN["DESIGN"]
HA, HB = M.SPLITS_MODERN["HOLDOUT"]


def daily(r):
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


def stats(d):
    eq = (1 + d).cumprod()
    return {
        "sharpe": d.mean() / d.std() * np.sqrt(365),
        "cagr": eq.iloc[-1] ** (365 / len(d)) - 1,
        "maxdd": (eq / eq.cummax() - 1).min(),
    }


def build(fee=L.FEE, m=(0.5, 2, 0.05)):
    ws = {
        "A": M.dilution_short(p, 30, 1),
        "B": M.btc_vs_alts(p, 20),
        "C": M.majors_trend(p, "donchian", 20, 55, 0.15),
        "M": M2.pump_short(p, m[0], m[1], size=m[2]),
    }
    return {k: daily(L.run(k, w, p, fee=fee).pnl).loc["2022":] for k, w in ws.items()}, ws


S, W = build()
S2, _ = build(fee=2 * L.FEE)
r3, _ = M.inv_vol_mix({k: S[k] for k in "ABC"})
r4, wts = M.inv_vol_mix(S)
r4_2x, _ = M.inv_vol_mix(S2)

print("#### 1. Stream M and portfolios: design vs holdout (holdout opened once)")
rows = []
for name, d in {
    "M pump short": S["M"],
    "round-3 portfolio (A+B+C)": r3,
    "round-4 portfolio (A+B+C+M)": r4,
    "round-4 portfolio, 2x costs": r4_2x,
}.items():
    for split, (a, b) in M.SPLITS_MODERN.items():
        rows.append({"series": name, "split": split, **stats(d.loc[a:b])})
print(pd.DataFrame(rows).set_index(["series", "split"]).round(3).to_string())

print("\n#### 2. Calendar years")
print(
    pd.DataFrame(
        {
            "M": M.yearly(S["M"]),
            "round3": M.yearly(r3),
            "round4": M.yearly(r4),
            "round4_2x": M.yearly(r4_2x),
        }
    )
    .round(4)
    .to_string()
)

m = r4.groupby(r4.index.to_period("M")).apply(lambda v: (1 + v).prod() - 1).loc["2024-01":]
print(
    f"\nround-4 holdout months positive {(m > 0).mean():.0%}; worst {m.min():.2%}; "
    f"total {(1 + m).prod() - 1:.2%}; without best month {(1 + m.drop(m.idxmax())).prod() - 1:.2%}"
)
print("avg weights (holdout):", wts.loc[HA:].mean().round(3).to_dict())
ent = (W["M"].drop(columns=[M.BTC]) < 0) & (W["M"].drop(columns=[M.BTC]).shift(1).fillna(0) == 0)
print(
    f"M entries: design {int(ent.loc[DA:DB].sum().sum())}, holdout {int(ent.loc[HA:].sum().sum())}"
)
print("daily corr of M to A/B/C (2022+):", pd.DataFrame(S).corr()["M"].round(2).to_dict())

print("\n#### 3. Plateau of M on the holdout (all design-plateau neighbours)")
pl = []
for j, h, sz in itertools.product((0.5, 0.6, 0.75), (1, 2, 3), (0.05, 0.1)):
    d = daily(L.run("", M2.pump_short(p, j, h, size=sz), p).pnl)
    pl.append({"jump": j, "hold": h, "size": sz, **stats(d.loc[HA:HB])})
pl = pd.DataFrame(pl)
print(pl.round(3).to_string())
print(
    f"holdout: {(pl.cagr > 0).mean():.0%} of {len(pl)} neighbours profitable; Sharpe median {pl.sharpe.median():.2f}"
)

print("\n#### 4. Bootstrap max DD of the round-4 portfolio 2022..2025 (10-day blocks)")
v = r4.to_numpy()
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
pd.DataFrame({"round3": r3, "round4": r4, "round4_2x": r4_2x, **S}).to_parquet(
    "modern_r4_pnl.parquet"
)
