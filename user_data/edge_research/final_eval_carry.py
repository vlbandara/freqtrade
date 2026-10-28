"""Robustness battery for the funding carry harvest (parameters locked on IS+VAL before OOS)."""

import carry_lab as C
import edge_lab as L
import pandas as pd


pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
p = L.load_panel()
close = p["close"]
spot = C.load_spot(close)
U50 = L.universe(p, 50)
CK = dict(k_days=3, thr_in=1e-4, thr_out=5e-5, top_k=10, perp_margin=0.2)


def rep(name, x):
    return L.report(L.Result(name, x, x, x * 0, x * 0, 0))


print("#### 1. Locked carry harvest")
pnl, w, comp = C.carry(p, spot, U50, **CK)
rep("CARRY locked", pnl)
pnl2, _, _ = C.carry(p, spot, U50, perp_fee=2 * L.FEE, spot_fee=2 * C.SPOT_FEE, **CK)
rep("CARRY at 2x fees+slippage", pnl2)
pnl3, _, _ = C.carry(p, spot, U50, perp_fee=3 * L.FEE, spot_fee=3 * C.SPOT_FEE, **CK)
rep("CARRY at 3x fees+slippage", pnl3)

entries = ((w > 0) & (w.shift(1).fillna(0) == 0)).sum()
print(
    f"\nround trips (entries): total {int(entries.sum())}, OOS "
    f"{int(((w > 0) & (w.shift(1).fillna(0) == 0)).loc['2024':].sum().sum())}; coins traded {int((entries > 0).sum())}"
)
print("avg holding (h):", round(float((w > 0).sum().sum() / max(entries.sum(), 1)), 1))

print("\n#### 2. Calendar years + components")
yrs = pd.DataFrame({"net": (1 + pnl).groupby(pnl.index.year).prod() - 1})
yrs = yrs.join(comp.groupby(comp.index.year).sum())
yrs["avg_deployed"] = w.sum(axis=1).groupby(w.index.year).mean()
print(yrs.round(4).to_string())

print("\n#### 3. Concentration: top coins by contribution (OOS)")
held = w.shift(1).fillna(0)
r_perp = close.pct_change(fill_method=None).fillna(0)
r_spot = spot.pct_change(fill_method=None).fillna(0)
contrib = (held * (r_spot - r_perp) + held * p["funding"]).loc["2024":].sum() / 1.2
print(contrib.sort_values(ascending=False).head(10).round(4).to_string())
tot = contrib.sum()
print(f"total {tot:.4f}; share of top-3 coins {contrib.nlargest(3).sum() / tot:.0%}")
ex3 = contrib.drop(contrib.nlargest(3).index).sum()
print(f"OOS gross contribution excluding top-3 coins: {ex3:.4f}")

print("\n#### 4. Plateau around every parameter (IS / VAL / OOS Sharpe and CAGR)")
rows = []
for k in (1, 3, 7):
    for thr in (0.5e-4, 1e-4, 2e-4):
        for top_k in (5, 10, 20):
            x, _, _ = C.carry(
                p, spot, U50, k_days=k, thr_in=thr, thr_out=thr / 2, top_k=top_k, perp_margin=0.2
            )
            row = dict(k_days=k, thr_in=thr, top_k=top_k)
            for s in ("IS", "VAL", "OOS"):
                m = L.metrics(x, *L.SPLITS[s])
                row[f"{s}_sh"] = m["sharpe"]
                row[f"{s}_cagr"] = m["cagr"]
            rows.append(row)
pl = pd.DataFrame(rows)
print(pl.round(3).to_string())
for s in ("IS", "VAL", "OOS"):
    print(
        f"{s}: profitable {(pl[f'{s}_cagr'] > 0).mean():.0%}, Sharpe median {pl[f'{s}_sh'].median():.2f}"
        f" min {pl[f'{s}_sh'].min():.2f}"
    )

print("\n#### 5. Leverage: gross hedged notional per unit of capital")
# margin needed for the short perp leg under portfolio margin uses the spot as collateral;
# we keep 20% of notional as buffer. Leverage L scales pnl; worst adverse basis move is shown.
for lev in (1, 2, 3):
    x = pnl * lev
    m = {s: L.metrics(x, *L.SPLITS[s]) for s in L.SPLITS}
    print(f"{lev}x: " + "  ".join(f"{s} CAGR {m[s]['cagr']:.1%} DD {m[s]['maxdd']:.1%}" for s in m))
worst_basis = (r_spot - r_perp).where(held > 0).min().nsmallest(5)
print("worst 1h basis moves on held coins:\n", worst_basis.round(4).to_string())
worst_pump = close.pct_change(24, fill_method=None).where(held > 0).max().nlargest(5)
print(
    "largest 24h pump on a held short-perp leg (margin stress):\n", worst_pump.round(3).to_string()
)

pd.DataFrame({"carry": pnl}).to_parquet("carry_pnl.parquet")
