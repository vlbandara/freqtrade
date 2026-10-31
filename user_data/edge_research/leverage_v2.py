"""Carry v2 at 1-3x with USDT borrow cost for the spot leg (0 / 8 / 15% APR). Same rules as v1."""

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import pandas as pd


p = L.load_panel()
spot = C.load_spot(p["close"])
w = C2.target_weights(p, spot, L.universe(p, 50), top_k=10)
rows = []
for lev in (1, 1.5, 2, 3):
    for apr in (0.0, 0.08, 0.15):
        r, comp, g = C2.simulate(p, spot, w, lev=lev, borrow_apr=apr)
        row = {"lev": lev, "borrow_apr": apr}
        for s in ("IS", "VAL", "OOS"):
            m = L.metrics(r, *L.SPLITS[s])
            row.update({f"{s}_cagr": m["cagr"], f"{s}_sh": m["sharpe"], f"{s}_dd": m["maxdd"]})
        yr = (1 + r).groupby(r.index.year).prod() - 1
        row.update({f"y{y}": yr.get(y) for y in range(2021, 2026)})
        rows.append(row)
        print({k: round(v, 3) for k, v in row.items()}, flush=True)
d = pd.DataFrame(rows)
d.to_csv("leverage_v2.csv", index=False)
