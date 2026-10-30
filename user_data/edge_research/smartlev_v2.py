"""Smart leverage (borrow only where funding APR > borrow APR + 5%) vs blind leverage."""

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import pandas as pd


p = L.load_panel()
spot = C.load_spot(p["close"])
U50 = L.universe(p, 50)
rows = []
for apr in (0.08, 0.15):
    for ml in (2, 3, 4):
        w = C2.smart_leverage_weights(p, spot, U50, max_lev=ml, borrow_apr=apr)
        for costs in ("taker", "taker2x"):
            r, comp, g = C2.simulate(p, spot, w, lev=1, borrow_apr=apr, costs=costs)
            row = {
                "borrow_apr": apr,
                "max_lev": ml,
                "costs": costs,
                "avg_gross": g.loc["2021":].mean(),
            }
            for s in ("IS", "VAL", "OOS"):
                m = L.metrics(r, *L.SPLITS[s])
                row.update({f"{s}_cagr": m["cagr"], f"{s}_sh": m["sharpe"], f"{s}_dd": m["maxdd"]})
            yr = (1 + r).groupby(r.index.year).prod() - 1
            row.update({f"y{y}": yr.get(y) for y in range(2021, 2026)})
            rows.append(row)
            print(
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                flush=True,
            )
            if apr == 0.08 and ml == 3 and costs == "taker":
                pd.DataFrame({"ret": r}).to_parquet("smartlev3_pnl.parquet")
pd.DataFrame(rows).to_csv("smartlev_v2.csv", index=False)
