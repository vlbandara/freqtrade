"""Carry v2 screen: engine change + capital-efficiency variants (IS + VAL shown; OOS for v1 rules
only, which were already seen in round 1)."""

import time

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import pandas as pd


pd.set_option("display.width", 250)
p = L.load_panel()
spot = C.load_spot(p["close"])
U = {n: L.universe(p, n) for n in (50, 100)}
rows = []


def rec(name, ret, gross, splits=("IS", "VAL")):
    row = {"name": name}
    for s in splits:
        m = L.metrics(ret, *L.SPLITS[s])
        row.update({f"{s}_sh": m["sharpe"], f"{s}_cagr": m["cagr"], f"{s}_dd": m["maxdd"]})
    row["gross"] = gross.loc["2021":].mean()
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


t0 = time.time()
w = C2.target_weights(p, spot, U[50])
r, comp, g = C2.simulate(p, spot, w)
print("sim seconds", round(time.time() - t0, 1))
rec("v1rules_v2engine", r, g, ("IS", "VAL", "OOS"))
for nm, (u, k, ti, to, sz) in {
    "top100": (100, 10, 1e-4, 5e-5, "equal"),
    "top100_k20": (100, 20, 1e-4, 5e-5, "equal"),
    "top50_fundw": (50, 20, 1e-4, 5e-5, "fund"),
    "top100_fundw": (100, 20, 1e-4, 5e-5, "fund"),
    "top100_fundw_thr2": (100, 20, 2e-4, 1e-4, "fund"),
    "top100_k20_thr2": (100, 20, 2e-4, 1e-4, "equal"),
}.items():
    w = C2.target_weights(p, spot, U[u], top_k=k, thr_in=ti, thr_out=to, sizing=sz)
    for costs in ("taker", "vip_bnb"):
        r, comp, g = C2.simulate(p, spot, w, costs=costs)
        rec(f"{nm}_{costs}", r, g)
pd.DataFrame(rows).set_index("name").round(3).to_csv("carry2_screen.csv")
print(pd.DataFrame(rows).set_index("name").round(3).to_string())
