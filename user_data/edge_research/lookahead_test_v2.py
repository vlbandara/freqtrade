"""Truncation lookahead test for round-2 streams (carry v2 target weights, cascade reversion):
weights computed on data cut at T must equal full-history weights for every bar before T."""

import carry2_lab as C2
import carry_lab as C
import cascade_lab as CL
import edge_lab as L
import pandas as pd


p_full = L.load_panel()
spot_full = C.load_spot(p_full["close"])


def build(p, spot):
    U100, U50 = L.universe(p, 100), L.universe(p, 50)
    wc = C2.target_weights(p, spot, U100, top_k=10, thr_in=1e-4, thr_out=5e-5)
    ws, _ = CL.cascade_weights(p, U50, flush=-0.03, hold=6, nworst=5)
    return wc, ws


wc_full, ws_full = build(p_full, spot_full)
ok = True
for cut in ("2021-05-19 14:00", "2022-06-13", "2023-03-10", "2024-08-05 03:00", "2025-04-07"):
    T = pd.Timestamp(cut, tz="UTC")
    p_cut = {k: v.loc[: T - pd.Timedelta("1h")] for k, v in p_full.items()}
    wc, ws = build(p_cut, C.load_spot(p_cut["close"]))
    for nm, a, b in (("carry_v2", wc, wc_full), ("cascade", ws, ws_full)):
        a = a.iloc[:-1]
        d = float((a - b.loc[a.index, a.columns]).abs().max().max())
        print(f"cut {cut} {nm:<9}: max |w_cut - w_full| = {d:.2e}")
        ok &= d < 1e-12
print("NO LOOKAHEAD" if ok else "LOOKAHEAD DETECTED")
