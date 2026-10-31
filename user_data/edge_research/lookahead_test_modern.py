"""Truncation lookahead test for round-3 streams A, B, C."""

import edge_lab as L
import modern_lab as M
import pandas as pd


p_full = L.load_panel()


def build(p):
    return {
        "A_dilution": M.dilution_short(p, 30, 1),
        "B_btc_vs_alts": M.btc_vs_alts(p, 20),
        "C_majors": M.majors_trend(p, "donchian", 20, 55, 0.15),
    }


full = build(p_full)
ok = True
for cut in ("2022-05-10", "2022-11-09 12:00", "2023-08-17", "2024-03-14", "2025-04-07"):
    T = pd.Timestamp(cut, tz="UTC")
    cutw = build({k: v.loc[: T - pd.Timedelta("1h")] for k, v in p_full.items()})
    for k in full:
        a = cutw[k].iloc[:-1]
        d = float((a - full[k].loc[a.index, a.columns]).abs().max().max())
        print(f"cut {cut:<17} {k:<14} max diff {d:.2e}")
        ok &= d < 1e-12
print("NO LOOKAHEAD" if ok else "LOOKAHEAD DETECTED")
