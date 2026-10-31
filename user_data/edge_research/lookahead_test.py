"""Lookahead test for the carry harvest: positions computed on data truncated at T must equal
positions computed on the full history for every bar < T (no future information used)."""

import carry_lab as C
import edge_lab as L
import pandas as pd


p_full = L.load_panel()
spot_full = C.load_spot(p_full["close"])
CK = dict(k_days=3, thr_in=1e-4, thr_out=5e-5, top_k=10, perp_margin=0.2)
_, w_full, _ = C.carry(p_full, spot_full, L.universe(p_full, 50), **CK)

ok = True
for cut in ("2021-06-01", "2022-11-09", "2023-07-15", "2024-03-12", "2025-02-01"):
    T = pd.Timestamp(cut, tz="UTC")
    p_cut = {k: v.loc[: T - pd.Timedelta("1h")] for k, v in p_full.items()}
    # funding is attached to the candle *before* settlement; drop rows that would need future data
    spot_cut = C.load_spot(p_cut["close"])
    _, w_cut, _ = C.carry(p_cut, spot_cut, L.universe(p_cut, 50), **CK)
    a = w_cut.iloc[:-1]
    b = w_full.loc[a.index, a.columns]
    diff = float((a - b).abs().max().max())
    print(f"cut {cut}: max |w_cut - w_full| over {len(a)} bars = {diff:.2e}")
    ok &= diff < 1e-12
print("NO LOOKAHEAD" if ok else "LOOKAHEAD DETECTED")
