"""Round 4, batch 2 screen on the DESIGN window only (same gate as screen_modern_r4.py)."""

import importlib.util

import modern2_lab as M2
import pandas as pd


spec = importlib.util.spec_from_file_location("s4", "screen_modern_r4.py")
# reuse rec()/gate from the batch-1 screen without re-running its grid
src = open("screen_modern_r4.py").read().split("for k in (1.5, 2.5):")[0]
ns = {}
exec(compile(src, "screen_modern_r4_head", "exec"), ns)
rec, p, rows = ns["rec"], ns["p"], ns["rows"]

for mode in ("btc_long", "alt_short"):
    rec("J_weekend", mode, M2.weekend(p, mode))
for b, a in ((1, 3), (2, 2), (0, 5)):
    rec("K_turn_of_month", f"before{b}_after{a}", M2.turn_of_month(p, b, a))
for pct in (0.1, 0.2, 0.3):
    rec("L_funding_capitulation", f"pct{pct}", M2.funding_capitulation(p, pct))
for j in (0.2, 0.3, 0.5):
    for h in (1, 3, 7):
        rec("M_pump_short", f"jump{j}_hold{h}d", M2.pump_short(p, j, h))
res = pd.DataFrame(rows)
pd.set_option("display.width", 260)
print(res.round(3).to_string())
res.round(4).to_csv("modern_r4b_screen.csv", index=False)
