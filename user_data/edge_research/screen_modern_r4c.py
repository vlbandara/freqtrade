"""Round 4, design-only plateau check for M (pump short) around the 50% threshold."""

import modern2_lab as M2
import pandas as pd


src = open("screen_modern_r4.py").read().split("for k in (1.5, 2.5):")[0]
ns = {}
exec(compile(src, "screen_modern_r4_head", "exec"), ns)
rec, p, rows = ns["rec"], ns["p"], ns["rows"]
for j in (0.4, 0.5, 0.6, 0.75):
    for h in (1, 2, 3):
        for sz in (0.05, 0.1):
            rec("M_pump_short", f"jump{j}_hold{h}d_size{sz}", M2.pump_short(p, j, h, size=sz))
res = pd.DataFrame(rows)
res.round(4).to_csv("modern_r4c_screen.csv", index=False)
pd.set_option("display.width", 260)
print(
    res[["name", "sharpe", "cagr", "maxdd", "y2022", "y2023", "sharpe_2x", "entries", "gate"]]
    .round(3)
    .to_string()
)
