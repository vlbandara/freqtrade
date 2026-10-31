"""Round 4 screen on the DESIGN window only (2022-01 .. 2023-12). Holdout untouched.

Entry gate: Sharpe >= 0.8, 2022 and 2023 positive, positive at 2x costs, corr < 0.5 to A/B/C.
"""

import edge_lab as L
import modern2_lab as M2
import modern_lab as M
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
A_, B_ = M.SPLITS_MODERN["DESIGN"]
p = L.load_panel()


def daily(r):
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


base = {
    "A": daily(L.run("", M.dilution_short(p, 30, 1), p).pnl),
    "B": daily(L.run("", M.btc_vs_alts(p, 20), p).pnl),
    "C": daily(L.run("", M.majors_trend(p, "donchian", 20, 55, 0.15), p).pnl),
}
BASE = pd.DataFrame(base).loc[A_:B_]
rows = []


def rec(fam, name, w):
    d1 = daily(L.run("", w, p).pnl).loc[A_:B_]
    d2 = daily(L.run("", w, p, fee=2 * L.FEE).pnl).loc[A_:B_]
    eq = (1 + d1).cumprod()
    yr = M.yearly(d1)
    entries = int(((w != 0) & (w.shift(1).fillna(0) == 0)).loc[A_:B_].sum().sum())
    row = {
        "family": fam,
        "name": name,
        "sharpe": d1.mean() / d1.std() * np.sqrt(365) if d1.std() > 0 else 0,
        "cagr": eq.iloc[-1] ** (365 / len(d1)) - 1,
        "maxdd": (eq / eq.cummax() - 1).min(),
        "y2022": yr.get(2022),
        "y2023": yr.get(2023),
        "sharpe_2x": d2.mean() / d2.std() * np.sqrt(365) if d2.std() > 0 else 0,
        "corr_A": d1.corr(BASE["A"]),
        "corr_B": d1.corr(BASE["B"]),
        "corr_C": d1.corr(BASE["C"]),
        "entries": entries,
    }
    row["gate"] = bool(
        row["sharpe"] >= 0.8
        and row["y2022"] > 0
        and row["y2023"] > 0
        and row["sharpe_2x"] > 0
        and max(abs(row["corr_A"]), abs(row["corr_B"]), abs(row["corr_C"])) < 0.5
    )
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


for k in (1.5, 2.5):
    for md in (5, 10):
        rec("E_pullback", f"k{k}_max{md}d", M2.pullback(p, k=k, max_days=md))
for f, s in ((20, 55), (10, 30), (30, 90)):
    rec("F_ratio_trend", f"donchian{f}_{s}", M2.ratio_trend(p, f, s))
for k in (2.0, 3.0):
    for h in (1, 3):
        rec("G_shock_reversal", f"k{k}_hold{h}d", M2.shock_reversal(p, k=k, hold_days=h))
for n in (10, 20):
    rec("H_broad_trend [holdout partly seen]", f"top{n}_donchian20_55", M2.broad_trend(p, n=n))
for mx in (180, 365):
    rec("I_young_short [holdout partly seen]", f"age30_{mx}d", M2.young_short(p, 30, mx))

res = pd.DataFrame(rows)
print(res.round(3).to_string())
res.round(4).to_csv("modern_r4_screen.csv", index=False)
