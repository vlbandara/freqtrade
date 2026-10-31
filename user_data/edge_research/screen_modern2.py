"""Round 3 refinement on the DESIGN window only: A2 (confirmed dilution short) + A/B/C mixes."""

import itertools

import edge_lab as L
import modern_lab as M
import numpy as np
import pandas as pd


A, B = M.SPLITS_MODERN["DESIGN"]
p = L.load_panel()


def daily(r):
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


def stats(name, d):
    d = d.loc[A:B]
    eq = (1 + d).cumprod()
    sh = d.mean() / d.std() * np.sqrt(365)
    yr = M.yearly(d)
    print(
        f"{name:<44} Sharpe {sh:5.2f}  CAGR {eq.iloc[-1] ** (365 / len(d)) - 1:6.1%}  "
        f"maxDD {(eq / eq.cummax() - 1).min():6.1%}  2022 {yr.get(2022):6.1%}  2023 {yr.get(2023):6.1%}",
        flush=True,
    )


for n, e in itertools.product((30, 60), (24, 72, 168)):
    stats(
        f"A2_confirmed_n{n}_ema{e}h", daily(L.run("", M.dilution_short_confirmed(p, n, e), p).pnl)
    )

sA = daily(L.run("", M.dilution_short(p, 30, 1), p).pnl)
sB = daily(L.run("", M.btc_vs_alts(p, 20), p).pnl)
sC = daily(L.run("", M.majors_trend(p, "donchian", 20, 55, 0.15), p).pnl)
print("\ncorrelations (design window):")
print(pd.DataFrame({"A": sA, "B": sB, "C": sC}).loc[A:B].corr().round(2))
for name, st in {
    "A+B+C": {"A": sA, "B": sB, "C": sC},
    "A+C": {"A": sA, "C": sC},
    "B+C": {"B": sB, "C": sC},
}.items():
    mix, _ = M.inv_vol_mix(st)
    stats("mix " + name, mix)
