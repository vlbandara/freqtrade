"""Round 6: rebuild the daily return series (2022-01 .. 2025-05) of every modern-regime trial
screened in rounds 3-4, plus the 27 locked-portfolio neighbours, for the overfitting audit.

Writes r6_trials.pkl (DataFrame: index = day, columns = trial name). The round-3 intraday
family (D) is left out because it needs 5m data; it was strongly negative anyway, so leaving
it out makes the audit *less* harsh, not more.
"""

import itertools
import os

import edge_lab as L
import modern2_lab as M2
import modern_lab as M
import pandas as pd
import r6_common as C6


p = L.load_panel()
OUT = os.environ.get("R6_TRIALS", os.path.join(os.path.dirname(__file__), "r6_trials.pkl"))
T = {}


def add(name, w):
    T[name] = C6.daily(L.run(name, w, p).pnl).loc["2022":]
    print(name, round(C6.stats(T[name].loc[:"2023"])["sharpe"], 3), flush=True)


A_w, B_w, C_w = {}, {}, {}
for n, s in itertools.product((30, 60, 90), (1, 7)):
    A_w[(n, s)] = M.dilution_short(p, n_days=n, skip_days=s)
    add(f"A_n{n}_skip{s}", A_w[(n, s)])
for n, e in itertools.product((30, 60), (24, 72, 168)):
    add(f"A2_n{n}_ema{e}", M.dilution_short_confirmed(p, n, e))
for ma in (10, 20, 50):
    B_w[ma] = M.btc_vs_alts(p, ma_days=ma)
    add(f"B_ma{ma}", B_w[ma])
add("B_always", M.btc_vs_alts(p, always=True))
for rule, f, s in (("ema", 5, 20), ("ema", 10, 50), ("donchian", 10, 20), ("donchian", 20, 55)):
    for tv in (0.15, 0.25):
        C_w[(rule, f, s, tv)] = M.majors_trend(p, rule=rule, fast_days=f, slow_days=s, target_vol=tv)
        add(f"C_{rule}{f}_{s}_tv{tv}", C_w[(rule, f, s, tv)])
for k, md in itertools.product((1.5, 2.5), (5, 10)):
    add(f"E_pullback_k{k}_md{md}", M2.pullback(p, k=k, max_days=md))
for f, s in ((20, 55), (10, 30), (30, 90)):
    add(f"F_ratio_{f}_{s}", M2.ratio_trend(p, f, s))
for k, h in itertools.product((2.0, 3.0), (1, 3)):
    add(f"G_shock_k{k}_h{h}", M2.shock_reversal(p, k=k, hold_days=h))
for n in (10, 20):
    add(f"H_broad_top{n}", M2.broad_trend(p, n=n))
for mx in (180, 365):
    add(f"I_young_{mx}", M2.young_short(p, 30, mx))
for mode in ("btc_long", "alt_short"):
    add(f"J_weekend_{mode}", M2.weekend(p, mode))
for b, a in ((1, 3), (2, 2), (0, 5)):
    add(f"K_tom_{b}_{a}", M2.turn_of_month(p, b, a))
for pct in (0.1, 0.2, 0.3):
    add(f"L_fundcap_{pct}", M2.funding_capitulation(p, pct))
for j, h in itertools.product((0.2, 0.3, 0.4, 0.5, 0.6, 0.75), (1, 2, 3, 7)):
    add(f"M_pump_j{j}_h{h}", M2.pump_short(p, j, h))

# the 27 neighbours of the locked portfolio, same grid as final_eval_modern.py section 5
pa = {k: T[f"A_n{k[0]}_skip{k[1]}"] for k in A_w}
for a, b, c in itertools.product(((30, 1), (60, 1), (30, 7)), (10, 20, 50), (
    ("donchian", 20, 55, 0.15), ("donchian", 10, 20, 0.15), ("ema", 5, 20, 0.15))):
    cname = f"C_{c[0]}{c[1]}_{c[2]}_tv{c[3]}"
    port, _ = M.inv_vol_mix({"A": pa[a], "B": T[f"B_ma{b}"], "C": T[cname]})
    T[f"P_A{a[0]}s{a[1]}_B{b}_{cname}"] = port
    print("portfolio", a, b, c, round(C6.stats(port.loc[:"2023"])["sharpe"], 3), flush=True)

D = pd.DataFrame(T).loc["2022-01-01":"2025-05-29"].fillna(0.0)
D.to_pickle(OUT)
print("saved", D.shape, "->", OUT)
