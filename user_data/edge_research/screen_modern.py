"""Round 3 screen on the DESIGN window only (2022-01 .. 2023-12). The holdout is never touched."""

import itertools

import edge_lab as L
import intraday_lab as I
import modern_lab as M
import pandas as pd


pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
A, B = M.SPLITS_MODERN["DESIGN"]
p = L.load_panel()
rows = []


def rec(fam, name, r, trades=None):
    r = r.loc[A:B]
    m = L.metrics(r)
    yr = M.yearly(r)
    row = {
        "family": fam,
        "name": name,
        "sharpe": m["sharpe"],
        "cagr": m["cagr"],
        "maxdd": m["maxdd"],
        "y2022": yr.get(2022),
        "y2023": yr.get(2023),
        "trades": trades,
    }
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


# A. dilution short
for n_days, skip in itertools.product((30, 60, 90), (1, 7)):
    w = M.dilution_short(p, n_days=n_days, skip_days=skip)
    r = L.run("", w, p)
    ent = int(((w < 0) & (w.shift(1).fillna(0) >= 0)).loc[A:B].sum().sum())
    rec("A_dilution", f"n{n_days}_skip{skip}", r.pnl, ent)
    rec("A_dilution", f"n{n_days}_skip{skip}_2xfee", L.run("", w, p, fee=2 * L.FEE).pnl, ent)

# B. BTC vs alts
for ma in (10, 20, 50):
    w = M.btc_vs_alts(p, ma_days=ma)
    rec("B_btc_vs_alts", f"ma{ma}", L.run("", w, p).pnl)
rec("B_btc_vs_alts", "always_on", L.run("", M.btc_vs_alts(p, always=True), p).pnl)

# C. majors trend
for rule, f, s in (("ema", 5, 20), ("ema", 10, 50), ("donchian", 10, 20), ("donchian", 20, 55)):
    for tv in (0.15, 0.25):
        w = M.majors_trend(p, rule=rule, fast_days=f, slow_days=s, target_vol=tv)
        rec("C_majors_trend", f"{rule}_{f}_{s}_tv{tv}", L.run("", w, p).pnl)

# D. intraday reversion (maker)
for n, k, mb in itertools.product((24, 48, 96), (2.0, 3.0), (12, 36)):
    hourly, trades, R = I.basket(n=n, k=k, max_bars=mb)
    tr = int(trades)
    rec("D_intraday_rev", f"n{n}_k{k}_mb{mb}", hourly, tr)

res = pd.DataFrame(rows)
print(res.round(3).to_string())
res.round(4).to_csv("modern_screen.csv", index=False)
