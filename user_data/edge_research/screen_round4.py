"""Round 4: structural edges. Selection on IS+VAL; per-year table printed for transparency.

R. Funding carry harvest: long spot + short perp (delta-neutral) on high-funding coins.
S. Time-series trend following with volatility targeting (long/short perps).
T. Hybrid: trend positions filtered by funding (don't pay crowded funding to ride a trend).
"""

import edge_lab as L
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
p = L.load_panel()
close = p["close"]
fk = p["funding_known"]
U = L.universe(p, 50)
rows = []


def per_year(pnl):
    return ((1 + pnl).groupby(pnl.index.year).prod() - 1).loc[2021:]


def rec(name, pnl, turnover, expo):
    r = L.Result(name, pnl, pnl, turnover, expo, 0)
    df = L.report(r, splits=("IS", "VAL"), show=False)
    row = {"name": name}
    for s in ("IS", "VAL"):
        for m in ("sharpe", "cagr", "maxdd"):
            row[f"{s}_{m}"] = df.loc[s, m]
    row["turn/d"] = df["turn/d"].mean()
    row["expo"] = df["expo"].mean()
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


# ------------------------------------------------------------------ R. carry harvest
spot = {}
for f in sorted(L.DATA_DIR.glob("*_USDT-1h.feather")):
    pair = f.name.replace("-1h.feather", "") + "_USDT"
    if pair in close.columns:
        spot[pair] = pd.read_feather(f).drop_duplicates("date").set_index("date")["close"]
spot = pd.DataFrame(spot).reindex(index=close.index, columns=close.columns)
has_spot = spot.notna()
r_perp = close.pct_change(fill_method=None).fillna(0)
r_spot = spot.pct_change(fill_method=None).fillna(0)
SPOT_FEE = 0.001 + 0.0002

for k_days in (1, 3):
    favg = fk.rolling(24 * k_days, min_periods=8).mean()
    for thr_in, thr_out in ((0.0001, 0.00005), (0.0002, 0.0001), (0.0003, 0.00015)):
        for top_k in (5, 10):
            elig = U & has_spot
            sig = favg.where(elig)
            rank = sig.rank(axis=1, ascending=False)
            state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
            state[(rank <= top_k) & (sig > thr_in)] = 1.0
            state[(sig < thr_out) | sig.isna()] = 0.0
            pos = state.ffill().fillna(0)
            # equal weight across held pairs, max 1/top_k each, notional per leg
            w = pos / top_k
            held = w.shift(1).fillna(0)
            dw = (w - w.shift(1).fillna(0)).abs()
            pnl = (
                (held * (r_spot - r_perp)).sum(axis=1)
                + (held * p["funding"]).sum(axis=1)  # short perp receives funding
                - dw.sum(axis=1).shift(1).fillna(0) * (L.FEE + SPOT_FEE)
            )
            # capital: spot leg fully funded + perp margin at 1x -> divide by 2 for return on capital
            pnl = pnl / 2
            rec(
                f"R_carry_k{k_days}_in{thr_in}_top{top_k}",
                pnl,
                dw.sum(axis=1) / 2,
                held.sum(axis=1),
            )

# ------------------------------------------------------------------ S. trend following
ret = close.pct_change(fill_method=None)
c4 = close.resample("4h", label="left", closed="left").last()
vol = ret.rolling(24 * 20, min_periods=24 * 10).std() * np.sqrt(24 * 365)
for majors in (10, 30):
    Um = L.universe(p, majors)
    for lb_days in (10, 20, 50):
        lb = 24 * lb_days
        ema_f = close.ewm(span=lb // 4, min_periods=lb // 4).mean()
        ema_s = close.ewm(span=lb, min_periods=lb).mean()
        for mode in ("ls", "long"):
            sig = np.sign(ema_f - ema_s)
            if mode == "long":
                sig = sig.clip(lower=0)
            # inverse-vol sizing, each coin targets 40%/sqrt(N) annual vol -> ~40% book vol cap
            raw = sig.where(Um) * (0.4 / vol) / majors
            w = raw.clip(-0.25, 0.25).fillna(0)
            gross = w.abs().sum(axis=1)
            w = w.div(np.maximum(gross, 2.0) / 2.0, axis=0)  # cap gross leverage at 2x
            w = L.staggered(w, 24, 4)
            r = L.run(f"S_trend_top{majors}_{lb_days}d_{mode}", w, p)
            rec(r.name, r.pnl, r.turnover, r.exposure)

            # T. hybrid: drop trend legs that pay extreme funding (crowded side)
            f3 = fk.rolling(72, min_periods=8).mean()
            pay = (np.sign(w) * f3) > 0.0003
            wt = w.where(~pay, 0.0)
            r = L.run(f"T_trend_fundfilter_top{majors}_{lb_days}d_{mode}", wt, p)
            rec(r.name, r.pnl, r.turnover, r.exposure)

res = pd.DataFrame(rows).set_index("name")
print(res.round(3).to_string())
res.to_csv("round4.csv")
