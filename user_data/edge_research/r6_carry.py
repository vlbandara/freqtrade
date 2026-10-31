"""Round 6: carry harvest upgrades.

V0  locked round-1 rules (3d average of the *per-settlement* funding rate, 0.01% in / 0.005% out
    per 8h, top 10, top-50 universe).
V1  bug fix: Binance moved many coins to 4h (and some to 1h) funding in 2023-25. The rate per
    settlement then understates the carry per 8h by 2-8x, and these are exactly the coins with
    extreme funding. V1 uses the funding actually *paid* over the window, expressed per 8h.
V2  forecast: walk-forward LightGBM predicts each coin's funding over the next 7 days (per 8h)
    from trailing funding, basis, returns, volatility and settlement frequency. Same thresholds
    applied to the forecast.
V3  V2 with a cost hurdle: enter only if the forecast 7-day carry exceeds the round-trip cost
    (both legs, both sides) by 1.5x; exit when the forecast falls below the V0 exit level.

Design on 2020-2023 (IS + VAL). The 2024-01..2025-05 OOS has been seen for V0 in earlier
rounds, but not for V1-V3; it is opened once with --oos.
"""

import sys

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import lightgbm as lgb
import numpy as np
import pandas as pd
import r6_common as C6


pd.set_option("display.width", 250)
OOS = "--oos" in sys.argv
p = L.load_panel()
close = p["close"]
spot = C.load_spot(close)
U = L.universe(p, 50)
elig = U & spot.notna() & close.notna()

# funding actually paid, known at its settlement time (p["funding"] sits one bar earlier)
paid = p["funding"].shift(1).fillna(0.0)
n_settle = (paid != 0).astype(float)


def per8h(hours):
    return paid.rolling(hours, min_periods=hours // 3).sum() / (hours / 8)


def book(sig, thr_in=1e-4, thr_out=5e-5, top_k=10, enter_mask=None):
    s = sig.where(elig)
    rank = s.rank(axis=1, ascending=False)
    state = pd.DataFrame(np.nan, index=s.index, columns=s.columns)
    ent = (rank <= top_k) & (s > thr_in)
    if enter_mask is not None:
        ent &= enter_mask
    state[ent] = 1.0
    state[(s < thr_out) | s.isna()] = 0.0
    pos = state.ffill().fillna(0)
    return pos.div(np.maximum(pos.sum(axis=1), top_k), axis=0)


# ---------------------------------------------------------------- V2 forecast model
fk = p["funding_known"]
feat = {
    "f8_1d": per8h(24), "f8_3d": per8h(72), "f8_7d": per8h(168), "f8_14d": per8h(336),
    "last_rate": fk, "settle_per_day": n_settle.rolling(72, min_periods=24).sum() / 3,
    "basis_24h": (close / spot - 1).rolling(24, min_periods=12).mean(),
    "ret_3d": close.pct_change(72, fill_method=None), "ret_14d": close.pct_change(336, fill_method=None),
    "vol_3d": close.pct_change(fill_method=None).rolling(72, min_periods=24).std(),
}
fwd7 = paid[::-1].rolling(168, min_periods=1).sum()[::-1].shift(-1) / 21  # next 7d, per 8h
sample = close.index[close.index.hour % 8 == 0]
Xs = pd.DataFrame({k: v.where(elig).loc[sample].stack(future_stack=True) for k, v in feat.items()})
Xs = Xs[Xs["f8_3d"].notna()]
ys = fwd7.loc[sample].stack(future_stack=True).reindex(Xs.index)
t_idx = Xs.index.get_level_values(0)
PAR = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300,
           subsample=0.7, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)
pred = []
end = "2025-06-01" if OOS else "2024-01-01"
for q in pd.date_range("2021-01-01", end, freq="QS", tz="UTC"):
    cutoff = q - pd.Timedelta(days=8)
    tr = (t_idx < cutoff) & ys.notna().to_numpy()
    te = (t_idx >= q) & (t_idx < q + pd.offsets.QuarterBegin(1, startingMonth=1))
    if te.sum() == 0:
        continue
    m = lgb.LGBMRegressor(random_state=7, **PAR).fit(Xs[tr], ys[tr])
    pred.append(pd.Series(m.predict(Xs[te]), index=Xs.index[te]))
    print("refit", q.date(), int(tr.sum()), flush=True)
fc = pd.concat(pred).unstack().reindex(index=close.index, columns=close.columns).ffill(limit=7)
ev = pd.DataFrame({"pred": pd.concat(pred), "y": ys}).dropna()
ev_t = ev.index.get_level_values(0)
for nm, (a, b) in {"2021-23": ("2021", "2024"), "2024-25": ("2024", "2026")}.items():
    e = ev[(ev_t >= a) & (ev_t < b)]
    if len(e):
        base = Xs["f8_3d"].reindex(e.index)
        print(f"forecast {nm}: corr(pred, fwd7)={e.pred.corr(e.y):.3f}  corr(trailing 3d, fwd7)={base.corr(e.y):.3f}  "
              f"MAE pred {np.abs(e.pred - e.y).mean() * 1e4:.2f}bp vs trailing {np.abs(base - e.y).mean() * 1e4:.2f}bp")

rt_cost = 2 * (C2.COSTS["taker"][0] + C2.COSTS["taker"][1])  # round trip, both legs
hurdle = fc * 21 > 1.5 * rt_cost
books = {
    "V0_locked": C2.target_weights(p, spot, U, top_k=10),
    "V1_paid_per8h": book(per8h(72)),
    "V2_forecast": book(fc),
    "V3_forecast_hurdle": book(fc, enter_mask=hurdle),
}
rows = []
splits = {"IS": L.SPLITS["IS"], "VAL": L.SPLITS["VAL"]}
if OOS:
    splits["OOS"] = L.SPLITS["OOS"]
for name, w in books.items():
    for costs in ("taker", "taker2x"):
        r, comp, g = C2.simulate(p, spot, w, lev=1, borrow_apr=0.08, costs=costs)
        row = {"variant": name, "costs": costs}
        for s, (a, b) in splits.items():
            m = L.metrics(r, a, b)
            row.update({f"{s}_cagr": m["cagr"], f"{s}_sh": m["sharpe"], f"{s}_dd": m["maxdd"]})
        yr = (1 + r).groupby(r.index.year).prod() - 1
        row.update({f"y{y}": yr.get(y) for y in range(2021, 2026 if OOS else 2024)})
        row["avg_gross"] = g.loc["2021":end].mean()
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
res = pd.DataFrame(rows)
print(res.round(3).to_string())
res.to_csv("r6_carry_oos.csv" if OOS else "r6_carry_design.csv", index=False)
pd.to_pickle(books, L.CACHE / "r6_carry_books.pkl")
