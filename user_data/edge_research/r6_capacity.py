"""Round 6: capacity of the carry harvest.

For each entry / exit / re-size of the locked carry book, compare the traded notional of each
leg with that coin's trailing 24h quote volume on the same market (spot and perp). Reports the
AUM at which the median and the 90th-percentile trade would be 1% / 5% of daily volume, plus
the AUM-dependent cost uplift under a square-root impact model:
    impact (fraction) = Y * sigma_daily * sqrt(trade / ADV),  Y = 0.5 (conservative crypto value).
"""

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import numpy as np
import pandas as pd


pd.set_option("display.width", 250)
p = L.load_panel()
close = p["close"]
spot = C.load_spot(close)
sv = {}
for f in sorted(L.DATA_DIR.glob("*_USDT-1h.feather")):
    pair = f.name.replace("-1h.feather", "") + "_USDT"
    if pair in close.columns:
        d = pd.read_feather(f).drop_duplicates("date").set_index("date")
        sv[pair] = d["close"] * d["volume"]
spot_qv = pd.DataFrame(sv).reindex(index=close.index, columns=close.columns)
adv_spot = spot_qv.rolling(24, min_periods=12).sum()
adv_perp = p["quote_volume"].rolling(24, min_periods=12).sum()
sig_d = close.pct_change(fill_method=None).rolling(24 * 7, min_periods=48).std() * np.sqrt(24)

w = C2.target_weights(p, spot, L.universe(p, 50), top_k=10)
r, comp, g = C2.simulate(p, spot, w, lev=1, borrow_apr=0.08)
eq = (1 + r).cumprod()
# executed trades: replicate carry2_lab.simulate's re-size band (|change| > 25% of the held
# target, or an entry / exit); unexecuted target wiggles are not trades
W = w.to_numpy(copy=True)
alive = (close.notna() & spot.notna()).to_numpy()
W[~alive] = 0.0
held = np.zeros(W.shape[1])
D = np.zeros_like(W)
for t in range(len(W)):
    tg = W[t]
    ch = (np.abs(tg - held) > 0.25 * np.maximum(held, 1e-12)) | ((tg == 0) != (held == 0))
    D[t] = np.where(ch, np.abs(tg - held), 0.0)
    held = np.where(ch, tg, held)
dw = pd.DataFrame(D, index=w.index, columns=w.columns)
k = 1 / 1.2
trades = dw.stack()
trades = trades[trades > 1e-9]
ix = trades.index
t_, c_ = ix.get_level_values(0), ix.get_level_values(1)
frac = trades.to_numpy() * k * eq.reindex(t_).to_numpy()  # trade notional per $1 of AUM
a_s = adv_spot.stack(future_stack=True).reindex(ix).to_numpy()
a_p = adv_perp.stack(future_stack=True).reindex(ix).to_numpy()
sg = sig_d.stack(future_stack=True).reindex(ix).to_numpy()
T = pd.DataFrame({"t": t_, "pair": c_, "frac": frac, "adv_spot": a_s, "adv_perp": a_p, "sig": sg}).dropna()
T["year"] = pd.DatetimeIndex(T.t).year
T["bind"] = np.minimum(T.adv_spot, T.adv_perp)
T["aum_at_1pct"] = 0.01 * T.bind / T.frac
print(f"trades: {len(T)}; median binding ADV ${T.bind.median() / 1e6:.1f}M (spot is the binding leg in "
      f"{(T.adv_spot < T.adv_perp).mean():.0%} of trades)")
print("\n#### AUM (USD) at which a trade is 1% of the binding leg's 24h volume, by year")
q = T.groupby("year")["aum_at_1pct"].quantile([0.1, 0.5]).unstack()
q.columns = ["10th pct trade (tight)", "median trade"]
print((q / 1e6).round(2).astype(str).add("M").to_string())

print("\n#### Square-root impact: net CAGR / Sharpe of the 1x book vs AUM (impact on both legs)")
rows = []
base_cost = C2.COSTS["taker"]
for aum in (1e5, 1e6, 5e6, 2e7, 5e7, 1e8):
    imp_s = 0.5 * T.sig * np.sqrt(aum * T.frac / T.adv_spot)
    imp_p = 0.5 * T.sig * np.sqrt(aum * T.frac / T.adv_perp)
    imp = (imp_s + imp_p) / 2
    # extra cost per $ of AUM = sum over trades of notional fraction * impact, each leg vs its own ADV
    extra = (T.frac * (imp_s + imp_p)).groupby(T.t).sum().reindex(r.index).fillna(0)
    rr = r - extra / eq.shift(1).fillna(1)
    row = {"AUM": f"${aum / 1e6:g}M", "avg impact bp/leg": float((imp * 1e4).mean())}
    for s in ("IS", "VAL", "OOS"):
        m = L.metrics(rr, *L.SPLITS[s])
        row[f"{s}_cagr"], row[f"{s}_sh"] = m["cagr"], m["sharpe"]
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
