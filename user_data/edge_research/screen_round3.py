"""Round 3: spot-perp basis, seasonality and other "weird" signals (IS + VAL only)."""

import edge_lab as L
import numpy as np
import pandas as pd


p = L.load_panel()
close = p["close"]
U = L.universe(p, 50)
fk = p["funding_known"]
rows = []


def rec(name, w, every=8, **kw):
    w = L.staggered(w, every, n=min(every, 4)) if every > 1 else w
    r = L.run(name, w, p, **kw)
    df = L.report(r, splits=("IS", "VAL"), show=False)
    row = {"name": name}
    for s in ("IS", "VAL"):
        for m in ("sharpe", "cagr", "maxdd"):
            row[f"{s}_{m}"] = df.loc[s, m]
    row["turn/d"] = df["turn/d"].mean()
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return r


# ---- spot / perp basis (perp premium over spot, smoothed)
spot = {}
for f in sorted((L.DATA_DIR).glob("*_USDT-1h.feather")):
    pair = f.name.replace("-1h.feather", "") + "_USDT"
    if pair in close.columns:
        d = pd.read_feather(f).drop_duplicates("date").set_index("date")["close"]
        spot[pair] = d
spot = pd.DataFrame(spot).reindex(index=close.index, columns=close.columns)
basis = close / spot - 1
for h in (8, 24, 72):
    b = basis.rolling(h, min_periods=h // 2).mean()
    rec(f"M_basis_contra_h{h}", L.long_short_from_rank(L.xs_rank(-b, U), 0.2))
# basis minus funding: premium not yet priced into funding
bz = basis.rolling(24, min_periods=12).mean() - fk.rolling(72, min_periods=8).mean() / 8
rec("M_basis_minus_fund", L.long_short_from_rank(L.xs_rank(-bz, U), 0.2))

# ---- time-series seasonality on BTC and an equal-weight alt index (hour of day)
ret = close.pct_change(fill_method=None)
ew = ret.where(U).mean(axis=1)
is_mask = (ret.index >= "2021-01-01") & (ret.index < "2023-01-01")
hr = ret.index.hour
prof = pd.DataFrame(
    {
        "btc": ret["BTC_USDT_USDT"][is_mask].groupby(hr[is_mask]).mean() * 1e4,
        "ew": ew[is_mask].groupby(hr[is_mask]).mean() * 1e4,
    }
)
print("IS hour-of-day mean return (bps):\n", prof.round(2).T)
# trade: long EW basket during the IS-best 6 hours, flat otherwise (choose hours on IS only)
best = prof["ew"].nlargest(6).index.tolist()
worst = prof["ew"].nsmallest(6).index.tolist()
print("best hours", sorted(best), "worst hours", sorted(worst))
nxt = pd.Series(hr, index=close.index).shift(-1)
ewU = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
rec("N_hour_long_best6", ewU.mul(nxt.isin(best), axis=0), 1)
rec("N_hour_ls_best_worst", ewU.mul(nxt.isin(best).astype(float) - nxt.isin(worst), axis=0), 1)

# ---- day-of-week on EW basket
dw = close.index.dayofweek
dprof = ew[is_mask].groupby(dw[is_mask]).mean() * 1e4
print("IS day-of-week mean hourly return (bps):", dprof.round(2).to_dict())

# ---- funding *dispersion* regime switch: when funding dispersion is high, the
# contrarian spread should be richer -> trade only then
f9 = fk.rolling(72, min_periods=8).mean()
disp = f9.where(U).std(axis=1)
dpct = disp.rolling(24 * 90, min_periods=24 * 30).rank(pct=True)
base = L.long_short_from_rank(L.xs_rank(-f9, U), 0.2)
rec("O_fc_high_dispersion", base.mul(dpct > 0.5, axis=0))
rec("O_fc_low_dispersion", base.mul(dpct <= 0.5, axis=0))

# ---- funding volatility: coins with unstable funding (leverage churn) short
fvol = fk.rolling(24 * 14, min_periods=24 * 5).std()
rec("P_fundvol_short", L.long_short_from_rank(L.xs_rank(-fvol, U), 0.2))

# ---- "skew of the book": long coins with many long lower wicks (absorbed selling)
lowwick = (np.minimum(p["open"], close) - p["low"]) / close
upwick = (p["high"] - np.maximum(p["open"], close)) / close
wick_bias = (lowwick - upwick).rolling(72, min_periods=36).mean()
rec("Q_wick_bias", L.long_short_from_rank(L.xs_rank(wick_bias, U), 0.2), 24)
rec("Q_wick_bias_rev", L.long_short_from_rank(L.xs_rank(-wick_bias, U), 0.2), 24)

res = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250)
print(res.round(3))
res.to_csv("round3.csv")
