"""Round 6: cross-sectional ML alpha (gradient-boosted trees) with purged walk-forward.

Protocol (fixed before the first run):
* Universe: point-in-time top-100 perps by 30d median quote volume, >= 30 days of history.
* Target: 3-day forward return incl. funding, minus the cross-sectional mean, ranked per day.
* Model: LightGBM regressor, 3-seed average, hyper-parameters below, never tuned on 2024-25.
  Linear baseline: ridge on the same ranked features.
* Walk-forward: retrain on the first day of every month from 2022-01; training rows are those
  whose target window ends >= 1 day before the month starts (purge + embargo). Training data
  starts 2020-07. Every prediction is out-of-sample.
* Books: daily, long top / short bottom quantile of predictions, equal-weight, dollar-neutral,
  gross 1, averaged over the last 3 daily books (staggered). Variants are chosen on 2022-23.
* Costs: 0.05% taker + 0.02% slippage per side; actual funding paid/received hourly.
"""

import os
import sys

import edge_lab as L
import lightgbm as lgb
import numpy as np
import pandas as pd
import r6_common as C6
import r6_ml_features as FE
from scipy import stats as st


pd.set_option("display.width", 250)
H = 3
SEEDS = (1, 2, 3)
PARAMS = dict(
    n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=500,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1,
    n_jobs=int(os.environ.get("R6_JOBS", 2)),
)
OPEN_HOLDOUT = "--holdout" in sys.argv
END = "2025-06-01" if OPEN_HOLDOUT else "2024-01-01"

p = L.load_panel()
F = FE.build(p, 100, H)
X, y = F["X"], F["y_rank"]
days = X.index.get_level_values(0)
feat_cols = list(X.columns)
_r = F["close"].pct_change(fill_method=None)
BETA = _r.rolling(30, min_periods=20).cov(_r[FE.BTC]).div(_r[FE.BTC].rolling(30, min_periods=20).var(), axis=0)
BETA = BETA.clip(0, 3)
print(f"rows {len(X):,}  features {len(feat_cols)}  days {days.nunique()}", flush=True)


def walk_forward(kind="gbm"):
    preds = []
    imps = []
    for m in pd.date_range("2022-01-01", END, freq="MS", tz="UTC")[:-1]:
        cutoff = m - pd.Timedelta(days=H + 2)
        tr = (days >= "2020-07-01") & (days < cutoff) & y.notna().to_numpy()
        te = (days >= m) & (days < m + pd.offsets.MonthBegin(1))
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        if kind == "gbm":
            pr = np.zeros(len(Xte))
            for s in SEEDS:
                mdl = lgb.LGBMRegressor(random_state=s, **PARAMS)
                mdl.fit(Xtr, ytr)
                pr += mdl.predict(Xte) / len(SEEDS)
                imps.append(pd.Series(mdl.booster_.feature_importance("gain"), index=feat_cols))
        else:  # ridge on ranks, NaN -> 0.5 (median rank)
            A = Xtr.fillna(0.5).to_numpy()
            A = np.c_[A, np.ones(len(A))]
            lam = 10.0 * len(A)
            beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr.to_numpy())
            pr = np.c_[Xte.fillna(0.5).to_numpy(), np.ones(len(Xte))] @ beta
        preds.append(pd.Series(pr, index=Xte.index))
        print(kind, m.date(), f"train={tr.sum():,}", flush=True)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False) if imps else None
    return pd.concat(preds), imp


def daily_ic(pred):
    df = pd.DataFrame({"p": pred, "y": F["y"].reindex(pred.index)}).dropna()
    return df.groupby(level=0).apply(lambda g: st.spearmanr(g.p, g.y).statistic if len(g) > 10 else np.nan)


def book(pred, q=0.2, mode="ls"):
    P = pred.unstack()
    r = P.rank(axis=1, pct=True)
    lo = (r <= q).astype(float)
    hi = (r > 1 - q).astype(float)
    w = pd.DataFrame(0.0, index=P.index, columns=P.columns)
    if mode in ("ls", "long"):
        w += hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * (0.5 if mode == "ls" else 1.0)
    if mode in ("ls", "short_hedged"):
        w -= lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
    if mode == "short_hedged":
        w[FE.BTC] = w.get(FE.BTC, 0) + 0.5
    if mode == "ls_bn":  # dollar-neutral L/S plus a BTC hedge that cancels the book's net beta
        w += hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
        w -= lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
        net_beta = (w * BETA.reindex_like(w).fillna(1.0)).sum(axis=1)
        w[FE.BTC] = w[FE.BTC] - net_beta
    w = w.rolling(H, min_periods=1).mean()  # staggered: average of the last H daily books
    # daily row D -> hourly row with open time D 23:00 (decided at D+1 00:00), held until next
    hourly = pd.DataFrame(np.nan, index=p["close"].index, columns=p["close"].columns)
    stamp = w.index + pd.Timedelta(hours=23)
    hourly.loc[stamp[stamp.isin(hourly.index)]] = w.reindex(columns=hourly.columns).fillna(0).to_numpy()[
        stamp.isin(hourly.index)
    ]
    return hourly.ffill().fillna(0.0)


def summarize(tag, pred):
    ic = daily_ic(pred)
    rows = []
    for mode, q in (("ls", 0.2), ("ls_bn", 0.2), ("ls", 0.1), ("short_hedged", 0.2), ("long", 0.2)):
        res = L.run(tag, book(pred, q, mode), p)
        d = C6.daily(res.pnl).loc["2022":END]
        for split, (a, b) in {"DESIGN": ("2022", "2023"), "HOLDOUT": ("2024", "2025")}.items():
            dd = d.loc[a:b]
            if len(dd) < 30:
                continue
            m = C6.stats(dd)
            btc_d = C6.daily(p["close"][FE.BTC].pct_change(fill_method=None).fillna(0)).reindex(dd.index)
            rows.append({
                "model": tag, "book": f"{mode}_q{q}", "split": split, **m,
                "corr_btc": dd.corr(btc_d), "turn/d": res.turnover.loc[a:b].mean() * 24,
                **{f"y{y}": v for y, v in ((1 + dd).groupby(dd.index.year).prod() - 1).items()},
                "IC_mean": ic.loc[a:b].mean(), "IC_t": ic.loc[a:b].mean() / ic.loc[a:b].std() * np.sqrt(ic.loc[a:b].count() / H),
            })
    return pd.DataFrame(rows), ic


out = []
tag = "holdout" if OPEN_HOLDOUT else "design"
if "--reuse" in sys.argv:
    _c = pd.read_pickle(L.CACHE / f"r6_ml_preds_{tag}.pkl")
    pg, pr_, imp = _c["gbm"], _c["ridge"], _c["imp"]
else:
    pg, imp = walk_forward("gbm")
    pr_, _ = walk_forward("ridge")
for mname, pred in (("gbm", pg), ("ridge", pr_)):
    tab, ic = summarize(mname, pred)
    out.append(tab)
    yr = ic.groupby(ic.index.year).mean()
    print(f"{mname} IC by year:", yr.round(4).to_dict())
res = pd.concat(out)
print(res.set_index(["model", "book", "split"]).round(3).to_string())
print("\nTop features by gain:\n", (imp / imp.sum()).head(15).round(3).to_string())
pd.to_pickle({"gbm": pg, "ridge": pr_, "imp": imp}, L.CACHE / f"r6_ml_preds_{tag}.pkl")
res.to_csv(f"r6_ml_{tag}.csv", index=False)

if OPEN_HOLDOUT:  # pass-bar checks for the locked book only (r6_ml_LOCK.md)
    wl = book(pg, 0.2, "ls_bn")
    for fx in (1, 2):
        d = C6.daily(L.run("locked", wl, p, fee=fx * L.FEE).pnl).loc["2024":"2025"]
        yr = (1 + d).groupby(d.index.year).prod() - 1
        print(f"LOCKED gbm ls_bn_q0.2 holdout, {fx}x costs:", {k: round(v, 3) for k, v in C6.stats(d).items()},
              "years", yr.round(3).to_dict())
    mo = (1 + d).groupby(d.index.to_period("M")).prod() - 1
    print("holdout months positive (2x costs):", f"{(mo > 0).mean():.0%}")
