"""Round 2 final evaluation: walk-forward carry v2 + cascade-reversion portfolio.

Walk-forward, years Y = 2022..2025 (2025 = Jan..May):
  1. per stream, choose the config with the best Sharpe on all data before Y (pre-declared grids)
  2. combine streams with inverse-vol weights from trailing 90 daily returns
  3. scale the book to a vol target chosen on pre-Y data (largest target whose pre-Y max DD
     <= 5%), using trailing 30-day realised vol, gross multiplier capped at 3
  4. drawdown brake: half size while the running drawdown is worse than 4%
Every input to a decision on day d uses data up to day d-1 only.
"""

import itertools

import carry2_lab as C2
import carry_lab as C
import cascade_lab as CL
import edge_lab as L
import numpy as np
import pandas as pd
from scipy.stats import norm


pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
rng = np.random.default_rng(5)
p = L.load_panel()
spot = C.load_spot(p["close"])
YEARS = [2022, 2023, 2024, 2025]
TARGETS = [0.03, 0.05, 0.08, 0.12]
MAXLEV = 3.0


def daily(r):
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


# ------------------------------------------------------------------ stream grids
print("building stream grids ...", flush=True)
U = {n: L.universe(p, n) for n in (50, 100)}
carry_grid, carry_grid_2x = {}, {}
for un, k, thr in itertools.product((50, 100), (10, 20), (1e-4, 2e-4)):
    w = C2.target_weights(p, spot, U[un], top_k=k, thr_in=thr, thr_out=thr / 2)
    key = f"carry_u{un}_k{k}_thr{thr}"
    carry_grid[key] = daily(C2.simulate(p, spot, w, costs="taker")[0])
    carry_grid_2x[key] = daily(C2.simulate(p, spot, w, costs="taker2x")[0])
    print(" ", key, flush=True)
casc_grid, casc_grid_2x = {}, {}
for fl, hold, n in itertools.product((-0.02, -0.03, -0.04, -0.05), (3, 6, 12), (5, 10)):
    key = f"casc_f{fl}_h{hold}_n{n}"
    casc_grid[key] = daily(CL.cascade(p, U[50], flush=fl, hold=hold, nworst=n)[0])
    casc_grid_2x[key] = daily(CL.cascade(p, U[50], fee=2 * L.FEE, flush=fl, hold=hold, nworst=n)[0])
N_TRIALS = len(carry_grid) + len(casc_grid) + len(TARGETS)
CG, CG2 = pd.DataFrame(carry_grid), pd.DataFrame(carry_grid_2x)
SG, SG2 = pd.DataFrame(casc_grid), pd.DataFrame(casc_grid_2x)


def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else -9


def maxdd(x):
    eq = (1 + x.fillna(0)).cumprod()
    return (eq / eq.cummax() - 1).min()


def combine(c, s, target):
    """inverse-vol mix + vol target + DD brake, all decided on lagged data."""
    vc = c.rolling(90, min_periods=30).std().shift(1)
    vs = s.rolling(90, min_periods=30).std().shift(1)
    wc, ws = 1 / vc, 1 / vs
    tot = wc + ws
    mix = (wc / tot).fillna(0.5) * c + (ws / tot).fillna(0.5) * s
    # vol target on the mixed stream (trailing 30d realised vol of the *unlevered* mix)
    mvol = mix.rolling(30, min_periods=15).std().shift(1) * np.sqrt(365)
    lev = (target / mvol).clip(upper=MAXLEV).fillna(1.0)
    out = np.zeros(len(mix))
    eq, peak = 1.0, 1.0
    mv, lv = mix.fillna(0).to_numpy(), lev.to_numpy()
    for i in range(len(mv)):
        brake = 0.5 if eq / peak - 1 < -0.04 else 1.0
        out[i] = mv[i] * lv[i] * brake
        eq *= 1 + out[i]
        peak = max(peak, eq)
    return pd.Series(out, index=mix.index), lev


# ------------------------------------------------------------------ walk-forward
wf, wf2x, log = [], [], []
for Y in YEARS:
    tr_end = pd.Timestamp(f"{Y}-01-01", tz="UTC")
    tr = slice(None, tr_end - pd.Timedelta("1D"))
    te = slice(tr_end, pd.Timestamp(f"{Y}-12-31", tz="UTC"))
    ck = max(CG.columns, key=lambda k: sharpe(CG[k].loc["2021":].loc[tr]))
    sk = max(SG.columns, key=lambda k: sharpe(SG[k].loc["2021":].loc[tr]))
    # vol target: largest target whose train-period max DD <= 5%
    tgt = TARGETS[0]
    for t in TARGETS:
        pr, _ = combine(CG[ck], SG[sk], t)
        if maxdd(pr.loc["2021":].loc[tr]) >= -0.05:
            tgt = t
    pr, lev = combine(CG[ck], SG[sk], tgt)
    pr2, _ = combine(CG2[ck], SG2[sk], tgt)
    wf.append(pr.loc[te])
    wf2x.append(pr2.loc[te])
    log.append(
        dict(year=Y, carry=ck, cascade=sk, vol_target=tgt, avg_lev=round(lev.loc[te].mean(), 2))
    )
WF = pd.concat(wf)
WF2 = pd.concat(wf2x)
print("\n#### Walk-forward selections (made with pre-year data only)")
print(pd.DataFrame(log).to_string(index=False))


def summary(name, x):
    eq = (1 + x).cumprod()
    yrs = len(x) / 365
    m = x.groupby(x.index.to_period("M")).apply(lambda v: (1 + v).prod() - 1)
    print(
        f"{name:<34} CAGR {eq.iloc[-1] ** (1 / yrs) - 1:7.1%}  total {eq.iloc[-1] - 1:7.1%}  "
        f"Sharpe {sharpe(x):5.2f}  Sortino {x.mean() / x[x < 0].std() * np.sqrt(365):5.2f}  "
        f"maxDD {maxdd(x):6.1%}  months+ {(m > 0).mean():4.0%}  worst month {m.min():6.1%}"
    )
    return m


print("\n#### Walk-forward result 2022-01 .. 2025-05 (out-of-sample of the selection procedure)")
mwf = summary("WF portfolio (taker costs)", WF)
summary("WF portfolio at 2x costs", WF2)
ck_last, sk_last = log[-1]["carry"], log[-1]["cascade"]
summary(
    "  carry stream alone (WF picks)", pd.concat([CG[r["carry"]].loc[str(r["year"])] for r in log])
)
summary(
    "  cascade stream alone (WF picks)",
    pd.concat([SG[r["cascade"]].loc[str(r["year"])] for r in log]),
)

print("\n#### Calendar years (WF)")
print(((1 + WF).groupby(WF.index.year).prod() - 1).round(4).to_string())
print("\n#### Monthly returns (WF)")
print(mwf.round(4).to_string())

print("\n#### Stream correlation (daily, 2021+)")
print(f"carry vs cascade: {CG[ck_last].loc['2021':].corr(SG[sk_last].loc['2021':]):.3f}")
btc = p["close"]["BTC_USDT_USDT"].resample("1D").last().pct_change()
print(f"WF portfolio vs BTC: {WF.corr(btc.reindex(WF.index)):.3f}")

print("\n#### Bootstrap max drawdown of the WF portfolio (10-day blocks, 5000 paths)")
v = WF.to_numpy()
dds = []
for _ in range(5000):
    idx = []
    while len(idx) < len(v):
        s0 = rng.integers(0, len(v))
        idx.extend(range(s0, min(s0 + 10, len(v))))
    dds.append(maxdd(pd.Series(v[idx[: len(v)]])))
dds = np.array(dds)
print(
    f"median {np.median(dds):.1%}, 95th pct {np.percentile(dds, 5):.1%}, 99th pct {np.percentile(dds, 1):.1%}"
)

print("\n#### Deflated Sharpe ratio (Bailey & Lopez de Prado), daily")
sr = WF.mean() / WF.std()
T = len(WF)
sk_, ku = WF.skew(), WF.kurt() + 3
trials = [sharpe(CG[c].loc["2022":]) / np.sqrt(365) for c in CG] + [
    sharpe(SG[c].loc["2022":]) / np.sqrt(365) for c in SG
]
v_sr = np.var(trials)
emax = np.sqrt(v_sr) * (
    (1 - 0.5772) * norm.ppf(1 - 1 / N_TRIALS) + 0.5772 * norm.ppf(1 - 1 / (N_TRIALS * np.e))
)
dsr = norm.cdf((sr - emax) * np.sqrt(T - 1) / np.sqrt(1 - sk_ * sr + (ku - 1) / 4 * sr**2))
print(
    f"trials {N_TRIALS}, annual SR {sr * np.sqrt(365):.2f}, SR* hurdle {emax * np.sqrt(365):.2f}, DSR = {dsr:.4f}"
)

# full-history curve of the configuration selected for the latest year (for the chart)
full, _ = combine(CG[ck_last], SG[sk_last], log[-1]["vol_target"])
pd.DataFrame({"wf": WF, "wf2x": WF2}).to_parquet("wf_v2.parquet")
pd.DataFrame(
    {"portfolio_latest_cfg": full, "carry_v2": CG[ck_last], "cascade": SG[sk_last]}
).to_parquet("v2_streams.parquet")
