"""Round 6: how the surviving streams combine, and ML-book capacity.

Streams (daily, 2022-01 .. 2025-05):
  CARRY  locked carry harvest at 1x (8% borrow APR), carry2_lab
  ML     locked GBM beta-neutral quintile L/S (r6_ml_LOCK.md), walk-forward predictions
  R3     locked round-3 A+B+C portfolio
Mix rule fixed in advance (no tuning): 90-day inverse-volatility weights, as modern_lab.inv_vol_mix.
NOTE: each component's holdout was opened before this mix was formed, so the mix numbers are
descriptive, not an out-of-sample test.
"""

import os
import sys

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import modern_lab as M
import numpy as np
import pandas as pd
import r6_common as C6


pd.set_option("display.width", 250)
sys.argv = [sys.argv[0], "--holdout", "--reuse"]
import r6_ml_alpha as ML  # noqa: E402  (reuses cached holdout predictions; prints its tables)


p = ML.p
spot = C.load_spot(p["close"])
r, _, _ = C2.simulate(p, spot, C2.target_weights(p, spot, L.universe(p, 50), top_k=10), lev=1, borrow_apr=0.08)
wml = ML.book(ML.pg, 0.2, "ls_bn")
S = {
    "CARRY": C6.daily(r).loc["2022":],
    "ML": C6.daily(L.run("ml", wml, p).pnl).loc["2022":],
    "R3": pd.read_pickle(os.environ.get("R6_TRIALS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r6_trials.pkl"))).astype(float)["P_A30s1_B20_C_donchian20_55_tv0.15"],
}
D = pd.DataFrame(S).loc["2022-01-01":"2025-05-29"].fillna(0)
print("\n#### Correlation of daily returns 2022-25")
print(D.corr().round(2).to_string())
mixes = {
    "CARRY+ML": ["CARRY", "ML"],
    "ML+R3": ["ML", "R3"],
    "CARRY+ML+R3": ["CARRY", "ML", "R3"],
}
rows = []
for name, cols in {**{k: [k] for k in D}, **mixes}.items():
    d = D[cols[0]] if len(cols) == 1 else M.inv_vol_mix({c: D[c] for c in cols})[0]
    for split, (a, b) in M.SPLITS_MODERN.items():
        yr = (1 + d.loc[a:b]).groupby(d.loc[a:b].index.year).prod() - 1
        rows.append({"stream": name, "split": split, **C6.stats(d.loc[a:b]),
                     **{f"y{y}": v for y, v in yr.items()}})
print("\n#### Streams and fixed inverse-vol mixes")
print(pd.DataFrame(rows).set_index(["stream", "split"]).round(3).to_string())

# ML capacity: daily traded notional per coin vs that coin's perp 24h volume
dwh = (wml - wml.shift(1).fillna(0)).abs()
dwd = dwh.resample("D").sum().loc["2024":]
adv = p["quote_volume"].resample("D").sum().reindex_like(dwd)
x = (dwd / adv).where(dwd > 1e-6).stack()
print("\n#### ML book capacity (perp only): AUM at which a coin's daily trade = 1% of its 24h volume")
print({f"p{int(q * 100)}": f"${0.01 / x.quantile(q) / 1e6:.1f}M" for q in (0.5, 0.9, 0.99)})

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ml_r3 = M.inv_vol_mix({"ML": D["ML"], "R3": D["R3"]})[0]
fig, ax = plt.subplots(figsize=(11, 5.5))
for name, d, c in (("ML (locked GBM, beta-neutral L/S)", D["ML"], "#2b6cb0"),
                   ("Round-3 portfolio A+B+C", D["R3"], "#999999"),
                   ("Carry harvest 1x", D["CARRY"], "#2f855a"),
                   ("ML + R3, inverse-vol (descriptive)", ml_r3, "#c05621")):
    ax.plot((1 + d).cumprod(), label=name, color=c, lw=1.6)
ax.axvline(pd.Timestamp("2024-01-01", tz="UTC"), color="k", ls="--", lw=1)
ax.text(pd.Timestamp("2024-01-10", tz="UTC"), ax.get_ylim()[1] * 0.97, "holdout", va="top")
ax.set_yscale("log")
ax.set_title("Round 6: surviving streams, 2022-01 .. 2025-05 (net of costs and funding)")
ax.set_ylabel("growth of $1 (log)")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equity_curves_r6.png"), dpi=110)
print("chart saved")

# deflated Sharpe for the ML stream, same machinery and trial variance as r6_overfit.py
from scipy import stats as st  # noqa: E402

EG = 0.5772156649
single = pd.read_pickle(os.environ.get("R6_TRIALS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r6_trials.pkl"))).astype(float)
single = single[[c for c in single.columns if not c.startswith("P_")]]
var_sr = (single.loc["2022":"2023"].mean() / single.loc["2022":"2023"].std()).var()


def psr(d, sr0=0.0):
    sr = d.mean() / d.std()
    den = np.sqrt(1 - st.skew(d) * sr + (st.kurtosis(d, fisher=False) - 1) / 4 * sr**2)
    return st.norm.cdf((sr - sr0) * np.sqrt(len(d) - 1) / den)


def sr0(n):
    return np.sqrt(var_sr) * ((1 - EG) * st.norm.ppf(1 - 1 / n) + EG * st.norm.ppf(1 - 1 / (n * np.e)))


print("\n#### ML stream: PSR / DSR (N = 10 ML configurations; N = 18 adds the 8 effective round-3/4 bets)")
ic = ML.daily_ic(ML.pg)
for nm, d, icx in (("holdout 2024-25", D["ML"].loc["2024":], ic.loc["2024":]),
                   ("walk-forward 2022-25", D["ML"], ic.loc["2022":])):
    print(f"{nm}: Sharpe {d.mean() / d.std() * np.sqrt(365):.2f}  PSR {psr(d):.3f}  DSR(N=10) {psr(d, sr0(10)):.3f}  "
          f"DSR(N=18) {psr(d, sr0(18)):.3f}  | rank IC {icx.mean():.3f}, overlap-adjusted t {icx.mean() / icx.std() * np.sqrt(icx.count() / 3):.1f}")
