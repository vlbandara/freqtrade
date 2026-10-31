"""Round 6 overfitting audit of everything claimed in rounds 1-4.

1. Probabilistic Sharpe Ratio (PSR): P(true Sharpe > 0) given track length, skew and kurtosis.
2. Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014): PSR against the Sharpe you
   would expect from the *best of N* skill-less trials, N = effective number of trials.
3. CSCV / Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2015) on
   the matrix of modern-regime trials rebuilt by r6_trials.py: split 2022-01..2025-05 into 16
   blocks, take every 8/8 split (12,870), pick the best trial in-sample and record where it
   ranks out-of-sample. PBO = share of splits where the in-sample winner is below the OOS median.
"""

import glob
import itertools
import os

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import numpy as np
import pandas as pd
import r6_common as C6
from scipy import stats as st


pd.set_option("display.width", 250)
EG = 0.5772156649
HERE = os.path.dirname(os.path.abspath(__file__))
TRIALS = os.environ.get("R6_TRIALS", os.path.join(HERE, "r6_trials.pkl"))


def psr(d: pd.Series, sr_star_d: float = 0.0) -> float:
    """d = daily returns, sr_star_d = benchmark Sharpe in *daily* units."""
    d = d.dropna()
    sr = d.mean() / d.std()
    g3, g4 = st.skew(d), st.kurtosis(d, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(st.norm.cdf((sr - sr_star_d) * np.sqrt(len(d) - 1) / den))


def expected_max_sr(var_sr_d: float, n: float) -> float:
    """Expected maximum daily Sharpe of n independent skill-less trials (False Strategy Thm)."""
    if n <= 1:
        return 0.0
    return np.sqrt(var_sr_d) * (
        (1 - EG) * st.norm.ppf(1 - 1 / n) + EG * st.norm.ppf(1 - 1 / (n * np.e))
    )


def n_eff(R: pd.DataFrame) -> float:
    """Effective number of independent trials from the correlation spectrum."""
    c = np.nan_to_num(R.corr().to_numpy())
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    return float(ev.sum() ** 2 / (ev**2).sum())


def cscv(R: pd.DataFrame, S: int = 16):
    X = R.to_numpy()
    T, N = X.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    s1 = np.array([X[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:])])
    s2 = np.array([(X[a:b] ** 2).sum(0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges)

    def sharpe(ix):
        n = cnt[ix].sum()
        m = s1[ix].sum(0) / n
        v = s2[ix].sum(0) / n - m**2
        return m / np.sqrt(np.maximum(v, 1e-18))

    lam, is_best, oos_best, oos_med = [], [], [], []
    for comb in itertools.combinations(range(S), S // 2):
        i = np.array(comb)
        o = np.setdiff1d(np.arange(S), i)
        si, so = sharpe(i), sharpe(o)
        b = int(np.argmax(si))
        w = (st.rankdata(so)[b]) / (N + 1)
        lam.append(np.log(w / (1 - w)))
        is_best.append(si[b])
        oos_best.append(so[b])
        oos_med.append(np.median(so))
    lam = np.array(lam)
    oos_best = np.array(oos_best) * np.sqrt(365)
    is_best = np.array(is_best) * np.sqrt(365)
    slope = np.polyfit(is_best, oos_best, 1)[0]
    return {
        "splits": len(lam),
        "PBO": float((lam <= 0).mean()),
        "median_logit": float(np.median(lam)),
        "IS_best_sharpe_med": float(np.median(is_best)),
        "OOS_of_IS_best_sharpe_med": float(np.median(oos_best)),
        "P(OOS_sharpe_of_IS_best<0)": float((oos_best < 0).mean()),
        "degradation_slope": float(slope),
    }


def count_trials():
    n = 0
    for f in glob.glob(os.path.join(HERE, "*.csv")):
        if os.path.basename(f).startswith(("leverage", "smartlev")):
            continue
        n += len(pd.read_csv(f))
    return n


R = pd.read_pickle(TRIALS)
single = R[[c for c in R.columns if not c.startswith("P_")]]
ports = R[[c for c in R.columns if c.startswith("P_")]]
DES = slice("2022-01-01", "2023-12-31")
HOLD = slice("2024-01-01", None)
LOCK = "P_A30s1_B20_C_donchian20_55_tv0.15"

print("#### 1. Trial universe (modern regime, rounds 3-4, rebuilt)")
sr_des = single.loc[DES].mean() / single.loc[DES].std()
print(f"single-stream trials: {single.shape[1]}, effective N (correlation spectrum) = {n_eff(single.loc[DES]):.1f}")
print(f"portfolio neighbours: {ports.shape[1]}, effective N = {n_eff(ports.loc[DES]):.1f}")
print(f"design-window Sharpe across single trials: median {np.median(sr_des) * np.sqrt(365):.2f}, "
      f"sd {sr_des.std() * np.sqrt(365):.2f}, best {sr_des.max() * np.sqrt(365):.2f} ({sr_des.idxmax()})")
print(f"all screen CSVs in this folder (rounds 1-5): {count_trials()} rows; prior REPORT counts ~170 + ~36 + ~60")

print("\n#### 2. CSCV probability of backtest overfitting (2022-01..2025-05, 16 blocks)")
rows = {
    "single streams": cscv(single),
    "portfolio neighbours": cscv(ports),
    "everything": cscv(R),
}
print(pd.DataFrame(rows).T.round(3).to_string())

print("\n#### 3. PSR / DSR of the claims")
var_sr = sr_des.var()
spot = None
claims = {}
p = L.load_panel()
spot = C.load_spot(p["close"])
w = C2.target_weights(p, spot, L.universe(p, 50), top_k=10)
r, _, _ = C2.simulate(p, spot, w, lev=1, borrow_apr=0.08)
carry = C6.daily(r)
claims["carry 1x, OOS 2024-25"] = carry.loc["2024":]
claims["carry 1x, 2022-25 (modern)"] = carry.loc["2022":]
claims["round-3 portfolio, holdout"] = R[LOCK].loc[HOLD]
claims["round-3 portfolio, 2022-25"] = R[LOCK]
claims["stream A, holdout"] = R["A_n30_skip1"].loc[HOLD]
claims["stream C, holdout"] = R["C_donchian20_55_tv0.15"].loc[HOLD]
claims["stream B, holdout"] = R["B_ma20"].loc[HOLD]
N_all = max(count_trials(), 300)
N_mod = n_eff(single.loc[DES])
out = []
for k, d in claims.items():
    sr_d = d.mean() / d.std()
    out.append({
        "claim": k,
        "days": len(d),
        "sharpe": sr_d * np.sqrt(365),
        "skew": st.skew(d),
        "kurt": st.kurtosis(d, fisher=False),
        "PSR(SR>0)": psr(d),
        f"DSR N_eff={N_mod:.0f}": psr(d, expected_max_sr(var_sr, N_mod)),
        f"DSR N={N_all}": psr(d, expected_max_sr(var_sr, N_all)),
        "SR0_ann(N_all)": expected_max_sr(var_sr, N_all) * np.sqrt(365),
    })
print(pd.DataFrame(out).set_index("claim").round(3).to_string())
print("\nDSR: probability the true Sharpe beats the best of N skill-less trials. >0.95 = significant.")

print("\n#### 4. Selection-rule check: does the design-window ranking carry into the holdout?")
des, hold = (single.loc[DES].mean() / single.loc[DES].std()), (single.loc[HOLD].mean() / single.loc[HOLD].std())
rho = st.spearmanr(des, hold)
print(f"Spearman(design Sharpe, holdout Sharpe) over {len(des)} trials: {rho.statistic:.2f} (p={rho.pvalue:.3f})")
top = des.sort_values(ascending=False).head(10).index
print(pd.DataFrame({"design": des[top] * np.sqrt(365), "holdout": hold[top] * np.sqrt(365)}).round(2).to_string())
