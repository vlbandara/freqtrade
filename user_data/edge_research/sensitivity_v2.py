"""Sensitivity (NOT selection): WF portfolio vs the leverage cap, and the cascade stream by year.
Uses the exact walk-forward picks logged in final_eval_v2.log."""

import carry2_lab as C2
import carry_lab as C
import cascade_lab as CL
import edge_lab as L
import numpy as np
import pandas as pd


p = L.load_panel()
spot = C.load_spot(p["close"])
U50 = L.universe(p, 50)


def daily(r):
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


picks = {
    2022: (-0.02, 6, 5, 0.12),
    2023: (-0.03, 6, 5, 0.12),
    2024: (-0.03, 6, 5, 0.12),
    2025: (-0.02, 12, 5, 0.08),
}
carry = daily(C2.simulate(p, spot, C2.target_weights(p, spot, U50, top_k=10))[0])
casc = {
    k: daily(CL.cascade(p, U50, flush=v[0], hold=v[1], nworst=v[2])[0]) for k, v in picks.items()
}

print("cascade stream (WF pick) by year:")
for y, s in casc.items():
    x = s.loc[str(y)]
    print(
        f"  {y}: return {(1 + x).prod() - 1:7.1%}  Sharpe {x.mean() / x.std() * np.sqrt(365):5.2f}"
    )


def combine(c, s, target, maxlev):
    vc = c.rolling(90, min_periods=30).std().shift(1)
    vs = s.rolling(90, min_periods=30).std().shift(1)
    wc, ws = 1 / vc, 1 / vs
    mix = (wc / (wc + ws)).fillna(0.5) * c + (ws / (wc + ws)).fillna(0.5) * s
    mvol = mix.rolling(30, min_periods=15).std().shift(1) * np.sqrt(365)
    lev = (target / mvol).clip(upper=maxlev).fillna(1.0).to_numpy()
    mv = mix.fillna(0).to_numpy()
    out, eq, peak = np.zeros(len(mv)), 1.0, 1.0
    for i in range(len(mv)):
        out[i] = mv[i] * lev[i] * (0.5 if eq / peak - 1 < -0.04 else 1.0)
        eq *= 1 + out[i]
        peak = max(peak, eq)
    return pd.Series(out, index=mix.index)


print("\nleverage-cap sensitivity of the walk-forward portfolio (2022-01..2025-05):")
for cap in (1, 2, 3, 4, 5):
    wf = pd.concat([combine(carry, casc[y], picks[y][3], cap).loc[str(y)] for y in picks])
    eq = (1 + wf).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (365 / len(wf)) - 1
    print(
        f"  cap {cap}x: CAGR {cagr:6.1%}  Sharpe {wf.mean() / wf.std() * np.sqrt(365):5.2f}  maxDD {dd:6.1%}"
    )
