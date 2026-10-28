"""Round 1: broad hypothesis screen on IS + VAL only (OOS stays blind)."""

import edge_lab as L
import numpy as np
import pandas as pd


p = L.load_panel()
close = p["close"]
U = L.universe(p, 50)
ret1h = close.pct_change(fill_method=None)
rows = []


def rec(name, w, every=8):
    w = L.staggered(w, every, n=min(every, 4)) if every > 1 else w
    r = L.run(name, w, p)
    df = L.report(r, splits=("IS", "VAL"), show=False)
    if df.empty:
        return
    rows.append(
        {
            "name": name,
            "IS_sh": df.loc["IS", "sharpe"] if "IS" in df.index else np.nan,
            "IS_cagr": df.loc["IS", "cagr"] if "IS" in df.index else np.nan,
            "VAL_sh": df.loc["VAL", "sharpe"] if "VAL" in df.index else np.nan,
            "VAL_cagr": df.loc["VAL", "cagr"] if "VAL" in df.index else np.nan,
            "VAL_dd": df.loc["VAL", "maxdd"] if "VAL" in df.index else np.nan,
            "turn/d": df["turn/d"].mean(),
        }
    )
    print(rows[-1], flush=True)


# A. funding contrarian (long low funding, short high funding)
fk = p["funding_known"]
for k in (1, 3, 9, 21):
    f = fk.rolling(8 * k, min_periods=8).mean()
    rec(f"A_fundcontra_k{k}", L.long_short_from_rank(L.xs_rank(-f, U), 0.2))
    rec(f"A_fundfollow_k{k}", L.long_short_from_rank(L.xs_rank(f, U), 0.2))

# B. price/funding disagreement: price up with low funding => long
for lb in (72, 168, 336):
    mom = close.pct_change(lb, fill_method=None)
    f = fk.rolling(24 * 3, min_periods=8).mean()
    score = L.xs_rank(mom, U) - L.xs_rank(f, U)
    rec(f"B_mom_minus_fund_lb{lb}", L.long_short_from_rank(L.xs_rank(score, U), 0.2))

# E. cross-sectional momentum / reversal at several horizons
for lb in (1, 4, 24, 72, 168, 336, 720):
    mom = close.pct_change(lb, fill_method=None)
    every = max(1, min(lb, 24))
    rec(f"E_mom_lb{lb}", L.long_short_from_rank(L.xs_rank(mom, U), 0.2), every)
    rec(f"E_rev_lb{lb}", L.long_short_from_rank(L.xs_rank(-mom, U), 0.2), every)

# D. MAX / lottery effect: short coins with biggest 1h / 24h spike over last 7d
for win in (1, 24):
    r = close.pct_change(win, fill_method=None)
    mx = r.rolling(168, min_periods=100).max()
    rec(f"D_max_short_win{win}", L.long_short_from_rank(L.xs_rank(-mx, U), 0.2), 24)

# F. low-vol anomaly
for lb in (168, 720):
    vol = ret1h.rolling(lb, min_periods=lb // 2).std()
    rec(f"F_lowvol_lb{lb}", L.long_short_from_rank(L.xs_rank(-vol, U), 0.2), 24)

# G. abnormal volume (vs 30d): continuation or reversal of the move it came with
qv = p["quote_volume"]
vz = np.log(qv.rolling(24).sum() / qv.rolling(720, min_periods=300).mean() / 24)
r24 = close.pct_change(24, fill_method=None)
rec("G_volshock_x_ret", L.long_short_from_rank(L.xs_rank(vz * np.sign(r24), U), 0.2), 24)
rec("G_volshock_x_ret_rev", L.long_short_from_rank(L.xs_rank(-vz * np.sign(r24), U), 0.2), 24)
rec("G_quiet_long", L.long_short_from_rank(L.xs_rank(-vz, U), 0.2), 24)

# I. BTC lead-lag: alt lagging behind BTC's last-hour move (beta-adj residual)
btc = ret1h["BTC_USDT_USDT"]
for lb in (1, 3):
    b = btc.rolling(lb).sum()
    a = ret1h.rolling(lb).sum()
    resid = a.sub(b, axis=0)
    rec(f"I_leadlag_resid_lb{lb}", L.long_short_from_rank(L.xs_rank(-resid, U), 0.2), 1)

res = pd.DataFrame(rows).set_index("name").sort_values("IS_sh", ascending=False)
pd.set_option("display.width", 200)
print(res.round(3))
res.to_csv("round1.csv")
