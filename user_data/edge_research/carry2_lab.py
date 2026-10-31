"""Funding carry harvest v2 - units-based simulation (positions drift between trades).

Differences to carry_lab.carry (v1):
* Holdings are coin units, re-sized only when a coin's target weight changes (entry / exit /
  re-weight beyond a tolerance band), so there is no hidden hourly rebalancing.
* Equity compounds; fees are charged on traded notional of both legs.
* Optional funding-proportional sizing, wider universe, leverage and cost scenarios.
"""

import edge_lab as L
import numpy as np
import pandas as pd


SPOT_FEE = 0.001 + 0.0002
COSTS = {
    "taker": (L.FEE, SPOT_FEE),  # perp 0.05%+0.02%, spot 0.10%+0.02%
    "vip_bnb": (0.00045 + 0.0001, 0.00075 + 0.0001),  # BNB-discount taker, tighter slippage
    "maker": (0.0002 + 0.0001, 0.00075 + 0.0001),  # post-only perp, BNB spot
    "taker2x": (2 * L.FEE, 2 * SPOT_FEE),  # stress test
}


def target_weights(
    p: dict,
    spot: pd.DataFrame,
    U: pd.DataFrame,
    k_days: float = 3,
    thr_in: float = 1e-4,
    thr_out: float = 5e-5,
    top_k: int = 10,
    sizing: str = "equal",
    cap: float = 0.15,
) -> pd.DataFrame:
    """Fraction of hedged notional per coin (sum <= 1). Only uses data up to each bar."""
    fk = p["funding_known"]
    favg = fk.rolling(int(24 * k_days), min_periods=8).mean()
    sig = favg.where(U & spot.notna() & p["close"].notna())
    rank = sig.rank(axis=1, ascending=False)
    state = pd.DataFrame(np.nan, index=sig.index, columns=sig.columns)
    state[(rank <= top_k) & (sig > thr_in)] = 1.0
    state[(sig < thr_out) | sig.isna()] = 0.0
    pos = state.ffill().fillna(0)
    if sizing == "equal":
        return pos.div(np.maximum(pos.sum(axis=1), top_k), axis=0)
    # funding-proportional: richer carry gets more capital, capped per coin
    raw = (pos * (sig - thr_out).clip(lower=0)).fillna(0)
    tot = raw.sum(axis=1)
    w = raw.div(tot.replace(0, np.nan), axis=0).fillna(0)
    for _ in range(5):  # iterative capping with redistribution
        w = w.clip(upper=cap)
        w = w.div(np.maximum(w.sum(axis=1), 1.0), axis=0)
    # deploy at most min(1, n*cap)
    return w


def simulate(
    p: dict,
    spot: pd.DataFrame,
    w: pd.DataFrame,
    lev: float = 1.0,
    perp_margin: float = 0.2,
    costs: str = "taker",
    band: float = 0.25,
    start: str | None = None,
    borrow_apr: float = 0.12,
):
    """Units-based backtest. Hedged notional per coin = w * lev * equity / (1 + perp_margin).

    A coin is re-sized only when its target changes by more than `band` (relative) or it
    enters / exits. Cash needed beyond equity (spot purchases + perp margin when lev > 1) is
    borrowed at `borrow_apr` (USDT margin loan). Returns (returns per bar, components DataFrame,
    gross notional/equity)."""
    perp_fee, spot_fee = COSTS[costs]
    close = p["close"]
    if start:
        close = close.loc[start:]
    idx = close.index
    cols = close.columns
    F = close.ffill().to_numpy()
    S = spot.reindex(index=idx, columns=cols).ffill().to_numpy()
    alive = (close.notna() & spot.reindex(index=idx, columns=cols).notna()).to_numpy()
    fund = p["funding"].reindex(index=idx, columns=cols).fillna(0).to_numpy()
    W = w.reindex(index=idx, columns=cols).fillna(0).to_numpy()
    W[~alive] = 0.0
    n, m = F.shape
    units = np.zeros(m)
    tgt_prev = np.zeros(m)
    eq = 1.0
    out = np.zeros((n, 5))  # ret, basis, funding, fees, borrow
    gross = np.zeros(n)
    k = lev / (1 + perp_margin)
    Sp, Fp = np.nan_to_num(S[0]), np.nan_to_num(F[0])
    for t in range(n):
        St, Ft = np.nan_to_num(S[t]), np.nan_to_num(F[t])
        # P&L of units held over bar t (long spot, short perp) + funding on the perp leg
        basis = np.sum(units * ((St - Sp) - (Ft - Fp)))
        fpay = np.sum(units * Fp * fund[t])
        e0 = eq
        cash_used = np.sum(units * Sp) * (1 + perp_margin)
        borrow = max(0.0, cash_used - e0) * borrow_apr / 8760
        eq = eq + basis + fpay - borrow
        # re-size at the close of bar t
        tgt = W[t]
        change = (np.abs(tgt - tgt_prev) > band * np.maximum(tgt_prev, 1e-12)) | (
            (tgt == 0) != (tgt_prev == 0)
        )
        fee = 0.0
        if change.any():
            new_units = np.where(St > 0, tgt * k * eq / np.where(St > 0, St, 1), 0.0)
            dn = np.where(change, np.abs(new_units - units), 0.0)
            fee = np.sum(dn * St) * spot_fee + np.sum(dn * Ft) * perp_fee
            units = np.where(change, new_units, units)
            tgt_prev = np.where(change, tgt, tgt_prev)
            eq -= fee
        out[t] = ((eq - e0) / e0, basis / e0, fpay / e0, -fee / e0, -borrow / e0)
        gross[t] = np.sum(units * St) / eq if eq > 0 else 0
        Sp, Fp = St, Ft
    comp = pd.DataFrame(out, index=idx, columns=["ret", "basis", "funding", "fees", "borrow"])
    return comp["ret"], comp, pd.Series(gross, index=idx)


def smart_leverage_weights(
    p: dict,
    spot: pd.DataFrame,
    U: pd.DataFrame,
    max_lev: float = 3.0,
    borrow_apr: float = 0.08,
    hurdle_spread: float = 0.05,
    k_days: float = 3,
    thr_in: float = 1e-4,
    thr_out: float = 5e-5,
    top_k: int = 10,
) -> pd.DataFrame:
    """Base book = v1 rules at 1x. Each held coin is scaled up to `max_lev` only while its
    3d average funding (annualised) exceeds borrow_apr + hurdle_spread, i.e. borrowed money is
    only used where the carry clearly pays for the loan. Use with simulate(lev=1)."""
    base = target_weights(p, spot, U, k_days, thr_in, thr_out, top_k)
    favg = p["funding_known"].rolling(int(24 * k_days), min_periods=8).mean()
    rich = favg * 3 * 365 > borrow_apr + hurdle_spread
    return base * (1 + (max_lev - 1) * rich.reindex_like(base).fillna(False))
