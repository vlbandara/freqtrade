"""Funding carry harvest (delta-neutral: long spot + short perp) - shared implementation."""

import edge_lab as L
import numpy as np
import pandas as pd


SPOT_FEE = 0.001 + 0.0002  # spot taker + slippage


def load_spot(close: pd.DataFrame) -> pd.DataFrame:
    spot = {}
    for f in sorted(L.DATA_DIR.glob("*_USDT-1h.feather")):
        pair = f.name.replace("-1h.feather", "") + "_USDT"
        if pair in close.columns:
            spot[pair] = pd.read_feather(f).drop_duplicates("date").set_index("date")["close"]
    spot = pd.DataFrame(spot).reindex(index=close.index, columns=close.columns)
    # guard against ticker mismatches (same name, different token / redenomination)
    basis = (close / spot - 1).abs()
    bad = basis.rolling(24 * 7, min_periods=24).median() > 0.03
    return spot.where(~bad)


def carry(
    p: dict,
    spot: pd.DataFrame,
    U: pd.DataFrame,
    k_days: float = 3,
    thr_in: float = 0.0001,
    thr_out: float = 0.00005,
    top_k: int = 10,
    perp_fee: float = L.FEE,
    spot_fee: float = SPOT_FEE,
    perp_margin: float = 1.0,
):
    """Returns (pnl on capital, weights, components). Capital = spot notional + perp margin."""
    close = p["close"]
    fk = p["funding_known"]
    favg = fk.rolling(int(24 * k_days), min_periods=8).mean()
    sig = favg.where(U & spot.notna())
    rank = sig.rank(axis=1, ascending=False)
    state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    state[(rank <= top_k) & (sig > thr_in)] = 1.0
    state[(sig < thr_out) | sig.isna()] = 0.0
    pos = state.ffill().fillna(0)
    n = pos.sum(axis=1)
    w = pos.div(np.maximum(n, top_k), axis=0)  # fraction of hedged notional, sum <= 1
    held = w.shift(1).fillna(0)
    r_perp = close.pct_change(fill_method=None).fillna(0)
    r_spot = spot.pct_change(fill_method=None).fillna(0)
    dw = (w - w.shift(1).fillna(0)).abs()
    basis_pnl = (held * (r_spot - r_perp)).sum(axis=1)
    fund_pnl = (held * p["funding"]).sum(axis=1)
    fees = dw.sum(axis=1).shift(1).fillna(0) * (perp_fee + spot_fee)
    cap = 1.0 + perp_margin
    pnl = (basis_pnl + fund_pnl - fees) / cap
    comp = pd.DataFrame({"basis": basis_pnl / cap, "funding": fund_pnl / cap, "fees": -fees / cap})
    return pnl, w, comp
