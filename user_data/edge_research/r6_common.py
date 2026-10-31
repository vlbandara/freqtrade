"""Round 6 shared helpers: data-integrity fixes found in the round-6 audit.

Truncated histories
-------------------
The data mirror does not always start a pair at its real Binance listing:

* 15 perps (ENA, TAO, ZRO, ZK, NOT, W, IO, ...) start at exactly 2024-07-04 00:00 with
  $36-615M of first-day volume, although they listed between April and June 2024.
* 1000SHIB, 1000LUNC, 1000XEC and LUNA2 start at exactly 2023-01-01 00:00.
* ICP, TLM, BNX, RIF, MOVR, FXS, W, CVC: funding history starts months or years before the
  first candle (re-listings or truncated candle files).

Anything that treats "first candle in the file" as a listing date (round-3 stream A, the
dilution short) therefore shorts established coins as if they were fresh launches.
`listing_age_hours_clean` only reports an age for pairs whose listing is genuine:
the first candle and the first funding settlement agree within 2 days, and the pair is not
part of a >=3-pair batch that starts at exactly 00:00 UTC (the signature of a file cut).
"""

from __future__ import annotations

import edge_lab as L
import numpy as np
import pandas as pd


def first_funding(close: pd.DataFrame) -> pd.Series:
    out = {}
    for c in close.columns:
        f = L.FUT_DIR / f"{c}-8h-funding_rate.feather"
        if f.exists():
            out[c] = pd.read_feather(f, columns=["date"])["date"].min().floor("h")
    return pd.Series(out)


def genuine_listing(close: pd.DataFrame) -> pd.Series:
    """Boolean per pair: True when the first candle in the data is the real listing."""
    first = close.apply(lambda s: s.first_valid_index())
    ff = first_funding(close).reindex(first.index)
    agree = (first - ff).abs() <= pd.Timedelta("2D")
    agree = agree | ff.isna()
    midnight = first.dt.hour.eq(0) & first.dt.minute.eq(0)
    batch = first.map(first.value_counts())
    cut = midnight & (batch >= 3)
    started_in_data = first > close.index[0] + pd.Timedelta("7D")
    return agree & ~cut & started_in_data


def listing_age_hours_clean(close: pd.DataFrame) -> pd.DataFrame:
    age = close.notna().cumsum()
    ok = genuine_listing(close)
    return age.where(ok.reindex(close.columns).fillna(False).to_numpy()[None, :] & close.notna().to_numpy())


def daily(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


def stats(d: pd.Series) -> dict:
    d = d.dropna()
    eq = (1 + d).cumprod()
    sd = d.std()
    return {
        "sharpe": d.mean() / sd * np.sqrt(365) if sd > 0 else 0.0,
        "cagr": eq.iloc[-1] ** (365 / len(d)) - 1,
        "maxdd": (eq / eq.cummax() - 1).min(),
        "days": len(d),
    }


def per_pair_pnl(w: pd.DataFrame, p: dict, fee: float = L.FEE) -> pd.DataFrame:
    """Hourly P&L contribution per pair, same accounting as edge_lab.run."""
    close = p["close"]
    w = w.reindex_like(close).fillna(0.0)
    ret = close.pct_change(fill_method=None).fillna(0.0).clip(-0.9, 5)
    held = w.shift(1).fillna(0.0)
    dw = (w - w.shift(1).fillna(0.0)).abs().shift(1).fillna(0.0)
    return held * ret - dw * fee - held * p["funding"]
