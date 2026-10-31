"""
Vectorized research harness for screening trading hypotheses on Binance USDT-M perps.

Conventions
-----------
* All frames are wide: index = candle open time (UTC, 1h), columns = pair.
* ``weights`` at row t are decided with information available at the *close* of candle t
  and are held over candle t+1.  PnL(t+1) = sum_i w_i(t) * ret_i(t+1).
* Costs: ``fee`` (taker fee + slippage, per side) charged on turnover |w(t) - w(t-1)|.
  Funding: a position held over the candle whose close coincides with a funding
  settlement pays ``w * funding_rate`` (longs pay positive funding, shorts receive it).
* Splits are fixed before any test is run:
  IS 2020-01-01 .. 2022-12-31, VAL 2023, OOS 2024-01-01 .. end of data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(
    os.environ.get("EDGE_DATA_DIR", Path(__file__).resolve().parents[1] / "data" / "binance")
)
FUT_DIR = DATA_DIR / "futures"
CACHE = Path(os.environ.get("EDGE_CACHE", "/tmp/edge_cache"))

SPLITS = {
    "IS": ("2020-01-01", "2022-12-31 23:59"),
    "VAL": ("2023-01-01", "2023-12-31 23:59"),
    "OOS": ("2024-01-01", "2030-01-01"),
}
FEE = 0.0005 + 0.0002  # taker fee + slippage per side
BARS_PER_DAY = 24
# indices / stablecoins that are not tradable "coins"
EXCLUDE = {
    "BTCDOM_USDT_USDT",
    "DEFI_USDT_USDT",
    "USDC_USDT_USDT",
    "FDUSD_USDT_USDT",
    "TUSD_USDT_USDT",
    "BLUEBIRD_USDT_USDT",
    "FOOTBALL_USDT_USDT",
}


# --------------------------------------------------------------------------- data
def load_panel(tf: str = "1h", min_rows: int = 24 * 60) -> dict[str, pd.DataFrame]:
    """Load all futures OHLCV + funding into wide frames (cached as pickle)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"panel_{tf}.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)
    fields = {k: {} for k in ("open", "high", "low", "close", "volume")}
    funding = {}
    for f in sorted(FUT_DIR.glob(f"*-{tf}-futures.feather")):
        pair = f.name.split(f"-{tf}-")[0]
        df = pd.read_feather(f)
        if len(df) < min_rows:
            continue
        df = df.drop_duplicates("date").set_index("date")
        for k in fields:
            fields[k][pair] = df[k]
        fr = FUT_DIR / f"{pair}-8h-funding_rate.feather"
        if fr.exists():
            fd = pd.read_feather(fr)
            fd["date"] = fd["date"].dt.floor("h")
            funding[pair] = fd.drop_duplicates("date").set_index("date")["open"]
    panel = {k: pd.DataFrame(v).sort_index() for k, v in fields.items()}
    idx = panel["close"].index
    # funding settled at time T is paid by positions held over the candle closing at T,
    # i.e. the candle opening at T - 1 bar.
    fund = pd.DataFrame(funding).reindex(columns=panel["close"].columns)
    fund.index = fund.index - pd.Timedelta(tf)
    panel["funding"] = fund.reindex(idx).fillna(0.0)
    # last known funding rate (for signals) - available at its settlement time
    fk = pd.DataFrame(funding).reindex(columns=panel["close"].columns)
    panel["funding_known"] = fk.reindex(idx.union(fk.index)).ffill().reindex(idx)
    panel["quote_volume"] = panel["close"] * panel["volume"]
    pd.to_pickle(panel, cache_file)
    return panel


def universe(panel: dict, top_n: int = 50, lookback_days: int = 30, min_age_days: int = 30):
    """Point-in-time liquidity universe: top_n pairs by trailing median quote volume."""
    qv = panel["quote_volume"].drop(columns=[c for c in panel["quote_volume"] if c in EXCLUDE])
    med = qv.rolling(lookback_days * BARS_PER_DAY, min_periods=lookback_days * 12).median()
    age = panel["close"].notna().cumsum()
    med = med.where(age >= min_age_days * BARS_PER_DAY)
    rank = med.rank(axis=1, ascending=False)
    return ((rank <= top_n) & panel["close"].notna()).reindex(
        columns=panel["close"].columns, fill_value=False
    )


# --------------------------------------------------------------------------- engine
@dataclass
class Result:
    name: str
    pnl: pd.Series  # per-bar net return of the portfolio
    gross: pd.Series
    turnover: pd.Series
    exposure: pd.Series
    n_trades: int


def run(
    name: str,
    weights: pd.DataFrame,
    panel: dict,
    fee: float = FEE,
    funding: bool = True,
) -> Result:
    close = panel["close"]
    w = weights.reindex_like(close).fillna(0.0)
    ret = close.pct_change(fill_method=None).fillna(0.0).clip(-0.9, 5)
    held = w.shift(1).fillna(0.0)
    gross = (held * ret).sum(axis=1)
    dw = (w - w.shift(1).fillna(0.0)).abs()
    cost = dw.sum(axis=1).shift(1).fillna(0.0) * fee
    fcost = (held * panel["funding"]).sum(axis=1) if funding else 0.0
    pnl = gross - cost - fcost
    entries = ((w != 0) & (w.shift(1).fillna(0) == 0)).sum().sum()
    return Result(name, pnl, gross, dw.sum(axis=1), held.abs().sum(axis=1), int(entries))


def metrics(pnl: pd.Series, start=None, end=None, turnover=None, n_trades=None) -> dict:
    p = pnl.loc[start:end]
    if len(p) < BARS_PER_DAY * 5:
        return {}
    daily = (1 + p).groupby(p.index.floor("D")).prod() - 1
    eq = (1 + daily).cumprod()
    years = len(daily) / 365.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = daily.std()
    sharpe = daily.mean() / sd * np.sqrt(365) if sd > 0 else 0.0
    dsd = daily[daily < 0].std()
    sortino = daily.mean() / dsd * np.sqrt(365) if dsd > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    pos, neg = daily[daily > 0].sum(), -daily[daily < 0].sum()
    out = {
        "ret": eq.iloc[-1] - 1,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "maxdd": dd,
        "pf_daily": pos / neg if neg > 0 else np.inf,
        "days": len(daily),
    }
    return out


def report(res: Result, splits=("IS", "VAL", "OOS"), show=True) -> pd.DataFrame:
    rows = {}
    for s in splits:
        a, b = SPLITS[s]
        m = metrics(res.pnl, a, b)
        if m:
            sl = res.turnover.loc[a:b]
            m["turn/d"] = sl.mean() * BARS_PER_DAY
            m["expo"] = res.exposure.loc[a:b].mean()
            rows[s] = m
    df = pd.DataFrame(rows).T
    if show:
        print(f"== {res.name}  (entries={res.n_trades})")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print(df)
    return df


def quarterly(res: Result) -> pd.Series:
    return (1 + res.pnl).groupby(res.pnl.index.to_period("Q")).prod() - 1


# --------------------------------------------------------------------------- helpers
def xs_rank(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0..1) among eligible pairs."""
    return df.where(mask).rank(axis=1, pct=True)


def long_short_from_rank(
    rank: pd.DataFrame, q: float = 0.2, gross: float = 1.0, long=True, short=True
) -> pd.DataFrame:
    """Dollar-neutral equal weight: long top q, short bottom q (rank high = long)."""
    lo = (rank <= q).astype(float)
    hi = (rank >= 1 - q).astype(float)
    w = pd.DataFrame(0.0, index=rank.index, columns=rank.columns)
    legs = int(long) + int(short)
    if long:
        w = w + hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * gross / legs
    if short:
        w = w - lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * gross / legs
    return w


def hysteresis_weights(
    rank: pd.DataFrame, q_in: float = 0.2, q_out: float = 0.35, per_pos: float = 0.05
) -> pd.DataFrame:
    """Freqtrade-like position logic: enter long when rank >= 1-q_in, hold until
    rank < 1-q_out (or leaves the universe); mirror image for shorts. Fixed size per position."""
    nan = np.nan
    lg = pd.DataFrame(nan, index=rank.index, columns=rank.columns)
    lg[rank >= 1 - q_in] = 1.0
    lg[(rank < 1 - q_out) | rank.isna()] = 0.0
    sh = pd.DataFrame(nan, index=rank.index, columns=rank.columns)
    sh[rank <= q_in] = 1.0
    sh[(rank > q_out) | rank.isna()] = 0.0
    return (lg.ffill().fillna(0) - sh.ffill().fillna(0)) * per_pos


def hold_rebalance(w: pd.DataFrame, every: int, offset: int = 0) -> pd.DataFrame:
    """Only change weights every `every` bars (others forward-filled)."""
    keep = (np.arange(len(w)) % every) == offset
    return w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)


def staggered(w_fn, every: int, n: int | None = None) -> pd.DataFrame:
    """Average of `n` copies of a rebalance-every schedule with staggered offsets
    (reduces timing luck)."""
    n = n or every
    offs = np.linspace(0, every - 1, n).astype(int)
    acc = None
    for o in offs:
        wi = hold_rebalance(w_fn, every, o)
        acc = wi if acc is None else acc + wi
    return acc / len(offs)
