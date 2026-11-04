"""
Synthetic futures market data with realistic intraday structure, for plumbing tests only.

Prices are a correlated random walk simulated at 1-minute resolution (so 5m candles get
real wicks), with session-dependent volatility and volume (quiet Asia, busy London / NY,
thin weekends) and a daily volatility regime. There is NO edge in this data: it is only
meant to make every module fire so the strategy and the backtest pipeline can be checked
end to end without network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PAIRS = {
    "BTC/USDT:USDT": 60000.0,
    "ETH/USDT:USDT": 3000.0,
    "SOL/USDT:USDT": 150.0,
}


def _session_multiplier(idx: pd.DatetimeIndex) -> np.ndarray:
    h = idx.hour + idx.minute / 60
    m = np.full(len(idx), 0.9)
    m[(h >= 0) & (h < 7)] = 0.6  # Asia: quiet
    m[(h >= 7) & (h < 10)] = 1.3  # London open
    m[(h >= 13.5) & (h < 17)] = 1.6  # NY open / overlap
    m[idx.dayofweek >= 5] *= 0.6  # weekend
    return m


def make_minute_paths(
    start: str = "2025-01-01", days: int = 60, seed: int = 7
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    sess = _session_multiplier(idx)
    day_regime = np.repeat(rng.lognormal(0.0, 0.35, days), 1440)
    base_sigma = 0.0006  # per-minute
    common = rng.standard_normal(len(idx))
    out = {}
    for i, (pair, p0) in enumerate(PAIRS.items()):
        idio = rng.standard_normal(len(idx))
        shock = common if i == 0 else 0.8 * common + 0.6 * idio  # unit variance, rho=0.8
        sigma = base_sigma * sess * day_regime * (1.0 + 0.3 * i)
        ret = sigma * shock
        # rare fat-tailed jumps (liquidation-like) with a volume burst
        jumps = rng.random(len(idx)) < 1 / 3000
        jump_ret = jumps * rng.choice([-1, 1], len(idx)) * sigma * 25
        ret = ret + jump_ret - 0.6 * np.concatenate([[0.0], jump_ret[:-1]])  # wick-like snap-back
        close = p0 * np.exp(np.cumsum(ret))
        open_ = np.concatenate([[p0], close[:-1]])
        wick = np.abs(rng.standard_normal(len(idx))) * sigma * close * 0.5
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        vol = 100 * sess * day_regime * (1 + 400 * np.abs(ret)) * rng.lognormal(0, 0.3, len(idx))
        vol = vol * np.where(jumps, 15.0, 1.0)
        out[pair] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
        )
    return out


def resample(df1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = df1m.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna().reset_index().rename(columns={"index": "date"})
    return agg


def make_ohlcv(timeframe: str = "5m", **kwargs) -> dict[str, pd.DataFrame]:
    rule = timeframe.replace("m", "min")
    return {p: resample(d, rule) for p, d in make_minute_paths(**kwargs).items()}


def make_funding(dates: pd.Series, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(dates.min().floor("8h"), dates.max(), freq="8h")
    rate = np.empty(len(idx))
    x = 0.0001
    for k in range(len(idx)):
        x = 0.0001 + 0.85 * (x - 0.0001) + rng.normal(0, 0.00012)
        rate[k] = x
    return pd.DataFrame(
        {"date": idx, "open": rate, "high": rate, "low": rate, "close": rate, "volume": 0.0}
    )


def write_datadir(datadir: Path, days: int = 60, start: str = "2025-01-01") -> list[str]:
    """Write 5m/1h/1d futures candles, 8h funding and 8h mark candles in feather format."""
    from freqtrade.data.history import get_datahandler
    from freqtrade.enums import CandleType

    dh = get_datahandler(datadir, "feather")
    minute = make_minute_paths(start=start, days=days)
    for pair, d1m in minute.items():
        for tf, rule in (("5m", "5min"), ("1h", "1h"), ("4h", "4h"), ("1d", "1D")):
            dh.ohlcv_store(pair, tf, resample(d1m, rule), CandleType.FUTURES)
        mark = resample(d1m, "8h")
        dh.ohlcv_store(pair, "8h", mark, CandleType.MARK)
        dh.ohlcv_store(pair, "8h", make_funding(mark["date"]), CandleType.FUNDING_RATE)
    return list(minute)
