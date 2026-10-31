# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""
REJECTED - DO NOT TRADE. Passed in-sample (Sharpe 2.2) and validation (1.8) but failed the blind
out-of-sample test 2024-01..2025-05 (Sharpe -1.5, -47%). Kept only as a record; see REPORT.md.

EdgeHuntFundingContrarian - cross-sectional "crowding" strategy for Binance USDT-M perps.

Idea
----
Perpetual funding is the price longs pay shorts to keep leverage open. A coin whose funding
has been persistently *higher than its peers* is crowded with leveraged longs; one with the
lowest funding is shunned (or crowded with shorts). Crowded trades unwind. So, every hour:

* universe  = top ``top_n`` perps by trailing 30d median quote volume (age >= 30 days)
* signal    = average funding over the last ``fund_days`` days (only settled, known rates)
* long      the ``q`` fraction of the universe with the LOWEST funding
* short     the ``q`` fraction with the HIGHEST funding
* exit      when a coin drifts out of its bucket by more than the hysteresis band

The book is roughly dollar-neutral (equal number of longs and shorts, equal stake), so it
mostly does not care where BTC goes. It also tends to *receive* funding on both legs.

The cross-sectional table is built once from the DataProvider for the whole whitelist and
cached; values at candle t only use data with date <= t (verified with lookahead-analysis).
"""

import logging

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class EdgeHuntFundingContrarian(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True

    # positions are managed by the rank signal only
    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True

    # 30d volume median + 30d listing age + funding window
    startup_candle_count = 24 * 31

    # ---- parameters (fixed before the out-of-sample run; not hyperopted)
    top_n = 50
    fund_days = 3
    q_entry = 0.20
    q_exit = 0.30
    leverage_x = 1.0

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._xs: DataFrame | None = None
        self._xs_key = None

    # ------------------------------------------------------------------ data
    def informative_pairs(self):
        pairs = self.dp.current_whitelist() if self.dp else []
        return [(p, "8h", "funding_rate") for p in pairs]

    def _build_cross_section(self) -> DataFrame:
        pairs = self.dp.current_whitelist()
        closes, qvols, funds = {}, {}, {}
        for p in pairs:
            ohlcv = self.dp.get_pair_dataframe(p, self.timeframe)
            if ohlcv is None or ohlcv.empty:
                continue
            o = ohlcv.set_index("date")
            closes[p] = o["close"]
            qvols[p] = o["close"] * o["volume"]
            fr = self.dp.get_pair_dataframe(p, "8h", "funding_rate")
            if fr is not None and not fr.empty:
                f = fr[["date", "open"]].copy()
                f["date"] = f["date"].dt.floor("h")
                funds[p] = f.drop_duplicates("date").set_index("date")["open"]
        close = DataFrame(closes).sort_index()
        qv = DataFrame(qvols).reindex(close.index)
        idx = close.index

        # point-in-time liquidity universe
        med = qv.rolling(24 * 30, min_periods=24 * 15).median()
        age = close.notna().cumsum()
        med = med.where(age >= 24 * 30)
        universe = (med.rank(axis=1, ascending=False) <= self.top_n) & close.notna()

        # last *settled* funding rate known at each candle open time t (date <= t)
        fk = DataFrame(funds).reindex(columns=close.columns)
        fk = fk.reindex(idx.union(fk.index)).ffill().reindex(idx)
        favg = fk.rolling(24 * self.fund_days, min_periods=8).mean()

        # rank of -funding: 1.0 = lowest funding (long side), 0.0 = highest (short side)
        rank = (-favg).where(universe).rank(axis=1, pct=True)
        return rank

    def _rank_for(self, pair: str) -> pd.Series:
        last = None
        if self.dp.runmode.value in ("live", "dry_run"):
            df = self.dp.get_pair_dataframe(self.dp.current_whitelist()[0], self.timeframe)
            last = df["date"].iloc[-1] if len(df) else None
        key = (tuple(self.dp.current_whitelist()), last)
        if self._xs is None or key != self._xs_key:
            self._xs = self._build_cross_section()
            self._xs_key = key
        if pair in self._xs.columns:
            return self._xs[pair]
        return pd.Series(dtype=float)

    # ------------------------------------------------------------------ signals
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rank = self._rank_for(metadata["pair"])
        dataframe["fc_rank"] = dataframe["date"].map(rank).astype(float)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        r = dataframe["fc_rank"]
        dataframe.loc[r >= 1 - self.q_entry, ["enter_long", "enter_tag"]] = (1, "low_funding")
        dataframe.loc[r <= self.q_entry, ["enter_short", "enter_tag"]] = (1, "crowded_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        r = dataframe["fc_rank"]
        dataframe.loc[(r < 1 - self.q_exit) | r.isna(), "exit_long"] = 1
        dataframe.loc[(r > self.q_exit) | r.isna(), "exit_short"] = 1
        return dataframe

    def leverage(
        self,
        pair,
        current_time,
        current_rate,
        proposed_leverage,
        max_leverage,
        entry_tag,
        side,
        **kwargs,
    ) -> float:
        return min(self.leverage_x, max_leverage)
