# pragma pylint: disable=missing-docstring, invalid-name
"""
EdgeHuntMajorsTrend - BTC / ETH long-or-flat breakout trend, volatility-targeted.

Enter long when the close breaks above the highest close of the previous `slow_days` days;
exit when it falls below the lowest close of the previous `fast_days` days. Position size
targets `target_vol` annual volatility per asset (capped at 1x), split across the two coins.

Research: user_data/edge_research (modern_lab.majors_trend, final_eval_modern.log).
Design 2022-23 Sharpe 0.71, holdout 2024-25 Sharpe 1.45 (part of the round-3 portfolio).
"""

from datetime import datetime

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class EdgeHuntMajorsTrend(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100}
    stoploss = -0.99
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 24 * 56

    fast_days = 20
    slow_days = 55
    target_vol = 0.15
    n_assets = 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        c = dataframe["close"]
        dataframe["hi"] = c.rolling(24 * self.slow_days).max().shift(1)
        dataframe["lo"] = c.rolling(24 * self.fast_days).min().shift(1)
        rv = c.pct_change().rolling(24 * 30, min_periods=24 * 10).std() * np.sqrt(24 * 365)
        dataframe["size"] = (self.target_vol / rv).clip(upper=1.0) / self.n_assets
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["close"] > dataframe["hi"], ["enter_long", "enter_tag"]] = (
            1,
            "breakout",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["close"] < dataframe["lo"], ["exit_long", "exit_tag"]] = (
            1,
            "breakdown",
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        size = float(df["size"].iloc[-1]) if len(df) and np.isfinite(df["size"].iloc[-1]) else 0.0
        equity = self.wallets.get_total_stake_amount()
        return float(np.clip(equity * size, min_stake or 0, max_stake))

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
        return 1.0
