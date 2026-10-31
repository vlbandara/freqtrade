# pragma pylint: disable=missing-docstring, invalid-name
"""
EdgeHuntDilutionShort - short freshly listed perps (token-dilution decay), hedged with BTC.

New perps (low float, high FDV, unlocks ahead) have tended to bleed against BTC in their first
weeks. Once a day (00:00 UTC) the strategy targets a short in every perp aged between 1 and
`n_days` days with at least `min_qv_day` USDT of 24h volume, sized by inverse volatility
(short book = `short_gross` of equity, max `cap` per coin). A BTC long of the same total
notional neutralises market direction. Every leg is resized to target daily (position adjustment).

Research: user_data/edge_research (modern_lab.dilution_short, final_eval_modern.log).
Design 2022-23 Sharpe 0.94, holdout 2024-25 Sharpe 1.36, drawdown about -21% on its own.
Use a whitelist that contains every USDT-M perp (new listings must be on it).
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)
BTC = "BTC/USDT:USDT"


class EdgeHuntDilutionShort(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.99
    use_exit_signal = True
    process_only_new_candles = True
    position_adjustment_enable = True
    # freqtrade's backtester drops the first startup_candle_count candles of EVERY pair
    # (converter.trim_dataframe), which would erase a new listing's first weeks - the exact
    # window traded here. Keep it at one day (= skip_days).
    startup_candle_count = 24

    n_days = 30
    skip_days = 1
    short_gross = 0.5
    cap = 0.1
    min_qv_day = 2e6
    rebalance_band = 0.01  # resize a leg when it is off target by more than 1% of equity

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._xs = None
        self._xs_key = None

    def _build(self) -> dict:
        closes, qvs = {}, {}
        for p in self.dp.current_whitelist():
            df = self.dp.get_pair_dataframe(p, self.timeframe)
            if df is None or df.empty:
                continue
            o = df.set_index("date")
            closes[p] = o["close"]
            qvs[p] = o["close"] * o["volume"]
        close = DataFrame(closes).sort_index()
        qv = DataFrame(qvs).reindex(close.index).rolling(24, min_periods=12).sum()
        first = close.apply(lambda s: s.first_valid_index())
        known = first > close.index[0] + pd.Timedelta("7D")  # listing seen inside the loaded data
        age = close.notna().cumsum().where(known.to_numpy()[None, :] & close.notna().to_numpy())
        elig = (age > 24 * self.skip_days) & (age <= 24 * self.n_days) & (qv > self.min_qv_day)
        elig = elig.drop(columns=[BTC], errors="ignore")
        vol = close.pct_change(fill_method=None).rolling(72, min_periods=24).std()
        inv = (1 / vol.reindex(columns=elig.columns)).where(elig)
        w = inv.div(inv.sum(axis=1), axis=0).clip(upper=self.cap / self.short_gross).fillna(0)
        w = w.div(np.maximum(w.sum(axis=1), 1.0), axis=0) * self.short_gross
        # decide once a day at 00:00 UTC, hold the decision for the day
        daily = w[w.index.hour == 0].reindex(w.index).ffill().fillna(0)
        return {"w": daily, "hedge": daily.sum(axis=1)}

    def _xsec(self) -> dict:
        last = None
        if self.dp.runmode.value in ("live", "dry_run"):
            df = self.dp.get_pair_dataframe(BTC, self.timeframe)
            last = df["date"].iloc[-1] if len(df) else None
        key = (tuple(self.dp.current_whitelist()), last)
        if self._xs is None or key != self._xs_key:
            self._xs = self._build()
            self._xs_key = key
        return self._xs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        xs = self._xsec()
        pair = metadata["pair"]
        if pair == BTC:
            dataframe["target"] = dataframe["date"].map(xs["hedge"]).fillna(0.0)
        else:
            col = xs["w"][pair] if pair in xs["w"].columns else pd.Series(dtype=float)
            dataframe["target"] = dataframe["date"].map(col).fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        on = dataframe["target"] > 0
        if metadata["pair"] == BTC:
            dataframe.loc[on, ["enter_long", "enter_tag"]] = (1, "hedge")
        else:
            dataframe.loc[on, ["enter_short", "enter_tag"]] = (1, "dilution")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        off = dataframe["target"] <= 0
        col = "exit_long" if metadata["pair"] == BTC else "exit_short"
        dataframe.loc[off, col] = 1
        return dataframe

    def _target(self, pair: str) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return float(df["target"].iloc[-1]) if len(df) else 0.0

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
        equity = self.wallets.get_total_stake_amount()
        return float(np.clip(equity * self._target(pair), min_stake or 0, max_stake))

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ):
        # Once a day (01:00 UTC, right after the daily decision) bring every leg back to its
        # target notional: squeezed shorts are cut back, decaying ones topped up, and the BTC
        # hedge follows the short book. Without this a losing short grows into a large position.
        if current_time.hour != 1:
            return None
        equity = self.wallets.get_total_stake_amount()
        target = equity * self._target(trade.pair)
        current = trade.amount * current_rate
        diff = target - current
        if abs(diff) < self.rebalance_band * equity or target <= 0:
            return None
        if diff > 0:
            return float(min(diff, max_stake)), "rebalance_up"
        return float(max(diff, -trade.stake_amount * 0.99)), "rebalance_down"

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
