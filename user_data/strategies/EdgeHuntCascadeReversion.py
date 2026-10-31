# pragma pylint: disable=missing-docstring, invalid-name
"""
EdgeHuntCascadeReversion - buy forced liquidations, hedge the market with BTC.

When the equal-weight basket of the top-50 perps (by trailing 30d median quote volume) drops
more than ``flush`` within one 1h candle, the market is liquidating leveraged longs. The
coins that fell the most in that candle are bought (``alloc / nworst`` of equity each) and
BTC is shorted for the same beta exposure. Every leg is closed ``hold`` hours later.

Research: user_data/edge_research (cascade_lab.py, REPORT.md). IS 2020-22 Sharpe 2.8,
validation 2023 Sharpe 2.8; ~half of the edge is earned in the first hour, so the bot must act
at candle close (process_only_new_candles + low throttle).
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


class EdgeHuntCascadeReversion(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.99
    use_exit_signal = True  # required for custom_exit (time exit)
    process_only_new_candles = True
    startup_candle_count = 24 * 62  # 30d listing age + 30d volume median

    # locked on IS + VAL (see cascade_screen.csv)
    flush = -0.03
    hold = 6
    nworst = 5
    alloc = 0.5
    top_n = 50

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._xs: dict | None = None
        self._xs_key = None

    def _build(self) -> dict:
        pairs = self.dp.current_whitelist()
        closes, qvols = {}, {}
        for p in pairs:
            df = self.dp.get_pair_dataframe(p, self.timeframe)
            if df is None or df.empty:
                continue
            o = df.set_index("date")
            closes[p] = o["close"]
            qvols[p] = o["close"] * o["volume"]
        close = DataFrame(closes).sort_index()
        qv = DataFrame(qvols).reindex(close.index)
        med = qv.rolling(24 * 30, min_periods=24 * 15).median()
        med = med.where(close.notna().cumsum() >= 24 * 30)
        universe = (med.rank(axis=1, ascending=False) <= self.top_n) & close.notna()
        ret = close.pct_change(fill_method=None)
        ew = ret.where(universe).mean(axis=1)
        trig = ew < self.flush
        cand = universe.drop(columns=[BTC], errors="ignore")
        worst = ret.where(cand).rank(axis=1, ascending=True) <= self.nworst
        enter = worst & trig.to_numpy()[:, None]
        btc = ret[BTC]
        beta = (
            ret.rolling(24 * 30, min_periods=240)
            .cov(btc)
            .div(btc.rolling(24 * 30, min_periods=240).var(), axis=0)
        )
        hedge = (enter * beta.fillna(1).clip(0, 3) * self.alloc / self.nworst).sum(axis=1)
        return {"enter": enter, "hedge": hedge.where(trig, 0.0)}

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
            dataframe["hedge_frac"] = dataframe["date"].map(xs["hedge"]).fillna(0.0)
            dataframe["cascade_buy"] = False
        else:
            col = xs["enter"][pair] if pair in xs["enter"].columns else pd.Series(dtype=bool)
            dataframe["cascade_buy"] = dataframe["date"].map(col).fillna(False).astype(bool)
            dataframe["hedge_frac"] = 0.0
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["cascade_buy"], ["enter_long", "enter_tag"]] = (1, "cascade")
        dataframe.loc[dataframe["hedge_frac"] > 0, ["enter_short", "enter_tag"]] = (1, "hedge")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        equity = self.wallets.get_total_stake_amount()
        if entry_tag == "hedge":
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            frac = float(df["hedge_frac"].iloc[-1]) if len(df) else 0.0
        else:
            frac = self.alloc / self.nworst
        return float(np.clip(equity * frac, min_stake or 0, max_stake))

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        if (current_time - trade.open_date_utc).total_seconds() >= self.hold * 3600:
            return "time_exit"
        return None

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
