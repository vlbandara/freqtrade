# pragma pylint: disable=missing-docstring, invalid-name
"""
SessionLiquidityStack: directional, intraday, session- and liquidity-driven futures strategy.

Core modules (see ``SESSION_LIQUIDITY.md`` for each hypothesis):
  * ``asia_sweep``: London / NY-AM sweep of the Asia range that closes back inside.
  * ``pd_sweep``: prior-day high/low sweep + reclaim, with abnormal volume and crowded funding.
  * ``ny_orb``: NY opening-range breakout, aligned with the day's direction, in a trending regime.

Research modules (off by default, enable via ``sls_modules`` once the event study passes):
  * ``late_mom``, ``wknd_vwap``, ``fund_window``.

Risk:
  * Every entry carries a structural stop, a target and a hard deadline computed on the
    signal candle (``sessionkit.signals``). They are frozen into the trade at entry.
  * Position size = ``risk_per_trade`` x equity / stop distance (fixed-fractional risk).
  * Breakeven stop once the trade has moved ``breakeven_at_r`` R in favour.
  * At most ``max_same_side`` concurrent trades in the same direction (crypto is one factor).

Config overrides (optional, handy for robustness sweeps without code changes).
``sls_modules`` only applies to the base class; the SLS_* subclasses pin their modules::

    "sls_modules": ["asia_sweep", "pd_sweep", "ny_orb"],
    "sls_params": {"asia_rvol_min": 1.8, "pd_rr": 1.2},
    "sls_risk_per_trade": 0.005
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sessionkit.features import event_blackout
from sessionkit.signals import ALL_MODULES, CORE_MODULES, Params, build_features, compute_signals

from freqtrade.enums import CandleType
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    stoploss_from_absolute,
    timeframe_to_minutes,
    timeframe_to_prev_date,
)


logger = logging.getLogger(__name__)


class SessionLiquidityStack(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    informative_timeframe = "1h"
    trend_timeframe = "4h"
    funding_timeframe = "8h"
    can_short = True

    # Exits are driven by the per-trade plan (custom stop / custom ROI / time stop).
    minimal_roi = {"0": 100.0}
    use_custom_roi = True
    # Hard backstop only (leveraged ratio: -0.25 at 3x = 8.3% price move). Every trade has
    # a tighter structural / trailing stop from its plan; this just bounds the worst case.
    stoploss = -0.25
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True  # required for custom_exit (time stop); exit columns stay 0

    process_only_new_candles = True
    # 10 days of 5m candles for time-of-day relative volume + indicator warm-up.
    startup_candle_count = 3000

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }

    # --- risk -----------------------------------------------------------------
    # None -> use config "sls_modules" (default: the three core modules).
    # Subclasses that pin ``modules`` always win over the config.
    modules: tuple[str, ...] | None = None
    risk_per_trade = 0.005
    max_leverage = 3.0
    max_same_side = 3
    breakeven_at_r = 1.0
    events_file = Path(__file__).parent / "sessionkit" / "events.json"

    plot_config = {
        "main_plot": {
            "asia_high": {"color": "#d9a400"},
            "asia_low": {"color": "#d9a400"},
            "pdh": {"color": "#7a3db8"},
            "pdl": {"color": "#7a3db8"},
            "or_high": {"color": "#2a78d6"},
            "or_low": {"color": "#2a78d6"},
            "vwap_day": {"color": "#888888"},
        },
        "subplots": {
            "Rel. volume (time-of-day)": {"rvol": {"type": "bar"}},
            "Regime": {"er_pct_1h": {}, "rv_pct_1h": {}},
            "Funding pct": {"funding_pct": {}},
        },
    }

    # ------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------
    def bot_start(self, **kwargs) -> None:
        self._events: list[str] = []
        if self.events_file.is_file():
            try:
                self._events = list(json.loads(self.events_file.read_text()).get("events", []))
            except (ValueError, OSError) as e:
                logger.warning(f"Could not read {self.events_file}: {e}")
        self._p = self._build_params()
        logger.info(
            f"SessionLiquidityStack modules={[m for m, on in self._p.enabled.items() if on]} "
            f"risk_per_trade={self._risk()} events={len(self._events)}"
        )

    def _build_params(self) -> Params:
        p = Params()
        for k, v in (self.config.get("sls_params") or {}).items():
            if not hasattr(p, k):
                raise ValueError(f"Unknown sls_params key: {k}")
            setattr(p, k, type(getattr(p, k))(v))
        modules = self.modules or self.config.get("sls_modules") or CORE_MODULES
        unknown = set(modules) - set(ALL_MODULES)
        if unknown:
            raise ValueError(f"Unknown modules: {unknown}")
        p.enabled = {m: (m in modules) for m in ALL_MODULES}
        return p

    @property
    def sls_params(self) -> Params:
        if not hasattr(self, "_p"):
            self._p = self._build_params()
        return self._p

    def _risk(self) -> float:
        return float(self.config.get("sls_risk_per_trade", self.risk_per_trade))

    def _is_futures(self) -> bool:
        return self.config.get("trading_mode", "spot") == "futures"

    def _btc_pair(self) -> str:
        stake = self.config["stake_currency"]
        return f"BTC/{stake}:{stake}" if self._is_futures() else f"BTC/{stake}"

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        inf = [(p, self.informative_timeframe) for p in pairs]
        inf += [(p, self.trend_timeframe) for p in pairs]
        if self._is_futures():
            inf += [(p, self.funding_timeframe, CandleType.FUNDING_RATE) for p in pairs]
        inf.append((self._btc_pair(), self.timeframe))
        return inf

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 6},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 288,
                "trade_limit": 4,
                "stop_duration_candles": 72,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 2016,
                "trade_limit": 10,
                "stop_duration_candles": 288,
                "max_allowed_drawdown": 0.08,
            },
        ]

    # ------------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        tf_min = timeframe_to_minutes(self.timeframe)
        df1h = self.dp.get_pair_dataframe(pair, self.informative_timeframe)
        df4h = self.dp.get_pair_dataframe(pair, self.trend_timeframe)
        funding = None
        if self._is_futures():
            funding = self.dp.get_pair_dataframe(
                pair, self.funding_timeframe, candle_type=CandleType.FUNDING_RATE
            )
        btc = None
        if pair != self._btc_pair():
            btc = self.dp.get_pair_dataframe(self._btc_pair(), self.timeframe)
        feats = build_features(dataframe, tf_min, df1h, funding, btc, self.sls_params, df4h=df4h)
        sig = compute_signals(feats, self.sls_params, is_btc=pair == self._btc_pair())
        # Signals are computed here and stored in private columns, so populate_entry_trend is
        # a pure mapping. freqtrade re-runs entry logic on its own output (lookahead-analysis,
        # cached backtests), which therefore stays idempotent even though the intermediate
        # features are dropped to keep multi-year 5m backtests within memory.
        sig = sig.rename(columns={"enter_long": "_sig_long", "enter_short": "_sig_short"})
        sig = sig.rename(columns={"enter_tag": "_sig_tag"})
        keep = [c for c in self.KEEP_COLUMNS if c in sig.columns] + [
            c for c in self._plot_columns() if c in sig.columns
        ]
        return sig[list(dict.fromkeys(keep))]

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = dataframe["_sig_long"]
        dataframe["enter_short"] = dataframe["_sig_short"]
        dataframe["enter_tag"] = dataframe["_sig_tag"]
        return dataframe

    KEEP_COLUMNS = (
        "date", "open", "high", "low", "close", "volume",
        "_sig_long", "_sig_short", "_sig_tag",
        "sig_stop", "sig_target", "sig_deadline", "sig_trail",
    )  # fmt: skip

    def _plot_columns(self) -> list[str]:
        cols = list(self.plot_config.get("main_plot", {}))
        for sub in self.plot_config.get("subplots", {}).values():
            cols += list(sub)
        return cols

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ------------------------------------------------------------------------
    # Trade plan (frozen at entry)
    # ------------------------------------------------------------------------
    def _signal_row(self, pair: str, signal_date: datetime) -> pd.Series | None:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        rows = df.loc[df["date"] == signal_date]
        if rows.empty:
            rows = df.loc[df["date"] <= signal_date].tail(1)
        return None if rows.empty else rows.iloc[-1]

    def _plan(self, pair: str, trade: Trade) -> dict:
        plan = trade.get_custom_data("plan")
        if plan:
            return plan
        tf = timedelta(minutes=timeframe_to_minutes(self.timeframe))
        signal_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc) - tf
        row = self._signal_row(pair, signal_date)
        side = -1 if trade.is_short else 1
        p = self.sls_params
        stop = target = deadline = trail = None
        if row is not None:
            stop = row.get("sig_stop")
            target = row.get("sig_target")
            deadline = row.get("sig_deadline")
            trail = row.get("sig_trail")
        # Fallbacks: a missing or wrong-side stop (gap through the level) gets the widest
        # allowed stop; a missing deadline gets 4h.
        if stop is None or not math.isfinite(stop) or side * (trade.open_rate - stop) <= 0:
            stop = trade.open_rate * (1 - side * p.max_stop_pct)
        if target is not None and (
            not math.isfinite(target) or side * (target - trade.open_rate) <= 0
        ):
            target = None
        if deadline is None or not math.isfinite(deadline):
            deadline = (trade.open_date_utc + timedelta(hours=4)).timestamp()
        plan = {
            "stop": float(stop),
            "target": None if target is None else float(target),
            "deadline": float(deadline),
            "risk": abs(trade.open_rate - float(stop)),
            "trail": float(trail) if trail is not None and math.isfinite(trail) else None,
        }
        trade.set_custom_data("plan", plan)
        return plan

    def _extreme_since_entry(self, pair: str, trade: Trade) -> float | None:
        """Highest high (long) / lowest low (short) of completed candles since entry."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        start = df["date"].searchsorted(pd.Timestamp(trade.open_date_utc))
        part = df.iloc[start:]
        if part.empty:
            return None
        return float(part["low"].min() if trade.is_short else part["high"].max())

    # ------------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        plan = self._plan(pair, trade)
        stop = plan["stop"]
        fee_pad = 2 * (trade.fee_open or 0.0005)
        trail = plan.get("trail")
        if trail:
            # Chandelier stop: best price since entry minus the frozen trailing distance.
            # Uses COMPLETED candles only (not trade.max_rate, which in backtesting already
            # includes the high of the candle being evaluated -> optimistic stops).
            best = self._extreme_since_entry(pair, trade)
            if best is not None:
                stop = min(stop, best + trail) if trade.is_short else max(stop, best - trail)
        elif plan["risk"] > 0 and self.breakeven_at_r > 0:
            move = self.breakeven_at_r * plan["risk"]
            if not trade.is_short and (trade.max_rate or 0) >= trade.open_rate + move:
                stop = max(stop, trade.open_rate * (1 + fee_pad))
            elif trade.is_short and (trade.min_rate or math.inf) <= trade.open_rate - move:
                stop = min(stop, trade.open_rate * (1 - fee_pad))
        return stoploss_from_absolute(
            stop, current_rate, is_short=trade.is_short, leverage=trade.leverage
        )

    def custom_roi(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        trade_duration: int,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float | None:
        target = self._plan(pair, trade)["target"]
        if target is None:
            return None
        sign = -1 if trade.is_short else 1
        return sign * (target / trade.open_rate - 1) * (trade.leverage or 1.0)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        if current_time.timestamp() >= self._plan(pair, trade)["deadline"]:
            return "time_stop"
        return None

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
        if df is None or df.empty:
            return proposed_stake
        stop = df.iloc[-1].get("sig_stop")
        if stop is None or not math.isfinite(stop) or current_rate <= 0:
            return proposed_stake
        dist = abs(current_rate - stop) / current_rate
        if dist <= 0:
            return proposed_stake
        equity = self.wallets.get_total_stake_amount() if self.wallets else proposed_stake
        stake = equity * self._risk() / dist / max(leverage, 1.0)
        stake = min(stake, proposed_stake, max_stake)
        if min_stake and stake < min_stake:
            return 0  # cannot size this small: skip rather than over-risk
        return stake

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(self.max_leverage, max_leverage)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        events = getattr(self, "_events", None)
        if events and event_blackout(pd.Series([pd.Timestamp(current_time)]), events).iloc[0]:
            return False
        is_short = side == "short"
        same_side = sum(1 for t in Trade.get_trades_proxy(is_open=True) if t.is_short == is_short)
        return same_side < self.max_same_side


# ----------------------------------------------------------------------------
# Single-module variants for isolated evaluation:
#   freqtrade backtesting --strategy-list SessionLiquidityStack SLS_AsiaSweep SLS_PDSweep SLS_NYORB
# ----------------------------------------------------------------------------
class SLS_AsiaSweep(SessionLiquidityStack):
    modules = ("asia_sweep",)


class SLS_PDSweep(SessionLiquidityStack):
    modules = ("pd_sweep",)


class SLS_NYORB(SessionLiquidityStack):
    modules = ("ny_orb",)


class SLS_LateMom(SessionLiquidityStack):
    modules = ("late_mom",)


class SLS_WkndVwap(SessionLiquidityStack):
    modules = ("wknd_vwap",)


class SLS_FundWindow(SessionLiquidityStack):
    modules = ("fund_window",)


class SLS_PWSweep(SessionLiquidityStack):
    modules = ("pw_sweep",)


class SLS_EqSweep(SessionLiquidityStack):
    modules = ("eq_sweep",)


class SLS_LiqCascade(SessionLiquidityStack):
    modules = ("liq_cascade",)


class SLS_AsiaSqueeze(SessionLiquidityStack):
    modules = ("asia_squeeze",)


class SLS_TodSeason(SessionLiquidityStack):
    modules = ("tod_season",)


class SLS_CmeGap(SessionLiquidityStack):
    modules = ("cme_gap",)


class SLS_Trend4h(SessionLiquidityStack):
    modules = ("trend_4h",)


class SLS_Research(SessionLiquidityStack):
    """Every module, including the research-gated ones. For research, not for trading."""

    modules = ALL_MODULES
