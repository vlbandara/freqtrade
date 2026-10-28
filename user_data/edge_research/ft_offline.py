"""
Run freqtrade (backtesting / lookahead-analysis / recursive-analysis) without exchange API access.

The research container cannot reach Binance, but freqtrade needs exchange market metadata
during startup. This wrapper synthesises Binance USDT-M perpetual market entries for every
pair that has local futures data (tick sizes estimated from the candles themselves; leverage
tiers come from freqtrade's bundled binance_leverage_tiers.json) and injects them in place of
the network call. Everything else is stock freqtrade.

usage: python3 ft_offline.py backtesting --config ... --strategy ...
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.exchange import exchange as ft_exchange


FUT_DIR = Path(__file__).resolve().parents[1] / "data" / "binance" / "futures"


def _decimals(x: float) -> float:
    """Estimate a tick size from observed prices (smallest power of 10 that fits)."""
    for d in range(10):
        if abs(round(x, d) - x) < 1e-12:
            return 10.0**-d
    return 1e-9


def build_markets() -> dict:
    markets = {}
    for f in sorted(FUT_DIR.glob("*-1h-futures.feather")):
        base = f.name.split("_USDT_USDT")[0]
        sym = f"{base}/USDT:USDT"
        try:
            px = pd.read_feather(f, columns=["close"])["close"].tail(500)
        except Exception:
            continue
        tick = min(_decimals(float(v)) for v in px.iloc[::25])
        markets[sym] = {
            "id": f"{base}USDT",
            "lowercaseId": f"{base.lower()}usdt",
            "symbol": sym,
            "base": base,
            "quote": "USDT",
            "settle": "USDT",
            "baseId": base,
            "quoteId": "USDT",
            "settleId": "USDT",
            "type": "swap",
            "spot": False,
            "margin": False,
            "swap": True,
            "future": False,
            "option": False,
            "index": None,
            "active": True,
            "contract": True,
            "linear": True,
            "inverse": False,
            "subType": "linear",
            "taker": 0.0005,
            "maker": 0.0002,
            "contractSize": 1,
            "expiry": None,
            "expiryDatetime": None,
            "strike": None,
            "optionType": None,
            "precision": {"amount": 1e-6, "price": tick, "base": 1e-8, "quote": 1e-8},
            "limits": {
                "leverage": {"min": 1, "max": 20},
                "amount": {"min": 1e-6, "max": 1e12},
                "price": {"min": tick, "max": 1e12},
                "cost": {"min": 5, "max": None},
                "market": {"min": 1e-6, "max": 1e12},
            },
            "marginModes": {"cross": True, "isolated": True},
            "created": 0,
            "tierBased": False,
            "percentage": True,
            "feeSide": "get",
            "info": {},
        }
    return markets


_MARKETS = None


def _offline_load_async_markets(self, reload: bool = False) -> None:
    global _MARKETS
    if _MARKETS is None:
        _MARKETS = build_markets()
    self._api_async.set_markets(_MARKETS)
    # precision values above are absolute tick sizes
    self._api_async.precisionMode = 4  # ccxt.TICK_SIZE
    self._api.precisionMode = 4


ft_exchange.Exchange._load_async_markets = _offline_load_async_markets

# Delisted perps (MATIC, EOS, ...) are in the dataset but not in freqtrade's bundled tier file.
# Give them a conservative generic tier (copied from a small-cap) so they can be backtested.
from freqtrade.exchange.binance import Binance  # noqa: E402


_orig_tiers = Binance.load_leverage_tiers


def _tiers_with_defaults(self):
    tiers = _orig_tiers(self)
    if tiers:
        template = tiers.get("ZRX/USDT:USDT") or next(iter(tiers.values()))
        for sym in self._api_async.markets or {}:
            tiers.setdefault(sym, template)
    return tiers


Binance.load_leverage_tiers = _tiers_with_defaults


if __name__ == "__main__":
    from freqtrade.main import main

    np.seterr(all="ignore")
    main(sys.argv[1:])
