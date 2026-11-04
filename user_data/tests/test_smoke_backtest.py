"""
End-to-end smoke test: the strategy set runs through freqtrade's real backtesting engine
(futures, funding, custom stop / ROI / time stop, risk sizing) on synthetic data, with the
exchange mocked so no network access is needed.

This checks the PLUMBING only. Synthetic data has no edge; profitability is irrelevant here.

    pytest user_data/tests/test_smoke_backtest.py -q
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "strategies"))

from synthetic import PAIRS, write_datadir  # noqa: E402

from freqtrade.commands.optimize_commands import setup_optimize_configuration  # noqa: E402
from freqtrade.configuration import setup_utils_configuration  # noqa: E402
from freqtrade.enums import RunMode  # noqa: E402
from freqtrade.optimize.backtesting import Backtesting  # noqa: E402
from tests.conftest import EXMS, get_args, get_markets, patch_exchange  # noqa: E402


STRATEGIES = ["SessionLiquidityStack", "SLS_AsiaSweep", "SLS_PDSweep", "SLS_NYORB", "SLS_Research"]


def _markets() -> dict:
    markets = get_markets()
    template = markets["ETH/USDT:USDT"]
    for pair in PAIRS:
        base = pair.split("/")[0]
        m = copy.deepcopy(template)
        m.update({"symbol": pair, "id": f"{base}USDT", "base": base, "baseId": base})
        # Binance-like USDT-M contract: 1 coin per contract, small minimum size
        m["contractSize"] = 1
        # patch_exchange() uses ccxt DECIMAL_PLACES precision mode: values are digit counts
        m["precision"] = {"price": 2, "amount": 3}
        m["limits"]["amount"] = {"min": 0.001, "max": 100000}
        m["limits"]["cost"] = {"min": 5, "max": None}
        markets[pair] = m
    return markets


@pytest.fixture(scope="module")
def datadir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("sls_data")
    write_datadir(d, days=75, start="2025-01-01")
    return d


def _run(mocker, datadir: Path, tmp_path: Path, strategies: list[str], extra: dict | None = None):
    patch_exchange(mocker, mock_markets=_markets())
    # the mocked ccxt object has no "features"; real Binance USDT-M serves 1500 candles per call
    mocker.patch(f"{EXMS}.ohlcv_candle_limit", return_value=1500)
    cfg = json.loads((ROOT / "user_data" / "config_sls_futures.json").read_text())
    cfg["exchange"]["pair_whitelist"] = list(PAIRS)
    cfg.update(extra or {})
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    args = [
        "backtesting",
        "--config", str(cfg_path),
        "--datadir", str(datadir),
        "--strategy-path", str(HERE.parent / "strategies"),
        "--strategy-list", *strategies,
        "--timerange", "20250120-20250316",
        "--export", "none",
        "--cache", "none",
        "--userdir", str(tmp_path),
    ]  # fmt: skip
    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    bt = Backtesting(config)
    bt.start()
    return bt.results["strategy"]


def test_smoke_backtest_all_variants(mocker, datadir, tmp_path):
    res = _run(mocker, datadir, tmp_path, STRATEGIES)
    assert set(res) == set(STRATEGIES)

    research = res["SLS_Research"]
    trades = research["trades"]
    assert len(trades) > 20
    assert {t["pair"] for t in trades} == set(PAIRS)
    tags = {t["enter_tag"] for t in trades}
    assert {"late_mom", "wknd_vwap", "fund_window"} <= tags
    exits = {t["exit_reason"] for t in trades}
    assert "time_stop" in exits
    assert exits & {"stop_loss", "trailing_stop_loss", "roi"}
    assert any(t["is_short"] for t in trades) and any(not t["is_short"] for t in trades)
    for t in trades:
        assert t["leverage"] <= 3.0 + 1e-9

    # single-module variants only trade their own module
    for strat, tag in (
        ("SLS_AsiaSweep", "asia_sweep"),
        ("SLS_PDSweep", "pd_sweep"),
        ("SLS_NYORB", "ny_orb"),
    ):
        assert {t["enter_tag"] for t in res[strat]["trades"]} <= {tag}


def test_risk_sizing_caps_loss_per_trade(mocker, datadir, tmp_path):
    res = _run(mocker, datadir, tmp_path, ["SLS_Research"], {"sls_risk_per_trade": 0.005})
    trades = res["SLS_Research"]["trades"]
    stops = [t for t in trades if t["exit_reason"] in ("stop_loss", "trailing_stop_loss")]
    assert stops
    wallet = 10000
    # A stopped-out trade should lose ~0.5% of equity (+ fees/slippage); allow generous slack
    worst = min(t["profit_abs"] for t in stops)
    assert worst > -wallet * 0.005 * 2.5


def test_freqtrade_lookahead_analysis(mocker, datadir, tmp_path):
    """freqtrade's own bias detector: re-runs each trade's backtest on truncated data."""

    from freqtrade.optimize.analysis.lookahead_helpers import LookaheadAnalysisSubFunctions
    from freqtrade.resolvers import StrategyResolver

    patch_exchange(mocker, mock_markets=_markets())
    mocker.patch(f"{EXMS}.ohlcv_candle_limit", return_value=1500)
    cfg = json.loads((ROOT / "user_data" / "config_sls_futures.json").read_text())
    cfg["exchange"]["pair_whitelist"] = list(PAIRS)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    args = [
        "lookahead-analysis",
        "--config", str(cfg_path),
        "--datadir", str(datadir),
        "--strategy-path", str(HERE.parent / "strategies"),
        "--strategy", "SLS_Research",
        "--timerange", "20250120-20250316",
        "--userdir", str(tmp_path),
        "--minimum-trade-amount", "10",
        "--targeted-trade-amount", "20",
    ]  # fmt: skip
    config = setup_utils_configuration(get_args(args), RunMode.UTIL_NO_EXCHANGE)
    config = LookaheadAnalysisSubFunctions.calculate_config_overrides(config)
    objs = StrategyResolver.search_all_objects(config, enum_failed=False)
    obj = next(o for o in objs if o["name"] == "SLS_Research")
    inst = LookaheadAnalysisSubFunctions.initialize_single_lookahead_analysis(config, obj)
    ca = inst.current_analysis
    assert ca.total_signals >= 10
    assert not ca.has_bias, (
        f"biased entries={ca.false_entry_signals} exits={ca.false_exit_signals} "
        f"indicators={ca.false_indicators}"
    )
