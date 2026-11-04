# Session & Liquidity stack

This is a directional, intraday strategy set for Binance USDT-M perpetuals. It trades long and short, on 5m bars, with isolated margin and at most 3x leverage.
It does no arbitrage. Every module trades one market-structure claim that can be checked against data.

> **Status:** the code, the look-ahead tests and the research harness are done and pass on synthetic data.
> **Nothing has been validated on real market data yet.** Run the research protocol below before you put any money behind it.

## 1. Why these edges should exist

Crypto trades 24/7, but its liquidity does not.
- Order-book depth, volume and volatility rise and fall as trading passes between Asia, Europe and the US.
- Some events happen on a fixed clock: the US equity open (09:30 New York), the funding settlements (00/08/16 UTC), the UTC daily close, and the weekend when CME is shut.
- Stop orders and liquidation prices cluster at obvious levels: the Asia-session high/low, the prior-day high/low, and the opening range.

Two behaviours follow from this, and they pull in opposite directions:

| Behaviour | Mechanism | Modules |
|---|---|---|
| **Stop runs at the hand-off reverse** | When a thin session hands over to a liquid one, the first push takes out the stops resting beyond the obvious level. Larger players fill against that forced flow, and once the stops are used up price comes back inside the range. | `asia_sweep`, `pd_sweep` |
| **Real US participation continues** | A break of the US opening range that comes with abnormal volume, in the same direction as the day so far, reflects new information or flow. It tends to keep going. | `ny_orb`, `late_mom` |
| **Thin books overshoot** | Weekend books are thin, so deviations from fair value (VWAP) are often liquidity-driven and revert. | `wknd_vwap` |
| **Crowded positioning unwinds** | When funding is extreme, the crowded side pays to hold its position. Some of them close before the settlement time. | `fund_window`, and as a filter in `pd_sweep` |

## 2. Modules

| Tag | Direction | Entry (on the signal candle's close) | Stop | Target | Time stop |
|---|---|---|---|---|---|
| `asia_sweep` | MR | During London 08:00–11:00 or NY 08:30–11:30, the first sweep of the Asia range (00:00–07:00 UTC) that closes back inside within 3 bars. Conditions: time-of-day relative volume ≥ 1.5, Asia range ≤ 1.3× its 20-day median, 1h trend efficiency not extreme, R:R ≥ 1 to the Asia midpoint. | sweep extreme ± 0.25 ATR | Asia midpoint | 4h or 00:00 UTC |
| `pd_sweep` | MR | The first sweep and reclaim of the prior-day high/low on a weekday. Conditions: relative volume ≥ 2, and funding percentile shows the swept side is crowded. | sweep extreme ± 0.25 ATR | 1.5R | 4h or 00:00 UTC |
| `ny_orb` | Momentum | The first close outside the 09:30–10:00 ET range, between 10:00 and 12:00 ET. Conditions: relative volume ≥ 1.5, same direction as the return since 00:00 UTC, 1h trend efficiency at or above its 50th percentile, range ≤ 1.5× 1h ATR. | opening-range midpoint | 2R | 16:00 ET |
| `late_mom` *(research)* | Momentum | At 15:30 ET, when the day's return is at least 1σ of its expected volatility, trade in that direction. | 1.5× 1h ATR | – | 23:55 UTC |
| `wknd_vwap` *(research)* | MR | On weekends, the first bar more than 2σ away from the weekend-anchored VWAP, fading the move. | 1× 1h ATR | VWAP | 6h |
| `fund_window` *(research)* | Directional | One hour before each funding settlement, fade the side whose funding is in the top or bottom 10% of the last 30 days. | 1× 1h ATR | – | settlement + 15 min |

These filters apply to every module:
- Skip when 1h realized volatility is above its 95th percentile (news shocks).
- For altcoins, block longs while BTC has moved down more than 2σ in the last hour, and block shorts when it has moved up more than 2σ.
- Block entries 30 minutes before to 60 minutes after any time listed in `sessionkit/events.json` (FOMC, CPI and so on). Fill this list in yourself.
- Drop setups whose stop distance falls outside 0.15%–2%.

### Key feature engineering
- **Time-of-day relative volume.** Volume is divided by the median volume of the same 5-minute slot over the previous 10 days. A value of 2 means "twice the normal volume *for this time of day*". Without this, every US open looks like a volume spike.
- **DST-correct sessions.** Session windows are defined in `America/New_York` and `Europe/London` local time using `zoneinfo`, so they shift correctly when daylight saving changes.
- **Look-ahead safety by construction.** A session's levels only appear after that session has closed, and PDH/PDL require a complete previous day. A unit test re-computes every column on truncated data and requires the values to be identical. Planting look-ahead bugs makes that test fail. freqtrade's `lookahead-analysis` also reports no bias.

## 3. Risk
- **Position size.** Stake = `sls_risk_per_trade` (0.5%) × equity / stop distance / leverage, capped at the per-slot stake. In the smoke backtest, a stopped-out trade loses about 0.5% of equity.
- **Plan frozen at entry.** The stop, target and deadline are fixed from the signal candle when the trade opens. The stop is enforced with `custom_stoploss` and an on-exchange stop. The target is enforced with `custom_roi`, and the deadline with `custom_exit`.
- **Breakeven.** The stop moves to breakeven plus fees once the trade has been 1R in profit.
- **Portfolio limits.** At most 4 open trades and at most 3 in the same direction. Protections (Cooldown, StoplossGuard, MaxDrawdown 8%) are defined in the strategy.

## 4. Files
| Path | What it is |
|---|---|
| `user_data/strategies/sessionkit/features.py` | Look-ahead-safe features: sessions, ranges, PDH/PDL, anchored VWAP, relative volume, sweeps, regime |
| `user_data/strategies/sessionkit/signals.py` | `Params`, feature assembly and the six modules. Shared by the strategy and the research script |
| `user_data/strategies/SessionLiquidityStack.py` | The freqtrade strategy, plus the `SLS_AsiaSweep`, `SLS_PDSweep`, `SLS_NYORB` and `SLS_Research` variants |
| `user_data/research/event_study.py` | Event study with day-clustered t-stats, placebo test and the gate |
| `user_data/config_sls_futures.json` | Binance USDT-M config with 20 majors, isolated margin, 0.05% fee |
| `user_data/tests/` | Unit and look-ahead tests, an end-to-end smoke backtest, and tests for the gate |

## 5. Research protocol (run on your machine)

**Step 1. Download data.** In futures mode, funding-rate and mark candles are downloaded automatically.
```bash
freqtrade download-data -c user_data/config_sls_futures.json --timeframes 5m 1h --timerange 20201201- --trading-mode futures
```

**Step 2. Run the event study (in-sample and out-of-sample).** The holdout period starts 2026-01-01; leave it out for now.
```bash
python user_data/research/event_study.py -c user_data/config_sls_futures.json --timerange 20210101-20251231 --oos-start 2024-07-01 --modules all
```
This writes `user_data/research/out/report.md`. A module stays on only if it **passes all four gates**:
1. In-sample (2021-01 to 2024-06): net mean > 0 with t > 2. The t-stat is clustered by UTC day across pairs, because 20 coins firing on the same day count as roughly one observation.
2. Out-of-sample (2024-07 to 2025-12): the mean has the same sign.
3. At least 2/3 of calendar years are positive.
4. It beats the placebo: the same trade plan, at the same hour and day type, on random days.

Research modules are only turned on if they pass, via `"sls_modules"` in the config.

**Step 3. Backtest the stack and each module on its own.**
```bash
freqtrade backtesting -c user_data/config_sls_futures.json --strategy-list SessionLiquidityStack SLS_AsiaSweep SLS_PDSweep SLS_NYORB --timerange 20210101-20251231 --breakdown month --export signals
```
```bash
freqtrade backtesting-analysis -c user_data/config_sls_futures.json --analysis-groups 0 1 2
```

**Step 4. Run the bias checks.**
```bash
freqtrade lookahead-analysis -c user_data/config_sls_futures.json --strategy SessionLiquidityStack --timerange 20240101-20240401
```
```bash
freqtrade recursive-analysis -c user_data/config_sls_futures.json --strategy SessionLiquidityStack --timerange 20240101-20240401
```

**Step 5. Check robustness instead of optimizing.** Do not hyperopt for the best point.
- Shift each parameter ±20% using `"sls_params"` in a second config, and look for a *plateau* of good results, not a single peak.
- Remove pairs one at a time.
- Double the fees (`--fee 0.001`).

```bash
freqtrade backtesting -c user_data/config_sls_futures.json -c sweep.json --timerange 20210101-20251231
```

**Step 6. Accept or reject on the out-of-sample period.** The bar is:
- profit factor > 1.2, daily Sharpe > 1, max drawdown < 15%, at least 150 trades;
- no single pair contributes more than 30% of profit;
- correlation with BTC buy-and-hold < 0.3.

If it clears the bar, run the **2026 holdout once**. If that fails, the idea is rejected. Do not re-tune on the holdout.

**Step 7. Paper trade.** Run 4–8 weeks of `dry_run` before going live, and compare live fills and slippage with the backtest.

## 6. Known limitations and caveats
- **Survivorship bias.** The 20-pair static list is chosen with today's knowledge. Results on it are optimistic.
- **Funding is lagged.** `pd_sweep` and `fund_window` use the *last settled* funding rate. The *predicted* rate (premium index) would be more timely. To add it, download `--candle-types premiumIndex` and merge it like funding.
- **Fills.** The backtest assumes market orders at the next open with a 0.05% fee. Liquidity sweeps happen at the moments when slippage is worst, so stress-test fees at 0.08–0.10%.
- **Synthetic tests check plumbing only.** They have no edge, and the event study correctly fails every module on them.
- **Ideas not built yet.** Order-flow confirmation (CVD / delta from `use_public_trades`) at sweep levels. Open-interest drops as a liquidation signal (needs an external data source). Weekend CME-gap behaviour.
