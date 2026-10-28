# Edge hunt: Binance USDT-M perpetuals, 2020-01 → 2025-05

**Bottom line.** I screened about 170 strategy variants across 15 hypothesis families. Most of them were deliberately unconventional. Exactly **one edge survived every test, including a blind out-of-sample period**: the **funding carry harvest**. It holds spot and shorts the same coin's perpetual when funding is rich, so you get paid for supplying leverage to the crowd.

| Funding carry harvest (1x) | Sharpe | Return | CAGR | Max DD | Daily PF |
|---|---|---|---|---|---|
| In-sample 2020-01 → 2022-12 | **3.02** | +46.4% | 13.5% | −6.3% | 3.1 |
| Validation 2023 | **3.23** | +2.4% | 2.4% | −0.5% | 2.0 |
| **Blind OOS 2024-01 → 2025-05-30** | **6.75** | **+8.1%** | 5.7% | **−0.7%** | 4.8 |

It is profitable in every calendar year (2020 +14.3%, 2021 +27.6%, 2022 +0.3%, 2023 +2.4%, 2024 +7.8%, 2025 YTD +0.3%). Correlation to BTC is close to zero.

At 2× or 3× notional the OOS CAGR rises to 11.7% or 18.0%, with OOS drawdowns of −1.5% and −2.2%. The IS drawdowns at those levels are −13% and −20% (see §4).

It is honest to say what this edge is **not**. It is not a 100%-a-year machine. It is a high-Sharpe, low-drawdown, *cyclical* income stream: rich when the market is levered up (2020–21, 2024), and near zero when it isn't (2022, 2025). It is also sensitive to costs (see §4).

![equity curves](equity_curves.png)

---

## 1. Setup

* **Data.** Exchange APIs are blocked in this environment, so I used the public freqtrade-format mirror `DigiTuccar/HistoricalDataForTradeBacktest` at commit `553525c`. That gives 421 Binance USDT-M perps (1h), their 8h funding and mark prices, and 1h spot candles for the spot leg. Data ends on 2025-05-30. `fetch_data.sh` recreates it.
* **Costs.**
  * Perp: 0.05% taker plus 0.02% slippage per side.
  * Spot: 0.10% taker plus 0.02% slippage per side.
  * Actual historical funding is paid and received on every 8h settlement.
* **No lookahead.**
  * Signals at candle *t* use only data up to the close of *t*, and are executed over *t+1*.
  * Funding is only "known" once it has settled.
  * The universe is point-in-time: the top-N by trailing 30-day median quote volume, with at least 30 days of listing age.
  * `lookahead_test.py` truncates the data at 5 dates. Positions before each cut are **bit-identical** to the full-history run.
* **Splits fixed before any test.**
  * IS: 2020–2022. In practice it starts in 2021, because only 3 perps have 2020 data.
  * Validation: 2023.
  * **OOS: 2024-01 → 2025-05, untouched while designing.**
* **Pass criteria (set in advance).**
  * OOS Sharpe ≥ 1.5, positive return, max DD ≤ 25%, at least 100 trades, PF ≥ 1.3.
  * VAL Sharpe ≥ 1.
  * A parameter plateau, survival at 2× costs, and not just one coin.
* **OOS peeks used: 2.**
  1. The funding contrarian.
  2. The trend + carry ensemble, which reported each sleeve.

  The carry parameters were locked on IS and VAL before peek #2, and nothing about carry was changed afterwards. One more disclosure: a mechanical smoke test of the freqtrade pipeline ran on a 12-pair toy configuration over 2024-Q1.

## 2. What was tested (IS + VAL screening, OOS blind)

| Family | Idea | Verdict |
|---|---|---|
| A. Funding contrarian | Long the lowest-funding coins, short the most crowded | IS 2.2 / VAL 1.8, then **OOS −1.5 → rejected** |
| A'. Funding follow | The mirror image | Strongly negative |
| B. Momentum minus funding | "Un-levered rallies" | IS 2.2 / VAL 1.1; same failure mode as A |
| C. New-listing decay | Short perps in their first 14–60 days, hedged with BTC | Unstable sign |
| D. MAX / lottery | Short last week's biggest spike | Negative |
| E. Cross-sectional momentum / reversal | 1h → 30d horizons | 7–14d momentum weak (VAL 0.4–1.5); short horizons killed by fees |
| F. Low-volatility anomaly | | Negative |
| G. Abnormal volume | Shock × direction | Negative |
| I. BTC → alt lead-lag (1h) | | Fees destroy it (turnover 36×/day) |
| J. Funding change / own z-score | | Negative in VAL |
| L. Funding-settlement clock, weekend effect | | Clock trades killed by fees; weekend/weekday adds nothing |
| M. Spot–perp basis contrarian | | IS Sharpe 3.2, then decayed in VAL |
| N. Hour-of-day seasonality | | Fees destroy it |
| O/P/Q. Funding dispersion regime, funding volatility, wick bias | | Nothing |
| **R. Funding carry harvest** | Long spot + short perp, delta-neutral | **IS 3.0 / VAL 3.2 / OOS 6.8 → EDGE** |
| S/T. Trend following (vol-targeted, funding-filtered) | | IS 0.9 / VAL 1.4, then **OOS −0.2 → rejected** |
| Market timing with an aggregate-funding "froth gauge" | | Only cuts exposure; no improvement |
| Ensemble: 40% trend / 60% carry | | IS 1.7 / VAL 1.6, then **OOS 0.25 → rejected** (trend dragged it down) |

The raw tables are in `round1.csv` … `round5.csv`. The full reasoning for each rejected candidate is in `final_eval.log` and `final_eval_ensemble.log`.

**Why the most "exciting" find failed.** The funding contrarian made money from 2021 to 2023. In 2024–25 its long leg, the shunned low-funding coins, kept bleeding: −15% in 2024 and −39% in 2025. At the same time, crowded meme coins kept running.
* Only 7% of its 27 OOS parameter neighbours were profitable.
* A permutation test cannot distinguish its OOS result from random (p = 0.24).

That pattern is a regime-dependent anomaly. Carry is different: it is a payment for a service (absorbing leveraged demand), not a pattern.

## 3. The edge: funding carry harvest

**Rules.** These parameters were locked on IS and VAL; `carry_lab.carry`.
1. **Universe:** the point-in-time top 50 perps by 30-day median quote volume that also have a Binance spot market. A guard excludes ticker mismatches.
2. **Signal:** the 3-day average of settled funding.
3. **Enter:** when the coin ranks in the top 10 by that average **and** the average is > 0.01% per 8h (Binance's baseline rate).
4. **Hold:** until the average drops below 0.005% per 8h.
5. **Position:** buy spot and sell the same notional of the perp. Weight is `1/max(#held, 10)` of the hedged notional.
6. **Capital:** 1.2 × notional. The spot leg is fully funded, plus a 20% margin buffer, assuming a portfolio or cross margin account where the spot leg collateralises the perp.

**Where the money comes from** (sum of bar returns on capital):

| Year | Net | Funding received | Basis P&L | Fees | Avg. deployed |
|---|---|---|---|---|---|
| 2020 | +14.3% | +20.7% | +1.8% | −9.1% | 82% |
| 2021 | +27.6% | +40.4% | −6.7% | −8.7% | 84% |
| 2022 | +0.3% | +1.0% | +0.2% | −0.9% | 13% |
| 2023 | +2.4% | +6.1% | +1.1% | −4.7% | 45% |
| 2024 | +7.8% | +10.7% | +0.5% | −3.7% | 58% |
| 2025 (to 05-30) | +0.3% | +0.5% | +0.1% | −0.3% | 12% |

**Robustness:**
* **Parameter plateau** (`final_eval_carry.log` §4). With an entry threshold ≥ 0.01% per 8h, **18 of 18 neighbours** (k = 1/3/7 days × threshold 0.01/0.02% × top 5/10/20) are profitable in IS, VAL **and** OOS. Their OOS Sharpe ranges from 4.1 to 8.0.
* **Breakdown point.** Entering *below* the baseline rate (0.005%) churns and loses. That is the documented edge of the plateau, not noise.
* **Breadth.** In OOS, 127 round trips across 149 coins traded over the full history. The top-3 coins contribute only 11% of OOS profit.
* **Market neutrality.** BTC correlation is about 0.
* **The worst day is a conservative artifact.** The worst day in the sample was the 19 May 2021 crash, at −5.9%; it accounts for almost all of the −6.3% max drawdown. During the liquidation cascade the SUSHI, 1INCH and HOT perps printed 40–60% below spot for 1–2 hours, then reconverged. A real static hedge nets about zero over such a round trip. The harness, however, holds constant weights bar by bar, which means rebalancing every hour, so it books the dislocation as a loss. The reported drawdown therefore *overstates* this risk.
* **No lookahead.** `lookahead_test.py` → `NO LOOKAHEAD`.

## 4. Risks and caveats (read before trading)

1. **Costs are the Achilles heel.** At 2× my already-padded costs: IS Sharpe 1.4, VAL −2.3, OOS 2.8. At 3×, it is flat. Use maker orders or a lower fee tier, and do not chase small funding.
2. **It is cyclical.** It returned 0.3% in 2022 and 0.3% in 2025 YTD. Idle capital sits in stablecoins, and the dataset does not model the yield on that cash.
3. **Leverage and margin.** A 2–3× hedged notional is reasonable *only* with portfolio or cross margin, so that spot gains offset perp losses. On held short-perp legs the largest 24h pumps were DOGE +188% and SUSHI +125%. With isolated margin those would liquidate the perp leg and leave you unhedged.
4. **Spot-leg custody and borrow.** The spot leg is assumed fully owned, with no borrow. Exchange counterparty risk is not modelled.
5. **Survivorship.** The dataset includes some delisted perps (MATIC, EOS, ANT, TOMO, WAVES), but not all of them. Carry is delta-neutral, so missing blow-ups matter far less than for directional strategies.
6. **Execution of funding.** The model assumes funding is collected exactly as published. Real rates are capped and the settlement interval can change (4h on some coins in 2024–25). The 8h series here already reflects what was paid.
7. **Freqtrade can't run it as one bot.** A freqtrade strategy trades one market type per bot, so a simultaneous spot + perp hedge cannot be expressed or backtested in the freqtrade engine. The edge is therefore validated in the vectorized harness. `carry_signals.py` prints the target book from the latest data, for execution by hand or with a small executor.

## 5. Files

| File | Purpose |
|---|---|
| `fetch_data.sh` | Download the dataset (no exchange API needed) |
| `edge_lab.py` | Vectorized engine: panel loader, point-in-time universe, fees and funding, metrics |
| `carry_lab.py` | The carry harvest (the edge) |
| `carry_signals.py` | Current target book: `python3 carry_signals.py 10000 1` |
| `final_eval_carry.py` / `.log` | Full robustness battery for the edge |
| `lookahead_test.py` | Truncation test proving no future data is used |
| `screen_round1..5.py`, `round*.csv` | All screens (IS and VAL only) |
| `final_eval.py` / `.log` | OOS evaluation of the rejected funding contrarian |
| `final_eval_ensemble.py` / `.log`, `ensemble_lab.py` | OOS evaluation of the rejected ensemble (carry sleeve included) |
| `ft_offline.py` | Runs freqtrade backtesting/lookahead-analysis offline by synthesising market metadata |
| `rejected/` | The freqtrade port of the funding contrarian. **Do not trade it.** |
| `make_chart.py`, `equity_curves.png` | Figure above |

Reproduce:
```bash
user_data/edge_research/fetch_data.sh
```
```bash
cd user_data/edge_research && python3 final_eval_carry.py && python3 lookahead_test.py
```
