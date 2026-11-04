# Runbook: real-data run of the Session & Liquidity research

These are the steps to run in a fresh cloud container, in order. Each step must succeed before you start the next one.
The network policy must allow `fapi.binance.com`, `api.binance.com` and `data.binance.vision`.

## Opening prompt for a new session

> On branch `claude/magical-wozniak-5btji9`, follow `user_data/research/RUNBOOK.md` end to end.
> Run the real-data research and backtests, and show me the equity curves as an artifact page.
> Report every module's result, including the losers.

## 0. Connectivity
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://fapi.binance.com/fapi/v1/ping
```
This must print `200`. If it prints anything else, stop: the network policy still blocks Binance.

## 1. Environment (about 5 min)
`sdnotify` fails to build its wheel, and the full dev requirements pull in torch, so install the pieces separately.
```bash
grep -v -i sdnotify requirements.txt > /tmp/req.txt && pip install -q -r /tmp/req.txt
```
```bash
pip install -q --no-deps -e .
```
```bash
mkdir -p /tmp/sdn && pip download -q --no-deps --no-binary :all: sdnotify==0.3.2 -d /tmp/sdn && tar xzf /tmp/sdn/sdnotify-0.3.2.tar.gz -C /tmp/sdn && cp -r /tmp/sdn/sdnotify-0.3.2/sdnotify "$(python -c 'import site;print(site.getsitepackages()[0])')/"
```
```bash
pip install -q pytest pytest-mock pytest-xdist pytest-asyncio pytest-random-order time-machine matplotlib
```
```bash
python -m pytest user_data/tests -q -n0 -p no:cacheprovider
```
Expected result: `28 passed`.

## 2. Data (about 20–40 min)
```bash
python -m freqtrade download-data -c user_data/config_sls_futures.json --timeframes 5m 1h 4h 1d --timerange 20201201-20261031 --trading-mode futures
```
Funding-rate and mark candles are downloaded automatically. Then record the first candle of each pair. Pairs that listed late have a shorter history, and this goes into the survivorship note.

## 3. Event study (edge before strategy)
```bash
python user_data/research/event_study.py -c user_data/config_sls_futures.json --timerange 20210101-20260101 --oos-start 2024-07-01 --modules all
```
This writes `user_data/research/out/report.md` and `events.csv`. Read the gate verdict for each module.

## 4. Single-module backtest grid (about 1 h)
```bash
python user_data/research/run_backtests.py --splits is oos --jobs 3
```
This runs 13 modules, in-sample and out-of-sample. Results go to `user_data/backtest_results/sls/<split>/<module>/`.
If memory runs short, use `--jobs 2`.

## 5. Portfolio
A module joins the portfolio only if it **passed the event-study gate and had an out-of-sample Sharpe above 0** in the grid.
```bash
python user_data/research/run_backtests.py --splits is oos --no-singles --portfolio <modules...> --name portfolio
```
Then the fee stress test, at double fees:
```bash
python user_data/research/run_backtests.py --splits is oos --no-singles --portfolio <modules...> --name portfolio --fee 0.001 --out user_data/backtest_results/sls_fee2x
```
If no module qualifies, say so plainly and skip steps 5–7 for the portfolio. Still show every module's curve.

## 6. Bias checks on the portfolio modules
```bash
python -m freqtrade lookahead-analysis -c user_data/config_sls_futures.json --strategy SessionLiquidityStack --timerange 20240101-20240401
```
```bash
python -m freqtrade recursive-analysis -c user_data/config_sls_futures.json --strategy SessionLiquidityStack --timerange 20240101-20240401
```
Before running these, set `"sls_modules"` in `user_data/config_sls_futures.json` to the portfolio modules.

## 7. Holdout: run once
```bash
python user_data/research/run_backtests.py --splits holdout --confirm-holdout --no-singles --portfolio <modules...> --name portfolio
```
Never re-tune anything after seeing these results.

## 8. Report
```bash
python user_data/research/equity_report.py --events user_data/research/out/events.csv
```
- Publish the equity-curve page as an Artifact. Run the Artifact quickstart first, then load the `dataviz` and `artifact-design` skills. The page includes:
  - portfolio vs BTC, with a drawdown panel;
  - one small chart per module, losers included;
  - the metrics table and the monthly returns;
  - gate verdicts and the fee-stress result.
- Send the PNGs from `user_data/research/out/`.
- Update `user_data/strategies/SESSION_LIQUIDITY.md` with the real results, failures included. Leave `sls_modules` set to the surviving modules.
- Commit and push. Data and backtest results stay gitignored.
