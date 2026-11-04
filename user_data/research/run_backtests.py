#!/usr/bin/env python3
"""
Run the Session & Liquidity backtest grid with freqtrade: every single-module strategy
(and optionally a portfolio of modules) on each split, in parallel worker processes.

Splits (fixed, so nobody "discovers" a better split after looking at results):
  is       2021-01-01 -> 2024-07-01   in-sample
  oos      2024-07-01 -> 2026-01-01   out-of-sample
  holdout  2026-01-01 -> 2026-11-01   final check, run ONCE (needs --confirm-holdout)

Each run writes to ``<out>/<split>/<name>/`` (freqtrade results zip + run.log), which
``equity_report.py`` reads.

Examples (repo root):
    python user_data/research/run_backtests.py --splits is oos
    python user_data/research/run_backtests.py --splits is oos \
        --portfolio asia_sweep pw_sweep trend_4h --name portfolio
    python user_data/research/run_backtests.py --splits holdout --confirm-holdout \
        --portfolio asia_sweep pw_sweep trend_4h --name portfolio --no-singles
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPLITS = {
    "is": "20210101-20240701",
    "oos": "20240701-20260101",
    "holdout": "20260101-20261101",
}
SINGLE_STRATEGIES = {
    "asia_sweep": "SLS_AsiaSweep",
    "pd_sweep": "SLS_PDSweep",
    "ny_orb": "SLS_NYORB",
    "late_mom": "SLS_LateMom",
    "wknd_vwap": "SLS_WkndVwap",
    "fund_window": "SLS_FundWindow",
    "pw_sweep": "SLS_PWSweep",
    "eq_sweep": "SLS_EqSweep",
    "liq_cascade": "SLS_LiqCascade",
    "asia_squeeze": "SLS_AsiaSqueeze",
    "tod_season": "SLS_TodSeason",
    "cme_gap": "SLS_CmeGap",
    "trend_4h": "SLS_Trend4h",
}


def build_jobs(a: argparse.Namespace, out: Path) -> list[dict]:
    jobs = []
    singles = [] if a.no_singles else (a.modules or list(SINGLE_STRATEGIES))
    for split in a.splits:
        for module in singles:
            jobs.append(
                {
                    "split": split,
                    "name": module,
                    "strategy": SINGLE_STRATEGIES[module],
                    "override": None,
                    "fee": a.fee,
                }
            )
        if a.portfolio:
            jobs.append(
                {
                    "split": split,
                    "name": a.name,
                    "strategy": "SessionLiquidityStack",
                    "override": {"sls_modules": a.portfolio},
                    "fee": a.fee,
                }
            )
    return jobs


def run_job(job: dict, a: argparse.Namespace, out: Path) -> tuple[dict, int, float]:
    d = out / job["split"] / job["name"]
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "freqtrade", "backtesting",
        "-c", str(a.config),
        "--strategy", job["strategy"],
        "--strategy-path", str(ROOT / "user_data" / "strategies"),
        "--timerange", SPLITS[job["split"]],
        "--export", "trades",
        "--backtest-directory", str(d),
        "--cache", "none",
        "--fee", str(job["fee"]),
    ]  # fmt: skip
    if a.datadir:
        cmd += ["--datadir", str(a.datadir)]
    if job["override"]:
        ov = d / "override.json"
        ov.write_text(json.dumps(job["override"]))
        cmd[4:4] = ["-c", str(ov)]
    t0 = time.time()
    with (d / "run.log").open("w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return job, rc, time.time() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", default=ROOT / "user_data" / "config_sls_futures.json")
    ap.add_argument("--datadir", default=None)
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["is", "oos"])
    ap.add_argument("--modules", nargs="*", choices=list(SINGLE_STRATEGIES), default=None)
    ap.add_argument("--no-singles", action="store_true", help="skip single-module runs")
    ap.add_argument("--portfolio", nargs="*", default=None, help="modules for a combined run")
    ap.add_argument("--name", default="portfolio", help="result folder name of the portfolio")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--jobs", type=int, default=3, help="parallel freqtrade processes")
    ap.add_argument("--out", default=ROOT / "user_data" / "backtest_results" / "sls")
    ap.add_argument("--confirm-holdout", action="store_true")
    a = ap.parse_args(argv)

    if "holdout" in a.splits and not a.confirm_holdout:
        ap.error("The holdout is run once, on the final portfolio. Pass --confirm-holdout.")
    out = Path(a.out)
    jobs = build_jobs(a, out)
    if not jobs:
        ap.error("Nothing to run.")
    print(f"{len(jobs)} backtests, {a.jobs} in parallel -> {out}")
    failed = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = [pool.submit(run_job, j, a, out) for j in jobs]
        for f in as_completed(futs):
            job, rc, secs = f.result()
            status = "ok" if rc == 0 else f"FAILED rc={rc} (see run.log)"
            failed += rc != 0
            print(f"[{job['split']:>7}] {job['name']:<14} {secs / 60:5.1f} min  {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
