#!/usr/bin/env python3
"""
Event study for the Session & Liquidity modules: does the edge exist BEFORE we trade it?

For every module signal it measures, net of costs:
  * forward returns at fixed horizons (entry at the next candle open), signed by trade side,
  * the outcome of the actual trade plan (stop / target / deadline) in R multiples,
  * a placebo: same pair, same side, same UTC hour and weekday/weekend type, random day.

Standard errors are clustered by UTC day ACROSS pairs: crypto is close to a single-factor
market, so 20 pairs firing on the same day are not 20 independent observations.

The gate (applied per module) is:
  1. in-sample net mean > 0 with t > 2,
  2. out-of-sample net mean has the same sign,
  3. positive in at least 2/3 of calendar years,
  4. beats the placebo.

Usage (from the repo root, after ``freqtrade download-data``):

    python user_data/research/event_study.py -c user_data/config_sls_futures.json \
        --timerange 20210101-20251231 --oos-start 2024-07-01 --modules all

Writes ``report.md`` and ``events.csv`` to ``--out`` (default user_data/research/out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "user_data" / "strategies"))

from sessionkit.signals import (  # noqa: E402
    ALL_MODULES,
    MODULE_FUNCS,
    Params,
    build_features,
)

from freqtrade.configuration import TimeRange  # noqa: E402
from freqtrade.data.history import load_pair_history  # noqa: E402
from freqtrade.enums import CandleType  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_pair(datadir: Path, pair: str, futures: bool) -> dict[str, pd.DataFrame]:
    ct = CandleType.FUTURES if futures else CandleType.SPOT
    out = {
        "5m": load_pair_history(datadir=datadir, timeframe="5m", pair=pair, candle_type=ct),
        "1h": load_pair_history(datadir=datadir, timeframe="1h", pair=pair, candle_type=ct),
        "4h": load_pair_history(datadir=datadir, timeframe="4h", pair=pair, candle_type=ct),
    }
    out["funding"] = (
        load_pair_history(
            datadir=datadir, timeframe="8h", pair=pair, candle_type=CandleType.FUNDING_RATE
        )
        if futures
        else pd.DataFrame()
    )
    return out


# ---------------------------------------------------------------------------
# Event measurement
# ---------------------------------------------------------------------------
def simulate_plan(
    o: np.ndarray,
    h: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    ts: np.ndarray,
    i: int,
    side: int,
    stop: float,
    target: float,
    deadline: float,
    tf_sec: int,
    trail: float = np.nan,
    breakeven_r: float = 1.0,
    fee_pad: float = 0.001,
) -> tuple[float, int, str]:
    """Walk the trade forward from the entry at o[i+1]. Returns (gross return, bars held, reason).

    Mirrors the strategy's exit logic: time stop at the candle open, structural stop,
    target, then either a chandelier trail (``trail`` = distance in price) or a breakeven
    stop once the trade has been ``breakeven_r`` R in profit. Stops are only moved using
    COMPLETED candles, and if stop and target are both inside one candle, the stop is
    assumed first (conservative).
    """
    entry = o[i + 1]
    n = len(o)
    risk = abs(entry - stop)
    use_trail = np.isfinite(trail) and trail > 0
    best = entry
    for j in range(i + 1, n):
        if ts[j] >= deadline:
            return side * (o[j] / entry - 1), j - i - 1, "time"
        if side == 1:
            if lo[j] <= stop:
                return min(stop, o[j]) / entry - 1, j - i, "stop"
            if np.isfinite(target) and h[j] >= target:
                return max(target, o[j]) / entry - 1, j - i, "target"
            best = max(best, h[j])
            if use_trail:
                stop = max(stop, best - trail)
            elif breakeven_r > 0 and risk > 0 and best >= entry + breakeven_r * risk:
                stop = max(stop, entry * (1 + fee_pad))
        else:
            if h[j] >= stop:
                return -(max(stop, o[j]) / entry - 1), j - i, "stop"
            if np.isfinite(target) and lo[j] <= target:
                return -(min(target, o[j]) / entry - 1), j - i, "target"
            best = min(best, lo[j])
            if use_trail:
                stop = min(stop, best + trail)
            elif breakeven_r > 0 and risk > 0 and best <= entry - breakeven_r * risk:
                stop = min(stop, entry * (1 - fee_pad))
    return side * (c[-1] / entry - 1), n - i - 1, "eod"


def events_for_pair(
    pair: str,
    data: dict[str, pd.DataFrame],
    btc5: pd.DataFrame | None,
    p: Params,
    modules: list[str],
    horizons: list[int],
    cost: float,
    n_placebo: int,
    rng: np.random.Generator,
    n_plan_placebo: int = 5,
) -> pd.DataFrame:
    df5 = data["5m"]
    if df5.empty:
        return pd.DataFrame()
    is_btc = pair.startswith("BTC/")
    f = build_features(
        df5, 5, data["1h"], data["funding"], None if is_btc else btc5, p, df4h=data["4h"]
    )
    o, h, lo, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ts = (f["date"].astype("int64") // 10**9).to_numpy()
    n = len(f)

    # placebo pools: same UTC hour + weekend flag
    hour = f["date"].dt.hour.to_numpy()
    wknd = f["is_weekend"].to_numpy()
    pool_key = hour * 2 + wknd
    pools = {k: np.flatnonzero(pool_key == k) for k in np.unique(pool_key)}
    max_h = max(horizons)

    rows = []
    for name in modules:
        m = MODULE_FUNCS[name](f, p, is_btc)
        for side_name, side in (("long", 1), ("short", -1)):
            idx = np.flatnonzero(m[side_name].to_numpy())
            idx = idx[idx + max_h + 1 < n]
            stops = m[f"stop_{side_name}"].to_numpy(float)
            targets = m[f"target_{side_name}"].to_numpy(float)
            deadlines = m["deadline"].to_numpy(float)
            trails = m["trail"].to_numpy(float)
            for i in idx:
                entry = o[i + 1]
                rec = {
                    "pair": pair,
                    "module": name,
                    "side": side_name,
                    "date": f["date"].iat[i],
                    "entry": entry,
                }
                for hz in horizons:
                    rec[f"ret_{hz}"] = side * (c[i + hz] / entry - 1) - cost
                    pl = pools[pool_key[i]]
                    pl = pl[pl + hz + 1 < n]
                    if len(pl):
                        s = rng.choice(pl, size=min(n_placebo, len(pl)), replace=False)
                        rec[f"placebo_{hz}"] = float(
                            np.mean(side * (c[s + hz] / o[s + 1] - 1)) - cost
                        )
                risk = abs(entry - stops[i]) / entry
                gross, bars, why = simulate_plan(
                    o, h, lo, c, ts, i, side, stops[i], targets[i], deadlines[i], 300,
                    trail=trails[i],
                )  # fmt: skip
                # Placebo for the full plan: same stop / target distance (in %), same time
                # budget, same side, but at random candles of the same hour / day type.
                pl = pools[pool_key[i]]
                pl = pl[pl + max_h + 1 < n]
                if len(pl) and n_plan_placebo:
                    s = rng.choice(pl, size=min(n_plan_placebo, len(pl)), replace=False)
                    stop_frac = stops[i] / entry
                    tgt_frac = targets[i] / entry
                    budget = deadlines[i] - ts[i]
                    trail_frac = trails[i] / entry
                    sims = [
                        simulate_plan(
                            o, h, lo, c, ts, j, side, o[j + 1] * stop_frac,
                            o[j + 1] * tgt_frac, ts[j] + budget, 300,
                            trail=o[j + 1] * trail_frac,
                        )[0]
                        for j in s
                    ]  # fmt: skip
                    rec["plan_placebo"] = float(np.mean(sims)) - cost
                rec["plan_ret"] = gross - cost
                rec["plan_R"] = (gross - cost) / risk if risk > 0 else np.nan
                rec["plan_bars"] = bars
                rec["plan_exit"] = why
                rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def clustered_t(x: pd.Series, clusters: pd.Series) -> tuple[float, float, int]:
    """Mean, day-clustered t-stat, number of clusters."""
    x = x.astype(float)
    ok = x.notna()
    x, clusters = x[ok], clusters[ok]
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else np.nan), np.nan, n
    mu = x.mean()
    g = (x - mu).groupby(clusters).sum()
    se = np.sqrt((g**2).sum()) / n
    g_n = len(g)
    if g_n > 1:
        se *= np.sqrt(g_n / (g_n - 1))
    return float(mu), float(mu / se) if se > 0 else np.nan, g_n


def summarize(ev: pd.DataFrame, col: str) -> dict:
    mu, t, days = clustered_t(ev[col], ev["day"])
    return {
        "n": len(ev),
        "days": days,
        "mean_bps": mu * 1e4 if np.isfinite(mu) else np.nan,
        "t": t,
        "hit": float((ev[col] > 0).mean()) if len(ev) else np.nan,
    }


def gate(ev: pd.DataFrame, col: str, oos_start: pd.Timestamp) -> dict:
    is_ = ev[ev["date"] < oos_start]
    oos = ev[ev["date"] >= oos_start]
    s_is, s_oos = summarize(is_, col), summarize(oos, col)
    by_year = ev.groupby(ev["date"].dt.year)[col].mean()
    pcol = "plan_placebo" if col == "plan_ret" else col.replace("ret_", "placebo_")
    placebo_ok, placebo_t = False, np.nan
    if pcol in ev:
        both = ev[[col, pcol, "day"]].dropna()
        mu_d, placebo_t, _ = clustered_t(both[col] - both[pcol], both["day"])
        placebo_ok = bool(np.isfinite(mu_d) and mu_d > 0)
    checks = {
        "IS mean>0 & t>2": bool(s_is["mean_bps"] > 0 and (s_is["t"] or 0) > 2),
        "OOS same sign": bool(np.sign(s_oos["mean_bps"]) == np.sign(s_is["mean_bps"]))
        if s_oos["n"]
        else False,
        ">=2/3 years positive": bool((by_year > 0).mean() >= 2 / 3) if len(by_year) else False,
        "beats placebo": placebo_ok,
    }
    return {
        "IS": s_is,
        "OOS": s_oos,
        "checks": checks,
        "pass": all(checks.values()),
        "placebo_t": placebo_t,
    }


def _fmt(s: dict) -> str:
    return (
        f"n={s['n']:>5} days={s['days']:>4} mean={s['mean_bps']:7.1f}bp "
        f"t={s['t']:5.2f} hit={s['hit']:.2f}"
        if s["n"]
        else "n=0"
    )


def write_report(
    ev: pd.DataFrame, horizons: list[int], oos_start: pd.Timestamp, out: Path, cost: float
):
    lines = [
        "# Session & Liquidity event study",
        "",
        (
            f"Events: {len(ev)}; pairs: {ev['pair'].nunique()}; "
            f"period {ev['date'].min():%Y-%m-%d} -> {ev['date'].max():%Y-%m-%d}; "
            f"OOS from {oos_start:%Y-%m-%d}; round-trip cost {cost * 1e4:.1f} bp."
        ),
        "",
        (
            "t-stats are clustered by UTC day across pairs. `plan_ret` is the full trade plan "
            "(stop / target / time stop), conservative intrabar ordering."
        ),
        "",
    ]
    verdicts = []
    for module, g in ev.groupby("module"):
        lines += [f"## {module}", ""]
        for col in ["plan_ret"] + [f"ret_{h}" for h in horizons]:
            res = gate(g, col, oos_start)
            lines.append(f"- **{col}** IS: `{_fmt(res['IS'])}`  OOS: `{_fmt(res['OOS'])}`")
            if col == "plan_ret":
                verdicts.append((module, res))
                lines.append(
                    "  - gate: "
                    + ", ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in res["checks"].items())
                    + f" -> **{'PASS' if res['pass'] else 'FAIL'}**"
                    + f" (edge vs placebo t={res['placebo_t']:.2f})"
                )
        pl = ["plan_placebo"] + [f"placebo_{h}" for h in horizons]
        pl = [c for c in pl if c in g]
        if pl:
            lines.append(
                "- placebo means (bp): " + ", ".join(f"{c}={g[c].mean() * 1e4:.1f}" for c in pl)
            )
        lines.append("")
        lines.append("| split | n | mean bp (plan) | hit | avg R |")
        lines.append("|---|---|---|---|---|")
        for key, part in list(g.groupby(g["date"].dt.year)) + list(g.groupby("side")):
            lines.append(
                f"| {key} | {len(part)} | {part['plan_ret'].mean() * 1e4:.1f} | "
                f"{(part['plan_ret'] > 0).mean():.2f} | {part['plan_R'].mean():.2f} |"
            )
        top = g.groupby("pair")["plan_ret"].agg(["count", "mean"]).sort_values("mean")
        lines.append("")
        lines.append(
            "Worst / best pairs (mean bp): "
            + ", ".join(f"{i} {r['mean'] * 1e4:.0f}" for i, r in top.head(3).iterrows())
            + " | "
            + ", ".join(f"{i} {r['mean'] * 1e4:.0f}" for i, r in top.tail(3).iterrows())
        )
        lines.append("")
    lines += ["## Verdict", ""]
    for module, res in verdicts:
        lines.append(f"- `{module}`: {'PASS' if res['pass'] else 'FAIL'}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    ev.to_csv(out / "events.csv", index=False)
    return verdicts


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", default=str(ROOT / "user_data" / "config_sls_futures.json"))
    ap.add_argument("--datadir", default=None, help="default: user_data/data/<exchange>")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument(
        "--timerange", default=None, help="YYYYMMDD-YYYYMMDD (events only; warm-up kept)"
    )
    ap.add_argument("--oos-start", default="2024-07-01")
    ap.add_argument("--modules", nargs="*", default=["all"])
    ap.add_argument("--horizons", nargs="*", type=int, default=[3, 12, 48], help="in 5m bars")
    ap.add_argument(
        "--cost-bps", type=float, default=10.0, help="round trip, default 2 x 5bp taker"
    )
    ap.add_argument("--placebo", type=int, default=20, help="random draws per event (horizons)")
    ap.add_argument("--plan-placebo", type=int, default=5, help="random plan replays per event")
    ap.add_argument("--params", default="{}", help="JSON overrides, e.g. '{\"pd_rr\": 1.2}'")
    ap.add_argument("--out", default=str(ROOT / "user_data" / "research" / "out"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    cfg = json.loads(Path(a.config).read_text())
    futures = cfg.get("trading_mode") == "futures"
    datadir = Path(a.datadir or ROOT / "user_data" / "data" / cfg["exchange"]["name"])
    pairs = a.pairs or cfg["exchange"]["pair_whitelist"]
    modules = list(ALL_MODULES) if a.modules == ["all"] else a.modules
    p = Params()
    for k, v in json.loads(a.params).items():
        setattr(p, k, type(getattr(p, k))(v))
    cost = a.cost_bps / 1e4
    rng = np.random.default_rng(a.seed)

    stake = cfg["stake_currency"]
    btc_pair = f"BTC/{stake}:{stake}" if futures else f"BTC/{stake}"
    btc5 = load_pair(datadir, btc_pair, futures)["5m"]

    frames = []
    for pair in pairs:
        data = load_pair(datadir, pair, futures)
        if data["5m"].empty:
            print(f"[skip] no 5m data for {pair} in {datadir}")
            continue
        ev = events_for_pair(
            pair, data, btc5, p, modules, a.horizons, cost, a.placebo, rng, a.plan_placebo
        )
        print(f"{pair}: {len(ev)} events")
        frames.append(ev)
    if not frames or all(f.empty for f in frames):
        print("No events. Did you download data? See SESSION_LIQUIDITY.md.")
        return 1
    ev = pd.concat(frames, ignore_index=True)
    if a.timerange:
        tr = TimeRange.parse_timerange(a.timerange)
        if tr.startts:
            ev = ev[ev["date"] >= pd.Timestamp(tr.startts, unit="s", tz="UTC")]
        if tr.stopts:
            ev = ev[ev["date"] < pd.Timestamp(tr.stopts, unit="s", tz="UTC")]
    ev["day"] = ev["date"].dt.floor("D")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    verdicts = write_report(ev, a.horizons, pd.Timestamp(a.oos_start, tz="UTC"), out, cost)
    print((out / "report.md").read_text())
    return 0 if any(v["pass"] for _, v in verdicts) else 2


if __name__ == "__main__":
    sys.exit(main())
