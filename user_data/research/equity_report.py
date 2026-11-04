#!/usr/bin/env python3
"""
Equity curves and metrics for the Session & Liquidity backtest grid.

Reads ``<results>/<split>/<name>/`` folders written by ``run_backtests.py``, builds a
daily equity curve per strategy, chains the splits (IS -> OOS -> holdout) into one growth
index, adds a BTC buy-and-hold benchmark, and writes:

  * ``metrics.csv``         one row per (strategy, split)
  * ``equity_daily.csv``    chained growth index per strategy + BTC
  * ``report_data.json``    everything above + monthly table + gate verdicts (for the page)
  * ``portfolio_vs_btc.png``, ``modules_grid.png``

Usage (repo root):
    python user_data/research/equity_report.py \
        --results user_data/backtest_results/sls \
        --events user_data/research/out/events.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

SPLIT_ORDER = ["is", "oos", "holdout"]
SPLIT_LABEL = {"is": "In-sample", "oos": "Out-of-sample", "holdout": "Holdout"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_run(folder: Path) -> tuple[pd.DataFrame, dict] | None:
    from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

    try:
        stats = load_backtest_stats(folder)
    except ValueError:
        return None
    strat_name, strat = next(iter(stats["strategy"].items()))
    trades = load_backtest_data(folder, strategy=strat_name)
    return trades, strat


def daily_equity(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, balance: float):
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    if trades.empty:
        return pd.Series(balance, index=days)
    pnl = trades.groupby(trades["close_date"].dt.floor("D"))["profit_abs"].sum()
    pnl = pnl.reindex(days, fill_value=0.0)
    return balance + pnl.cumsum()


def exposure(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if trades.empty:
        return 0.0
    iv = trades[["open_date", "close_date"]].sort_values("open_date").to_numpy()
    covered = pd.Timedelta(0)
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    covered += cur_e - cur_s
    return covered / (end - start)


def metrics(eq: pd.Series, trades: pd.DataFrame, btc_ret: pd.Series | None, span) -> dict:
    ret = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    dd = eq / eq.cummax() - 1
    sd = ret.std()
    down = ret[ret < 0].std()
    wins = trades["profit_abs"] > 0 if not trades.empty else pd.Series(dtype=bool)
    gross_win = trades.loc[wins, "profit_abs"].sum() if not trades.empty else 0.0
    gross_loss = -trades.loc[~wins, "profit_abs"].sum() if not trades.empty else 0.0
    corr = np.nan
    if btc_ret is not None and len(ret) > 30:
        joined = pd.concat([ret, btc_ret], axis=1, join="inner").dropna()
        if len(joined) > 30 and joined.iloc[:, 0].std() > 0:
            corr = float(joined.corr().iloc[0, 1])
    return {
        "trades": len(trades),
        "win_rate": float(wins.mean()) if len(trades) else np.nan,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.nan,
        "total_return": float(total),
        "cagr": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "sharpe": float(ret.mean() / sd * math.sqrt(365)) if sd > 0 else np.nan,
        "sortino": float(ret.mean() / down * math.sqrt(365)) if down and down > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "avg_hold_h": float(
            ((trades["close_date"] - trades["open_date"]).dt.total_seconds() / 3600).mean()
        )
        if len(trades)
        else np.nan,
        "exposure": float(exposure(trades, *span)),
        "corr_btc": corr,
    }


def btc_daily(datadir: Path, stake: str) -> pd.Series | None:
    from freqtrade.data.history import load_pair_history
    from freqtrade.enums import CandleType

    df = load_pair_history(
        datadir=datadir,
        timeframe="1d",
        pair=f"BTC/{stake}:{stake}",
        candle_type=CandleType.FUTURES,
    )
    if df.empty:
        return None
    return df.set_index("date")["close"]


def gate_verdicts(events_csv: Path, oos_start: str) -> dict:
    from event_study import gate

    ev = pd.read_csv(events_csv, parse_dates=["date"])
    ev["date"] = pd.to_datetime(ev["date"], utc=True)
    ev["day"] = ev["date"].dt.floor("D")
    out = {}
    for module, g in ev.groupby("module"):
        res = gate(g, "plan_ret", pd.Timestamp(oos_start, tz="UTC"))
        out[module] = {
            "pass": bool(res["pass"]),
            "checks": res["checks"],
            "is_mean_bps": res["IS"]["mean_bps"],
            "is_t": res["IS"]["t"],
            "oos_mean_bps": res["OOS"]["mean_bps"],
            "oos_t": res["OOS"]["t"],
            "n_events": len(g),
        }
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _fmt_axes(ax, log: bool = False):
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, NullFormatter

    loc = mdates.AutoDateLocator(maxticks=6)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    if log:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())


def _shade(ax, bounds: dict):
    colors = {"oos": "#2a78d6", "holdout": "#d9a400"}
    for split, (s, e) in bounds.items():
        if split in colors:
            ax.axvspan(s, e, color=colors[split], alpha=0.07, lw=0)
            ax.text(
                s + (e - s) / 2, 1.0, SPLIT_LABEL[split], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=colors[split],
            )  # fmt: skip


def plot_main(growth: pd.DataFrame, focus: str, bounds: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(growth.index, growth[focus], color="#1f6f43", lw=1.8, label=focus)
    if "BTC buy & hold" in growth:
        ax.plot(
            growth.index, growth["BTC buy & hold"], color="#9a9a9a", lw=1.1, label="BTC buy & hold"
        )
    ax.set_yscale("log")
    _fmt_axes(ax, log=True)
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    _shade(ax, bounds)
    dd = growth[focus] / growth[focus].cummax() - 1
    ax2.fill_between(dd.index, dd.to_numpy() * 100, 0, color="#b23b3b", alpha=0.5, lw=0)
    ax2.set_ylabel("Drawdown %")
    _fmt_axes(ax2)
    ax2.grid(alpha=0.25)
    _shade(ax2, {k: v for k, v in bounds.items()})
    for t in ax2.texts[:]:
        t.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid(
    growth: pd.DataFrame, names: list[str], table: pd.DataFrame, bounds: dict, path: Path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), sharex=True, squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        g = growth[name]
        oos = table[(table["strategy"] == name) & (table["split"] == "oos")]
        good = not oos.empty and oos["sharpe"].iloc[0] > 0
        ax.plot(g.index, g, color="#1f6f43" if good else "#b23b3b", lw=1.2)
        ax.axhline(1, color="#999", lw=0.6)
        _shade(ax, bounds)
        for t in ax.texts[:]:
            t.remove()
        label = name
        if not oos.empty:
            label += f"   OOS Sharpe {oos['sharpe'].iloc[0]:.2f}"
        ax.set_title(label, fontsize=9, loc="left")
        _fmt_axes(ax)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=ROOT / "user_data" / "backtest_results" / "sls")
    ap.add_argument("-c", "--config", default=ROOT / "user_data" / "config_sls_futures.json")
    ap.add_argument("--datadir", default=None)
    ap.add_argument("--events", default=None, help="event_study events.csv for gate verdicts")
    ap.add_argument("--oos-start", default="2024-07-01")
    ap.add_argument("--focus", default="portfolio", help="strategy shown in the main chart")
    ap.add_argument("--out", default=ROOT / "user_data" / "research" / "out")
    a = ap.parse_args(argv)

    cfg = json.loads(Path(a.config).read_text())
    datadir = Path(a.datadir or ROOT / "user_data" / "data" / cfg["exchange"]["name"])
    res_root = Path(a.results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, tuple]] = {}
    for split in SPLIT_ORDER:
        for folder in sorted((res_root / split).glob("*")) if (res_root / split).is_dir() else []:
            loaded = load_run(folder)
            if loaded:
                runs.setdefault(folder.name, {})[split] = loaded
    if not runs:
        print(f"No backtest results under {res_root}")
        return 1

    btc = btc_daily(datadir, cfg["stake_currency"])
    btc_ret = btc.pct_change() if btc is not None else None

    rows, curves, bounds = [], {}, {}
    for name, splits in runs.items():
        chained, level = [], 1.0
        for split in SPLIT_ORDER:
            if split not in splits:
                continue
            trades, strat = splits[split]
            start = pd.Timestamp(strat["backtest_start"], tz="UTC")
            end = pd.Timestamp(strat["backtest_end"], tz="UTC")
            bounds.setdefault(split, (start, end))
            bal = float(strat["starting_balance"])
            eq = daily_equity(trades, start, end, bal)
            m = metrics(eq, trades, btc_ret, (start, end))
            rows.append({"strategy": name, "split": split, **m})
            g = eq / bal * level
            level = float(g.iloc[-1])
            chained.append(g)
        curve = pd.concat(chained)
        curves[name] = curve[~curve.index.duplicated(keep="last")]

    table = pd.DataFrame(rows)
    growth = pd.DataFrame(curves).sort_index().ffill()
    if btc is not None:
        b = btc.reindex(growth.index, method="ffill")
        growth["BTC buy & hold"] = b / b.dropna().iloc[0]
    table.to_csv(out / "metrics.csv", index=False)
    growth.to_csv(out / "equity_daily.csv", index_label="date")

    focus = a.focus if a.focus in growth else table.sort_values("sharpe").iloc[-1]["strategy"]
    names = [n for n in growth.columns if n not in ("BTC buy & hold",) and n != "portfolio"]
    names.sort(key=lambda n: -table[(table.strategy == n) & (table.split == "oos")]["sharpe"].sum())
    plot_main(growth, focus, bounds, out / "portfolio_vs_btc.png")
    plot_grid(growth, names, table, bounds, out / "modules_grid.png")

    f = growth[focus]
    monthly = (f.resample("ME").last().pct_change().dropna() * 100).round(2)
    gates = gate_verdicts(Path(a.events), a.oos_start) if a.events else {}
    data = {
        "focus": focus,
        "bounds": {k: [str(v[0].date()), str(v[1].date())] for k, v in bounds.items()},
        "series": {
            "date": [str(d.date()) for d in growth.index],
            **{c: [None if pd.isna(x) else round(float(x), 5) for x in growth[c]] for c in growth},
        },
        "metrics": json.loads(table.to_json(orient="records")),
        "monthly_focus": {str(k.date())[:7]: float(v) for k, v in monthly.items()},
        "gates": gates,
    }
    (out / "report_data.json").write_text(json.dumps(data, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    print(f"\nWrote {out}/metrics.csv, equity_daily.csv, report_data.json, *.png (focus={focus})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
