"""The event-study gate must reject noise and accept a genuine (planted) edge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from event_study import clustered_t, gate


def _events(mean_bp: float, seed: int, n_days: int = 900, per_day: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", periods=n_days, freq="D", tz="UTC")
    date = np.repeat(days, per_day)
    day_shock = np.repeat(rng.normal(0, 0.004, n_days), per_day)  # common factor per day
    ret = mean_bp / 1e4 + day_shock + rng.normal(0, 0.006, len(date))
    placebo = day_shock + rng.normal(0, 0.006, len(date)) - 0.001
    return pd.DataFrame(
        {"date": date, "day": date.floor("D"), "plan_ret": ret, "plan_placebo": placebo}
    )


def test_gate_accepts_planted_edge():
    res = gate(_events(25, seed=1), "plan_ret", pd.Timestamp("2024-01-01", tz="UTC"))
    assert res["pass"], res


def test_gate_rejects_noise():
    passes = sum(
        gate(_events(0, seed=s), "plan_ret", pd.Timestamp("2024-01-01", tz="UTC"))["pass"]
        for s in range(20)
    )
    assert passes <= 1  # false-positive rate of the whole gate stays small


def test_clustered_t_is_more_conservative_than_naive():
    ev = _events(10, seed=5)
    _, t_cl, n_cl = clustered_t(ev["plan_ret"], ev["day"])
    naive_t = ev["plan_ret"].mean() / ev["plan_ret"].std(ddof=1) * np.sqrt(len(ev))
    assert n_cl == 900
    assert abs(t_cl) < abs(naive_t)  # same-day events are correlated
