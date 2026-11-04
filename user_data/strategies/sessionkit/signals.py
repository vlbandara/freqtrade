"""
Module signals for the Session & Liquidity stack.

The same code is used by the freqtrade strategy (``SessionLiquidityStack``) and by the
research event study (``user_data/research/event_study.py``), so the research measures
exactly what the strategy trades.

Every module returns, per candle: ``long`` / ``short`` booleans and the trade plan for
an entry at the NEXT candle open: structural ``stop``, ``target`` and ``deadline``
(UTC epoch seconds, a hard time stop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sessionkit.features import (
    NEW_YORK,
    UTC,
    anchored_vwap,
    atr,
    cme_weekend_anchor,
    detect_sweep,
    efficiency_ratio,
    equal_levels,
    first_breakout,
    funding_features,
    hourly_seasonality,
    htf_trend,
    in_window,
    local_day,
    merge_asof_backward,
    opening_range,
    prior_day_levels,
    prior_week_levels,
    realized_vol,
    rel_volume_tod,
    return_z,
    rolling_pct_rank,
    session_flags,
    session_range,
)


CORE_MODULES = ("asia_sweep", "pd_sweep", "ny_orb")
RESEARCH_MODULES = ("late_mom", "wknd_vwap", "fund_window")
# second wave: liquidity / session-timing / trend families
NEW_MODULES = (
    "pw_sweep",
    "eq_sweep",
    "liq_cascade",
    "asia_squeeze",
    "tod_season",
    "cme_gap",
    "trend_4h",
)
ALL_MODULES = CORE_MODULES + RESEARCH_MODULES + NEW_MODULES


@dataclass
class Params:
    # --- shared ---------------------------------------------------------------
    atr_len: int = 14
    rvol_lookback_days: int = 10
    stop_buffer_atr: float = 0.25  # added beyond the structural stop
    min_stop_pct: float = 0.0015  # stops tighter than this are noise + fees
    max_stop_pct: float = 0.02  # stops wider than this break the risk budget
    rv_pct_max: float = 0.95  # skip the top 5% realized-vol regime (news shocks)
    btc_z_block: float = 2.0  # block alt entries against a violent BTC move
    # --- asia_sweep -------------------------------------------------------------
    asia_rvol_min: float = 1.5
    asia_width_ratio_max: float = 1.3  # Asia range vs its 20d median ("compressed")
    asia_min_rr: float = 1.0
    asia_max_hold_h: float = 4.0
    mr_er_pct_max: float = 0.7  # mean-reversion only when 1h trend efficiency is not extreme
    # --- pd_sweep ---------------------------------------------------------------
    pd_rvol_min: float = 2.0
    pd_rr: float = 1.5
    pd_max_hold_h: float = 4.0
    pd_funding_crowd: float = 0.7  # funding percentile marking a crowded side
    pd_require_funding: bool = False
    # --- ny_orb -----------------------------------------------------------------
    orb_rvol_min: float = 1.5
    orb_er_pct_min: float = 0.5
    orb_width_atr1h_max: float = 1.5
    orb_rr: float = 2.0
    orb_entry_until: str = "12:00"  # NY time
    # --- late_mom (research) ----------------------------------------------------
    lm_entry_time: str = "15:30"  # NY time, signal candle open
    lm_z_min: float = 1.0
    lm_stop_atr1h: float = 1.5
    # --- wknd_vwap (research) ---------------------------------------------------
    wk_z_entry: float = 2.0
    wk_min_bars: int = 36
    wk_stop_atr1h: float = 1.0
    wk_max_hold_h: float = 6.0
    # --- fund_window (research) -------------------------------------------------
    fw_pct_extreme: float = 0.9
    fw_stop_atr1h: float = 1.0
    # --- pw_sweep ---------------------------------------------------------------
    pw_rvol_min: float = 1.5
    pw_rr_cap: float = 2.0
    pw_min_rr: float = 1.0
    pw_max_hold_h: float = 24.0
    # --- eq_sweep ---------------------------------------------------------------
    eq_swing_k: int = 3
    eq_lookback: int = 288
    eq_tol_atr: float = 0.1
    eq_rvol_min: float = 1.5
    eq_rr: float = 1.5
    eq_max_hold_h: float = 4.0
    # --- liq_cascade ------------------------------------------------------------
    lc_z: float = 4.0
    lc_rvol_min: float = 4.0
    lc_wick_min: float = 0.4
    lc_er_pct_max: float = 0.9
    lc_stop_atr1h: float = 1.0
    lc_rr: float = 1.5
    lc_max_hold_h: float = 2.0
    # --- asia_squeeze -----------------------------------------------------------
    sq_width_ratio_max: float = 0.7
    sq_rv_pct_max: float = 0.3
    sq_rvol_min: float = 1.5
    sq_rr: float = 2.0
    # --- tod_season -------------------------------------------------------------
    ts_t_min: float = 2.5
    ts_stop_atr1h: float = 1.5
    # --- cme_gap ----------------------------------------------------------------
    cg_gap_atr1d: float = 1.0
    cg_stop_atr1h: float = 1.0
    # --- trend_4h ---------------------------------------------------------------
    tr_er_pct_min: float = 0.5
    tr_rvol_min: float = 1.2
    tr_trail_atr4h: float = 3.0
    tr_max_stop_pct: float = 0.08
    tr_max_hold_d: float = 10.0
    enabled: dict = field(default_factory=lambda: {m: (m in CORE_MODULES) for m in ALL_MODULES})


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
def regime_1h(df1h: pd.DataFrame, atr_len: int = 14) -> pd.DataFrame:
    """1h regime features: trend efficiency, realized-vol percentile, ATR."""
    out = df1h[["date"]].copy()
    out["er"] = efficiency_ratio(df1h["close"], 24)
    out["er_pct"] = rolling_pct_rank(out["er"], 720, min_periods=240)
    out["rv"] = realized_vol(df1h["close"], 24)
    out["rv_pct"] = rolling_pct_rank(out["rv"], 720, min_periods=240)
    out["atr"] = atr(df1h, atr_len)
    return out


def build_features(
    df: pd.DataFrame,
    tf_minutes: int,
    df1h: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    btc: pd.DataFrame | None = None,
    p: Params | None = None,
    df4h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add every feature the modules need. ``df`` is the 5m frame of the traded pair."""
    from freqtrade.strategy import merge_informative_pair

    p = p or Params()
    df = df.copy()

    df = pd.concat([df, session_flags(df)], axis=1)
    df["utc_day"] = df["date"].dt.tz_localize(None).dt.floor("D")
    df["ny_day"] = local_day(df["date"], NEW_YORK)
    df["atr"] = atr(df, p.atr_len)
    df["rvol"] = rel_volume_tod(df, p.rvol_lookback_days, p.rvol_lookback_days // 2)
    df["rvol_max3"] = df["rvol"].rolling(3, min_periods=1).max()

    # Asia range (finalized at 07:00 UTC, valid for the rest of the UTC day)
    asia = session_range(
        df, df["sess_asia"], df["utc_day"], min_bars=int(0.9 * 7 * 60 / tf_minutes)
    )
    df["asia_high"], df["asia_low"], df["asia_mid"] = asia["high"], asia["low"], asia["mid"]
    width = (df["asia_high"] - df["asia_low"]).groupby(df["utc_day"]).last()
    width_med = width.shift(1).rolling(20, min_periods=10).median()
    df["asia_width_ratio"] = (df["asia_high"] - df["asia_low"]) / df["utc_day"].map(width_med)

    # Prior day levels + day open
    df = pd.concat([df, prior_day_levels(df, tf_minutes)], axis=1)
    df["day_ret"] = df["close"] / df["day_open"] - 1

    # Anchored VWAPs: UTC day, weekend (Sat 00:00 UTC anchor)
    dvwap = anchored_vwap(df, df["utc_day"])
    df["vwap_day"], df["vwap_day_sd"] = dvwap["vwap"], dvwap["vwap_sd"]
    wk_key = (df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")).dt.floor("D")
    wvwap = anchored_vwap(df, wk_key, mask=df["is_weekend"])
    df["vwap_wknd"], df["vwap_wknd_sd"] = wvwap["vwap"], wvwap["vwap_sd"]
    df["wknd_bars"] = df["is_weekend"].astype(int).groupby(wk_key).cumsum()

    # NY opening range (09:30-10:00 ET)
    orng = opening_range(df, NEW_YORK, "09:30", "10:00", "16:00", tf_minutes)
    df["or_high"], df["or_low"], df["or_mid"] = orng["high"], orng["low"], orng["mid"]

    # 1h regime
    if df1h is not None and not df1h.empty:
        reg = regime_1h(df1h, p.atr_len)
        df = merge_informative_pair(df, reg, f"{tf_minutes}m", "1h", ffill=True)
    else:
        for col in ("er_1h", "er_pct_1h", "rv_1h", "rv_pct_1h", "atr_1h"):
            df[col] = np.nan

    # Funding crowding
    if funding is not None and not funding.empty:
        df = merge_asof_backward(df, funding_features(funding))
    else:
        df["funding_rate"] = np.nan
        df["funding_pct"] = np.nan

    # BTC context (1h move z-score on 5m bars)
    if btc is not None and not btc.empty:
        b = btc[["date", "close", "high", "low", "open"]].copy()
        b_atr = atr(b, p.atr_len) / b["close"]
        n = max(1, 60 // tf_minutes)
        b["btc_z"] = (b["close"] / b["close"].shift(n) - 1) / (b_atr * np.sqrt(n))
        df = df.merge(b[["date", "btc_z"]], on="date", how="left")
    else:
        df["btc_z"] = np.nan

    # --- second-wave features ------------------------------------------------
    df["ret_z"] = return_z(df["close"])
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / rng
    df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / rng
    eq = equal_levels(df, df["atr"], p.eq_swing_k, p.eq_lookback, p.eq_tol_atr)
    df = pd.concat([df, eq], axis=1)
    df = pd.concat([df, cme_weekend_anchor(df)], axis=1)

    if df1h is not None and not df1h.empty:
        df = pd.concat([df, prior_week_levels(df, df1h)], axis=1)
        df = pd.concat([df, hourly_seasonality(df, df1h)], axis=1)
    else:
        for col in ("pwh", "pwl", "pw_mid", "season_mean", "season_t"):
            df[col] = np.nan
        df["week_key"] = df["utc_day"]

    if df4h is not None and not df4h.empty:
        df = merge_informative_pair(
            df, htf_trend(df4h, atr_len=p.atr_len), f"{tf_minutes}m", "4h", ffill=True
        )
    else:
        for col in ("dc_high_4h", "dc_low_4h", "ema_4h", "atr_4h", "er_pct_4h"):
            df[col] = np.nan

    df["tf_min"] = tf_minutes
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _epoch(ts: pd.Series) -> pd.Series:
    return (ts - pd.Timestamp("1970-01-01", tz=UTC)) // pd.Timedelta(seconds=1)


def _next_utc_midnight(df: pd.DataFrame) -> pd.Series:
    return df["date"].dt.floor("D") + pd.Timedelta(days=1)


def _ny_time_today(df: pd.DataFrame, hhmm: str) -> pd.Series:
    h, m = (int(x) for x in hhmm.split(":"))
    local = df["ny_day"] + pd.Timedelta(hours=h, minutes=m)
    return local.dt.tz_localize(
        NEW_YORK, nonexistent="shift_forward", ambiguous="NaT"
    ).dt.tz_convert(UTC)


def _empty(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["long"] = False
    out["short"] = False
    out["stop_long"] = np.nan
    out["stop_short"] = np.nan
    out["target_long"] = np.nan
    out["target_short"] = np.nan
    out["deadline"] = np.nan
    out["trail"] = np.nan
    return out


def _risk_filter(
    df: pd.DataFrame, out: pd.DataFrame, p: Params, max_stop_pct: float | None = None
) -> pd.DataFrame:
    """Drop setups whose stop distance is outside [min_stop_pct, max_stop_pct]."""
    hi = p.max_stop_pct if max_stop_pct is None else max_stop_pct
    c = df["close"]
    dl = (c - out["stop_long"]) / c
    ds = (out["stop_short"] - c) / c
    out["long"] &= dl.between(p.min_stop_pct, hi)
    out["short"] &= ds.between(p.min_stop_pct, hi)
    return out


def _entry_time(df: pd.DataFrame) -> pd.Series:
    return df["date"] + pd.Timedelta(minutes=int(df["tf_min"].iloc[0]))


def _btc_ok(df: pd.DataFrame, p: Params, is_btc: bool) -> tuple[pd.Series, pd.Series]:
    if is_btc:
        t = pd.Series(True, index=df.index)
        return t, t
    z = df["btc_z"]
    long_ok = ~(z < -p.btc_z_block)
    short_ok = ~(z > p.btc_z_block)
    return long_ok, short_ok


def _vol_ok(df: pd.DataFrame, p: Params) -> pd.Series:
    return ~(df["rv_pct_1h"] > p.rv_pct_max)


def _mr_regime(df: pd.DataFrame, p: Params) -> pd.Series:
    return ~(df["er_pct_1h"] > p.mr_er_pct_max)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
def asia_sweep(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Judas swing: London / NY-AM sweep of the Asia range that closes back inside."""
    out = _empty(df)
    sw = detect_sweep(df, df["asia_high"], df["asia_low"], df["atr"], df["utc_day"])
    window = (df["sess_london"] | df["sess_ny_am"]) & ~df["is_weekend"]
    base = (
        window
        & (df["rvol_max3"] >= p.asia_rvol_min)
        & ~(df["asia_width_ratio"] > p.asia_width_ratio_max)
        & _mr_regime(df, p)
        & _vol_ok(df, p)
    )
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    c = df["close"]
    out["stop_long"] = sw["sweep_ext_low"] - p.stop_buffer_atr * df["atr"]
    out["stop_short"] = sw["sweep_ext_high"] + p.stop_buffer_atr * df["atr"]
    out["target_long"] = df["asia_mid"]
    out["target_short"] = df["asia_mid"]
    rr_long = (out["target_long"] - c) / (c - out["stop_long"])
    rr_short = (c - out["target_short"]) / (out["stop_short"] - c)
    out["long"] = base & sw["sweep_long"] & long_ok & (rr_long >= p.asia_min_rr)
    out["short"] = base & sw["sweep_short"] & short_ok & (rr_short >= p.asia_min_rr)
    entry = df["date"] + pd.Timedelta(minutes=int(df["tf_min"].iloc[0]))
    deadline = np.minimum(
        _epoch(entry + pd.Timedelta(hours=p.asia_max_hold_h)), _epoch(_next_utc_midnight(df))
    )
    out["deadline"] = deadline
    return _risk_filter(df, out, p)


def pd_sweep(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Prior-day high/low sweep + reclaim, with abnormal volume and crowded funding."""
    out = _empty(df)
    sw = detect_sweep(df, df["pdh"], df["pdl"], df["atr"], df["utc_day"])
    base = (
        ~df["is_weekend"] & (df["rvol_max3"] >= p.pd_rvol_min) & _mr_regime(df, p) & _vol_ok(df, p)
    )
    fp = df["funding_pct"]
    missing = fp.isna() & (not p.pd_require_funding)
    crowd_long = missing | (fp >= p.pd_funding_crowd)  # longs crowded -> fade PDH sweep
    crowd_short = missing | (fp <= 1 - p.pd_funding_crowd)
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    c = df["close"]
    out["stop_long"] = sw["sweep_ext_low"] - p.stop_buffer_atr * df["atr"]
    out["stop_short"] = sw["sweep_ext_high"] + p.stop_buffer_atr * df["atr"]
    out["target_long"] = c + p.pd_rr * (c - out["stop_long"])
    out["target_short"] = c - p.pd_rr * (out["stop_short"] - c)
    out["long"] = base & sw["sweep_long"] & crowd_short & long_ok
    out["short"] = base & sw["sweep_short"] & crowd_long & short_ok
    entry = df["date"] + pd.Timedelta(minutes=int(df["tf_min"].iloc[0]))
    out["deadline"] = np.minimum(
        _epoch(entry + pd.Timedelta(hours=p.pd_max_hold_h)), _epoch(_next_utc_midnight(df))
    )
    return _risk_filter(df, out, p)


def ny_orb(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """NY opening-range breakout aligned with the day's direction, in a trending regime."""
    out = _empty(df)
    bo = first_breakout(df["close"], df["or_high"], df["or_low"], df["ny_day"])
    window = in_window(df["date"], NEW_YORK, "10:00", p.orb_entry_until) & ~df["is_weekend"]
    width_ok = ~((df["or_high"] - df["or_low"]) > p.orb_width_atr1h_max * df["atr_1h"])
    base = (
        window
        & (df["rvol"] >= p.orb_rvol_min)
        & ~(df["er_pct_1h"] < p.orb_er_pct_min)
        & width_ok
        & _vol_ok(df, p)
    )
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    c = df["close"]
    out["stop_long"] = df["or_mid"]
    out["stop_short"] = df["or_mid"]
    out["target_long"] = c + p.orb_rr * (c - out["stop_long"])
    out["target_short"] = c - p.orb_rr * (out["stop_short"] - c)
    out["long"] = base & bo["breakout_up"] & (df["day_ret"] > 0) & long_ok
    out["short"] = base & bo["breakout_down"] & (df["day_ret"] < 0) & short_ok
    out["deadline"] = _epoch(_ny_time_today(df, "16:00"))
    return _risk_filter(df, out, p)


def late_mom(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Intraday time-series momentum into the UTC daily close (research module)."""
    out = _empty(df)
    tf = int(df["tf_min"].iloc[0])
    at_time = in_window(
        df["date"],
        NEW_YORK,
        p.lm_entry_time,
        (pd.Timestamp(f"2000-01-01 {p.lm_entry_time}") + pd.Timedelta(minutes=tf)).strftime(
            "%H:%M"
        ),
    )
    hours = (df["date"] - df["date"].dt.floor("D")) / pd.Timedelta(hours=1)
    sigma = df["rv_1h"] * np.sqrt(hours.clip(lower=1))
    z = np.log1p(df["day_ret"]) / sigma
    base = at_time & ~df["is_weekend"] & _vol_ok(df, p)
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    c = df["close"]
    out["stop_long"] = c - p.lm_stop_atr1h * df["atr_1h"]
    out["stop_short"] = c + p.lm_stop_atr1h * df["atr_1h"]
    out["long"] = base & (z >= p.lm_z_min) & long_ok
    out["short"] = base & (z <= -p.lm_z_min) & short_ok
    out["deadline"] = _epoch(_next_utc_midnight(df) - pd.Timedelta(minutes=tf))
    return _risk_filter(df, out, p)


def wknd_vwap(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Weekend reversion to the weekend-anchored VWAP (research module)."""
    out = _empty(df)
    z = (df["close"] - df["vwap_wknd"]) / df["vwap_wknd_sd"]
    z_prev = z.shift(1)
    base = (
        df["is_weekend"] & (df["wknd_bars"] >= p.wk_min_bars) & _mr_regime(df, p) & _vol_ok(df, p)
    )
    c = df["close"]
    out["stop_long"] = c - p.wk_stop_atr1h * df["atr_1h"]
    out["stop_short"] = c + p.wk_stop_atr1h * df["atr_1h"]
    out["target_long"] = df["vwap_wknd"]
    out["target_short"] = df["vwap_wknd"]
    # enter on the first bar that stretches beyond the band (not every bar out there)
    out["long"] = base & (z <= -p.wk_z_entry) & ~(z_prev <= -p.wk_z_entry)
    out["short"] = base & (z >= p.wk_z_entry) & ~(z_prev >= p.wk_z_entry)
    entry = df["date"] + pd.Timedelta(minutes=int(df["tf_min"].iloc[0]))
    out["deadline"] = _epoch(entry + pd.Timedelta(hours=p.wk_max_hold_h))
    return _risk_filter(df, out, p)


def fund_window(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Fade the crowded side into a funding settlement (research module)."""
    out = _empty(df)
    tf = int(df["tf_min"].iloc[0])
    mod = df["date"].dt.hour * 60 + df["date"].dt.minute
    # signal candle closes exactly 1h before 00/08/16 UTC
    at_time = mod.isin([((h * 60) - 60 - tf) % 1440 for h in (0, 8, 16)])
    fp = df["funding_pct"]
    base = at_time & _vol_ok(df, p)
    c = df["close"]
    out["stop_long"] = c - p.fw_stop_atr1h * df["atr_1h"]
    out["stop_short"] = c + p.fw_stop_atr1h * df["atr_1h"]
    out["long"] = base & (fp <= 1 - p.fw_pct_extreme)
    out["short"] = base & (fp >= p.fw_pct_extreme)
    out["deadline"] = _epoch(df["date"] + pd.Timedelta(minutes=tf + 75))
    return _risk_filter(df, out, p)


# ---------------------------------------------------------------------------
# Second wave: liquidity family
# ---------------------------------------------------------------------------
def pw_sweep(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Prior-WEEK high/low sweep + reclaim: the biggest obvious stop pools of the week."""
    out = _empty(df)
    sw = detect_sweep(df, df["pwh"], df["pwl"], df["atr"], df["week_key"])
    base = (df["rvol_max3"] >= p.pw_rvol_min) & _mr_regime(df, p) & _vol_ok(df, p)
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    c = df["close"]
    out["stop_long"] = sw["sweep_ext_low"] - p.stop_buffer_atr * df["atr"]
    out["stop_short"] = sw["sweep_ext_high"] + p.stop_buffer_atr * df["atr"]
    risk_l = c - out["stop_long"]
    risk_s = out["stop_short"] - c
    # target: prior-week midpoint, but never further than pw_rr_cap R
    out["target_long"] = np.minimum(df["pw_mid"], c + p.pw_rr_cap * risk_l)
    out["target_short"] = np.maximum(df["pw_mid"], c - p.pw_rr_cap * risk_s)
    rr_l = (out["target_long"] - c) / risk_l
    rr_s = (c - out["target_short"]) / risk_s
    out["long"] = base & sw["sweep_long"] & long_ok & (rr_l >= p.pw_min_rr)
    out["short"] = base & sw["sweep_short"] & short_ok & (rr_s >= p.pw_min_rr)
    out["deadline"] = _epoch(_entry_time(df) + pd.Timedelta(hours=p.pw_max_hold_h))
    return _risk_filter(df, out, p)


def eq_sweep(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Sweep + reclaim of an equal-highs / equal-lows cluster (resting stop pool)."""
    out = _empty(df)
    nan = pd.Series(np.nan, index=df.index)
    key_h = df["eqh_id"].fillna(-1)
    key_l = df["eql_id"].fillna(-1)
    sh = detect_sweep(df, df["eqh"], nan, df["atr"], key_h)
    sl = detect_sweep(df, nan, df["eql"], df["atr"], key_l)
    base = (df["rvol_max3"] >= p.eq_rvol_min) & _mr_regime(df, p) & _vol_ok(df, p)
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    c = df["close"]
    out["stop_long"] = sl["sweep_ext_low"] - p.stop_buffer_atr * df["atr"]
    out["stop_short"] = sh["sweep_ext_high"] + p.stop_buffer_atr * df["atr"]
    out["target_long"] = c + p.eq_rr * (c - out["stop_long"])
    out["target_short"] = c - p.eq_rr * (out["stop_short"] - c)
    out["long"] = base & sl["sweep_long"] & long_ok
    out["short"] = base & sh["sweep_short"] & short_ok
    out["deadline"] = _epoch(_entry_time(df) + pd.Timedelta(hours=p.eq_max_hold_h))
    return _risk_filter(df, out, p)


def liq_cascade(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Fade a liquidation-cascade candle: extreme return, extreme volume, long rejection wick."""
    out = _empty(df)
    base = (df["rvol"] >= p.lc_rvol_min) & ~(df["er_pct_1h"] > p.lc_er_pct_max)
    c = df["close"]
    out["stop_long"] = df["low"] - p.lc_stop_atr1h * df["atr_1h"]
    out["stop_short"] = df["high"] + p.lc_stop_atr1h * df["atr_1h"]
    rr_l = c + p.lc_rr * (c - out["stop_long"])
    rr_s = c - p.lc_rr * (out["stop_short"] - c)
    vw = df["vwap_day"]
    out["target_long"] = np.where(vw > c, np.minimum(vw, rr_l), rr_l)
    out["target_short"] = np.where(vw < c, np.maximum(vw, rr_s), rr_s)
    out["long"] = base & (df["ret_z"] <= -p.lc_z) & (df["lower_wick"] >= p.lc_wick_min)
    out["short"] = base & (df["ret_z"] >= p.lc_z) & (df["upper_wick"] >= p.lc_wick_min)
    out["deadline"] = _epoch(_entry_time(df) + pd.Timedelta(hours=p.lc_max_hold_h))
    return _risk_filter(df, out, p)


# ---------------------------------------------------------------------------
# Second wave: volatility / session timing
# ---------------------------------------------------------------------------
def asia_squeeze(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Compressed Asia range -> first decisive close outside it in London / NY-AM (momentum)."""
    out = _empty(df)
    bo = first_breakout(df["close"], df["asia_high"], df["asia_low"], df["utc_day"])
    window = (df["sess_london"] | df["sess_ny_am"]) & ~df["is_weekend"]
    base = (
        window
        & (df["asia_width_ratio"] <= p.sq_width_ratio_max)
        & (df["rv_pct_1h"] <= p.sq_rv_pct_max)
        & (df["rvol"] >= p.sq_rvol_min)
    )
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    c = df["close"]
    out["stop_long"] = df["asia_mid"]
    out["stop_short"] = df["asia_mid"]
    out["target_long"] = c + p.sq_rr * (c - out["stop_long"])
    out["target_short"] = c - p.sq_rr * (out["stop_short"] - c)
    out["long"] = base & bo["breakout_up"] & long_ok
    out["short"] = base & bo["breakout_down"] & short_ok
    out["deadline"] = _epoch(_ny_time_today(df, "16:00"))
    return _risk_filter(df, out, p)


def tod_season(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """Walk-forward hour-of-day seasonality (weekday / weekend buckets), one-hour holds."""
    out = _empty(df)
    tf = int(df["tf_min"].iloc[0])
    t = df["season_t"]
    base = t.notna() & _vol_ok(df, p)
    c = df["close"]
    out["stop_long"] = c - p.ts_stop_atr1h * df["atr_1h"]
    out["stop_short"] = c + p.ts_stop_atr1h * df["atr_1h"]
    out["long"] = base & (t >= p.ts_t_min)
    out["short"] = base & (t <= -p.ts_t_min)
    out["deadline"] = _epoch(df["date"] + pd.Timedelta(minutes=tf + 60))
    return _risk_filter(df, out, p)


def cme_gap(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """At the CME reopen, fade the weekend move back toward the Friday CME-close price."""
    out = _empty(df)
    atr_1d = df["atr_1h"] * np.sqrt(24)
    gap = df["close"] - df["cme_fri_close"]
    base = df["cme_reopen"] & (gap.abs() >= p.cg_gap_atr1d * atr_1d)
    c = df["close"]
    out["stop_long"] = c - p.cg_stop_atr1h * df["atr_1h"]
    out["stop_short"] = c + p.cg_stop_atr1h * df["atr_1h"]
    out["target_long"] = df["cme_fri_close"]
    out["target_short"] = df["cme_fri_close"]
    out["long"] = base & (gap < 0)
    out["short"] = base & (gap > 0)
    # hold until Tuesday 00:00 UTC (the reopen candle is on Sunday UTC)
    out["deadline"] = _epoch(df["date"].dt.floor("D") + pd.Timedelta(days=2))
    return _risk_filter(df, out, p)


# ---------------------------------------------------------------------------
# Second wave: trend with liquid-hours entry
# ---------------------------------------------------------------------------
def trend_4h(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """
    4h Donchian breakout in the direction of the 4h EMA200, entered only on a 5m close
    during London / NY-AM with real volume, and managed with a 3x 4h-ATR chandelier stop.
    """
    out = _empty(df)
    c = df["close"]
    window = (df["sess_london"] | df["sess_ny_am"]) & ~df["is_weekend"]
    trending = df["er_pct_4h"] >= p.tr_er_pct_min
    base = window & trending & (df["rvol"] >= p.tr_rvol_min) & _vol_ok(df, p)
    up = (c > df["dc_high_4h"]) & ~(c.shift(1) > df["dc_high_4h"].shift(1))
    dn = (c < df["dc_low_4h"]) & ~(c.shift(1) < df["dc_low_4h"].shift(1))
    long_ok, short_ok = _btc_ok(df, p, is_btc)
    trail = p.tr_trail_atr4h * df["atr_4h"]
    out["stop_long"] = c - trail
    out["stop_short"] = c + trail
    out["trail"] = trail
    out["long"] = base & up & (c > df["ema_4h"]) & long_ok
    out["short"] = base & dn & (c < df["ema_4h"]) & short_ok
    out["deadline"] = _epoch(_entry_time(df) + pd.Timedelta(days=p.tr_max_hold_d))
    return _risk_filter(df, out, p, max_stop_pct=p.tr_max_stop_pct)


MODULE_FUNCS = {
    "asia_sweep": asia_sweep,
    "pd_sweep": pd_sweep,
    "ny_orb": ny_orb,
    "late_mom": late_mom,
    "wknd_vwap": wknd_vwap,
    "fund_window": fund_window,
    "pw_sweep": pw_sweep,
    "eq_sweep": eq_sweep,
    "liq_cascade": liq_cascade,
    "asia_squeeze": asia_squeeze,
    "tod_season": tod_season,
    "cme_gap": cme_gap,
    "trend_4h": trend_4h,
}


def compute_signals(df: pd.DataFrame, p: Params, is_btc: bool = False) -> pd.DataFrame:
    """
    Run every enabled module and combine them into freqtrade columns:
    ``enter_long``, ``enter_short``, ``enter_tag``, ``sig_stop``, ``sig_target``,
    ``sig_deadline``, ``sig_trail`` (trailing distance in price, NaN = fixed stop).
    Earlier modules in ``ALL_MODULES`` win ties; a candle that has both a long and a
    short setup is skipped.
    """
    df = df.copy()
    df["enter_long"] = 0
    df["enter_short"] = 0
    df["enter_tag"] = None
    df["sig_stop"] = np.nan
    df["sig_target"] = np.nan
    df["sig_deadline"] = np.nan
    df["sig_trail"] = np.nan
    taken = pd.Series(False, index=df.index)
    for name in ALL_MODULES:
        if not p.enabled.get(name, False):
            continue
        m = MODULE_FUNCS[name](df, p, is_btc)
        conflict = m["long"] & m["short"]
        for side, flag in (("long", "enter_long"), ("short", "enter_short")):
            sel = m[side] & ~conflict & ~taken
            df.loc[sel, flag] = 1
            df.loc[sel, "enter_tag"] = name
            df.loc[sel, "sig_stop"] = m.loc[sel, f"stop_{side}"]
            df.loc[sel, "sig_target"] = m.loc[sel, f"target_{side}"]
            df.loc[sel, "sig_deadline"] = m.loc[sel, "deadline"]
            df.loc[sel, "sig_trail"] = m.loc[sel, "trail"]
            taken |= sel
    return df
