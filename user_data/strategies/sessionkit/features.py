"""
Look-ahead-safe session and liquidity features.

Conventions
-----------
* ``df`` is a freqtrade OHLCV frame: columns ``date`` (tz-aware UTC, candle OPEN time),
  ``open``, ``high``, ``low``, ``close``, ``volume``, sorted ascending.
* A signal computed on row ``t`` may only use information available at the CLOSE of
  candle ``t``. Every function here is written so that computing it on ``df.iloc[:k]``
  gives exactly the same values for rows ``< k`` as computing it on the full frame.
  ``user_data/tests/test_sessionkit.py`` checks that invariant.
* Session windows are defined in the exchange-relevant local time zone (``zoneinfo``),
  so they follow daylight-saving switches automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


UTC = "UTC"
NEW_YORK = "America/New_York"
LONDON = "Europe/London"
CHICAGO = "America/Chicago"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def infer_tf_minutes(dates: pd.Series) -> int:
    """Return the candle size in minutes (median spacing of ``dates``)."""
    diffs = dates.diff().dropna()
    if diffs.empty:
        raise ValueError("Need at least two candles to infer the timeframe")
    return int(diffs.median() / pd.Timedelta(minutes=1))


def _hhmm(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def local_minute_of_day(dates: pd.Series, tz: str) -> pd.Series:
    local = dates.dt.tz_convert(tz)
    return local.dt.hour * 60 + local.dt.minute


def local_day(dates: pd.Series, tz: str) -> pd.Series:
    """Calendar day in ``tz`` as a naive Timestamp (usable as a groupby key)."""
    return dates.dt.tz_convert(tz).dt.tz_localize(None).dt.floor("D")


def in_window(dates: pd.Series, tz: str, start: str, end: str) -> pd.Series:
    """True when the candle OPEN time is within ``[start, end)`` local time."""
    mod = local_minute_of_day(dates, tz)
    s, e = _hhmm(start), _hhmm(end)
    if s <= e:
        return (mod >= s) & (mod < e)
    return (mod >= s) | (mod < e)  # window crossing midnight


def session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Boolean session columns (by candle open time).

    * ``sess_asia``      00:00-07:00 UTC   (Tokyo / Hong Kong / Singapore)
    * ``sess_london``    08:00-11:00 London (London open "kill zone")
    * ``sess_ny_am``     08:30-11:30 New York (US data releases + equity open)
    * ``sess_ny_rth``    09:30-16:00 New York (US cash equity hours)
    * ``is_weekend``     Saturday / Sunday UTC (CME closed, thin books)
    """
    d = df["date"]
    out = pd.DataFrame(index=df.index)
    out["sess_asia"] = in_window(d, UTC, "00:00", "07:00")
    out["sess_london"] = in_window(d, LONDON, "08:00", "11:00")
    out["sess_ny_am"] = in_window(d, NEW_YORK, "08:30", "11:30")
    out["sess_ny_rth"] = in_window(d, NEW_YORK, "09:30", "16:00")
    out["is_weekend"] = d.dt.dayofweek >= 5
    return out


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------
def session_range(
    df: pd.DataFrame, mask: pd.Series, day_key: pd.Series, min_bars: int
) -> pd.DataFrame:
    """
    High / low / mid of the candles in ``mask`` for each ``day_key`` group.

    Values are only populated on rows AFTER the session has finished (same group),
    so a row never sees a session extreme that is still being formed. Sessions with
    fewer than ``min_bars`` candles (data gaps, partial first day) yield NaN.
    """
    mask = mask.astype(bool)
    hi = df["high"].where(mask).groupby(day_key).transform("max")
    lo = df["low"].where(mask).groupby(day_key).transform("min")
    n_bars = mask.astype(int).groupby(day_key).transform("sum")
    seen = mask.astype(int).groupby(day_key).cumsum() > 0
    after = seen & ~mask & (n_bars >= min_bars)
    out = pd.DataFrame(index=df.index)
    out["high"] = hi.where(after)
    out["low"] = lo.where(after)
    out["mid"] = (out["high"] + out["low"]) / 2
    return out


def prior_day_levels(df: pd.DataFrame, tf_minutes: int, coverage: float = 0.95) -> pd.DataFrame:
    """
    Previous UTC day's high, low and close (PDH / PDL / PDC) plus today's open.

    The previous day must be the immediately preceding calendar day and contain at least
    ``coverage`` of its candles, otherwise NaN.
    """
    day = df["date"].dt.tz_convert(UTC).dt.tz_localize(None).dt.floor("D")
    agg = df.groupby(day).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"), n=("close", "size")
    )
    full = agg["n"] >= coverage * 1440 / tf_minutes
    agg.loc[~full, ["high", "low", "close"]] = np.nan
    prev = agg[["high", "low", "close"]].shift(1)
    contiguous = agg.index.to_series().diff() == pd.Timedelta(days=1)
    prev = prev.where(contiguous, np.nan)

    out = pd.DataFrame(index=df.index)
    out["pdh"] = day.map(prev["high"]).astype(float)
    out["pdl"] = day.map(prev["low"]).astype(float)
    out["pdc"] = day.map(prev["close"]).astype(float)

    first_is_midnight = df["date"].groupby(day).transform("first").dt.hour.eq(0) & df[
        "date"
    ].groupby(day).transform("first").dt.minute.eq(0)
    out["day_open"] = df["open"].groupby(day).transform("first").where(first_is_midnight)
    return out


def anchored_vwap(
    df: pd.DataFrame, anchor_key: pd.Series, mask: pd.Series | None = None
) -> pd.DataFrame:
    """
    VWAP and volume-weighted standard deviation, re-anchored at every change of
    ``anchor_key``. Rows where ``mask`` is False are excluded (NaN).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].astype(float)
    if mask is not None:
        vol = vol.where(mask, 0.0)
    cum_v = vol.groupby(anchor_key).cumsum()
    cum_pv = (tp * vol).groupby(anchor_key).cumsum()
    cum_p2v = (tp * tp * vol).groupby(anchor_key).cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = cum_pv / cum_v
        var = (cum_p2v / cum_v - vwap * vwap).clip(lower=0)
    out = pd.DataFrame(index=df.index)
    out["vwap"] = vwap.replace([np.inf, -np.inf], np.nan)
    out["vwap_sd"] = np.sqrt(var).replace([np.inf, -np.inf], np.nan)
    if mask is not None:
        out = out.where(mask)
    return out


def opening_range(
    df: pd.DataFrame,
    tz: str = NEW_YORK,
    start: str = "09:30",
    end: str = "10:00",
    valid_until: str = "16:00",
    tf_minutes: int = 5,
) -> pd.DataFrame:
    """Opening range (e.g. first 30 min of US cash hours), valid from ``end`` to ``valid_until``."""
    mask = in_window(df["date"], tz, start, end)
    key = local_day(df["date"], tz)
    n_expected = (_hhmm(end) - _hhmm(start)) // tf_minutes
    rng = session_range(df, mask, key, min_bars=n_expected)
    live = in_window(df["date"], tz, end, valid_until)
    return rng.where(live)


# ---------------------------------------------------------------------------
# Volume, volatility, regime
# ---------------------------------------------------------------------------
def rel_volume_tod(df: pd.DataFrame, lookback_days: int = 10, min_days: int = 5) -> pd.Series:
    """
    Volume relative to the median volume of the SAME time-of-day slot over the previous
    ``lookback_days`` days (current day excluded). Removes the intraday volume seasonality,
    so a value of 2 means "twice as busy as this time of day usually is".
    """
    slot = df["date"].dt.hour * 60 + df["date"].dt.minute
    vol = df["volume"].astype(float)
    med = vol.groupby(slot).transform(
        lambda s: s.shift(1).rolling(lookback_days, min_periods=min_days).median()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rvol = vol / med
    return rvol.replace([np.inf, -np.inf], np.nan)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def efficiency_ratio(close: pd.Series, n: int = 24) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / path length. ~1 = trending, ~0 = choppy."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        er = net / path
    return er.replace([np.inf, -np.inf], np.nan)


def realized_vol(close: pd.Series, n: int = 24) -> pd.Series:
    return np.log(close).diff().rolling(n, min_periods=n).std()


def rolling_pct_rank(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile (0..1] of the current value within the trailing window (inclusive)."""
    return s.rolling(window, min_periods=min_periods or window // 2).rank(pct=True)


# ---------------------------------------------------------------------------
# Liquidity sweeps
# ---------------------------------------------------------------------------
def detect_sweep(
    df: pd.DataFrame,
    level_high: pd.Series,
    level_low: pd.Series,
    atr_: pd.Series,
    day_key: pd.Series,
    min_pen_atr: float = 0.05,
    max_pen_atr: float = 1.5,
    max_reclaim_bars: int = 3,
    min_wick_frac: float = 0.4,
) -> pd.DataFrame:
    """
    Detect the FIRST sweep-and-reclaim of a level per ``day_key``.

    Short setup (buy-side liquidity taken above ``level_high``):
      1. a candle trades above ``level_high + min_pen_atr * ATR`` (stops triggered),
      2. within ``max_reclaim_bars`` bars of that first breach, a candle CLOSES back
         below the level, and this is the first such close since the breach,
      3. the excursion beyond the level stayed below ``max_pen_atr * ATR``
         (a sweep, not a genuine breakout),
      4. rejection quality: if the reclaim happens on the breach candle itself, its
         upper wick must be >= ``min_wick_frac`` of its range; otherwise the reclaim
         candle must be bearish.

    The long setup mirrors this below ``level_low``.

    Returns columns ``sweep_short``, ``sweep_long``, ``sweep_ext_high`` (highest high since
    breach, i.e. the structural stop for shorts) and ``sweep_ext_low``.
    """
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - lo).replace(0, np.nan)
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - lo) / rng
    pos = df.groupby(day_key).cumcount()

    def _side(level: pd.Series, short: bool) -> tuple[pd.Series, pd.Series]:
        valid = level.notna() & atr_.notna()
        if short:
            breach = valid & (h > level + min_pen_atr * atr_)
            inside = valid & (c < level)
            wick_ok = upper_wick >= min_wick_frac
            candle_ok = c < o
            extreme_src = h
        else:
            breach = valid & (lo < level - min_pen_atr * atr_)
            inside = valid & (c > level)
            wick_ok = lower_wick >= min_wick_frac
            candle_ok = c > o
            extreme_src = lo

        breached = breach.astype(int).groupby(day_key).cumsum() > 0
        first_pos = pos.where(breach).groupby(day_key).cummin()
        first_pos = first_pos.groupby(day_key).ffill()
        bars_since = pos - first_pos

        n_inside_after = (inside & breached).astype(int).groupby(day_key).cumsum()
        if short:
            extreme = extreme_src.where(breached).groupby(day_key).cummax()
            excursion = extreme - level
        else:
            extreme = extreme_src.where(breached).groupby(day_key).cummin()
            excursion = level - extreme

        quality = ((bars_since == 0) & wick_ok) | ((bars_since > 0) & candle_ok)
        signal = (
            breached
            & inside
            & (n_inside_after == 1)
            & (bars_since <= max_reclaim_bars)
            & (excursion <= max_pen_atr * atr_)
            & quality
        )
        return signal.fillna(False).astype(bool), extreme

    out = pd.DataFrame(index=df.index)
    out["sweep_short"], out["sweep_ext_high"] = _side(level_high, short=True)
    out["sweep_long"], out["sweep_ext_low"] = _side(level_low, short=False)
    return out


def first_breakout(
    close: pd.Series, level_high: pd.Series, level_low: pd.Series, day_key: pd.Series
) -> pd.DataFrame:
    """First close above ``level_high`` / below ``level_low`` per ``day_key``."""
    up = (close > level_high) & level_high.notna()
    dn = (close < level_low) & level_low.notna()
    out = pd.DataFrame(index=close.index)
    out["breakout_up"] = up & (up.astype(int).groupby(day_key).cumsum() == 1)
    out["breakout_down"] = dn & (dn.astype(int).groupby(day_key).cumsum() == 1)
    return out


# ---------------------------------------------------------------------------
# Funding / crowding
# ---------------------------------------------------------------------------
def funding_features(funding: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Funding-rate crowding features from a freqtrade ``funding_rate`` frame
    (rate in the ``open`` column, one row per settlement).

    ``funding_pct`` is the percentile of the latest settled rate versus the last
    ``window`` settlements (90 x 8h = 30 days). High => longs crowded.
    """
    f = funding[["date", "open"]].rename(columns={"open": "funding_rate"}).copy()
    f = f.sort_values("date").reset_index(drop=True)
    f["funding_pct"] = rolling_pct_rank(f["funding_rate"], window, min_periods=window // 3)
    return f


def merge_asof_backward(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest ``other`` row whose ``date`` <= the candle's date."""
    left = df.reset_index().rename(columns={"index": "_idx"})
    merged = pd.merge_asof(
        left.sort_values("date"), other.sort_values("date"), on="date", direction="backward"
    )
    return merged.set_index("_idx").sort_index().rename_axis(None)


def event_blackout(
    dates: pd.Series, events: list[str], before_min: int = 30, after_min: int = 60
) -> pd.Series:
    """True for candles within [event - before, event + after] of any scheduled event (UTC)."""
    block = pd.Series(False, index=dates.index)
    for ev in events:
        t = pd.Timestamp(ev)
        t = t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)
        block |= (dates >= t - pd.Timedelta(minutes=before_min)) & (
            dates <= t + pd.Timedelta(minutes=after_min)
        )
    return block


# ---------------------------------------------------------------------------
# Higher-timeframe levels, swing structure, shocks, seasonality
# ---------------------------------------------------------------------------
def week_key(dates: pd.Series) -> pd.Series:
    """Monday 00:00 UTC of the candle's ISO week (naive Timestamp, a groupby key)."""
    d = dates.dt.tz_convert(UTC).dt.tz_localize(None)
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.floor("D")


def prior_week_levels(df: pd.DataFrame, df1h: pd.DataFrame, coverage: float = 0.95) -> pd.DataFrame:
    """
    Previous UTC week's (Mon-Sun) high / low / mid for every row of ``df``.

    Computed from the 1h frame (which has a much longer history than the 5m startup
    window) and mapped by week key. Only completed, contiguous, >= ``coverage`` covered
    weeks are used.
    """
    wk1h = week_key(df1h["date"])
    agg = df1h.groupby(wk1h).agg(high=("high", "max"), low=("low", "min"), n=("close", "size"))
    full = agg["n"] >= coverage * 7 * 24
    agg.loc[~full, ["high", "low"]] = np.nan
    prev = agg[["high", "low"]].shift(1)
    contiguous = agg.index.to_series().diff() == pd.Timedelta(days=7)
    prev = prev.where(contiguous, np.nan)
    wk = week_key(df["date"])
    out = pd.DataFrame(index=df.index)
    out["pwh"] = wk.map(prev["high"]).astype(float)
    out["pwl"] = wk.map(prev["low"]).astype(float)
    out["pw_mid"] = (out["pwh"] + out["pwl"]) / 2
    out["week_key"] = wk
    return out


def swing_points(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Fractal swing highs / lows, reported on the bar where they become CONFIRMED
    (``k`` bars after the pivot), so they carry no look-ahead.

    Columns: ``swing_high`` / ``swing_low`` (pivot price or NaN) on the confirmation bar.
    """
    w = 2 * k + 1
    h, lo = df["high"], df["low"]
    piv_h = h.shift(k)
    piv_l = lo.shift(k)
    is_h = piv_h.notna() & (piv_h == h.rolling(w, min_periods=w).max())
    is_l = piv_l.notna() & (piv_l == lo.rolling(w, min_periods=w).min())
    out = pd.DataFrame(index=df.index)
    out["swing_high"] = piv_h.where(is_h)
    out["swing_low"] = piv_l.where(is_l)
    return out


def equal_levels(
    df: pd.DataFrame,
    atr_: pd.Series,
    k: int = 3,
    lookback: int = 288,
    tol_atr: float = 0.1,
    valid_bars: int = 288,
) -> pd.DataFrame:
    """
    "Equal highs / lows" liquidity pools: two confirmed swing highs (lows) within
    ``tol_atr`` x ATR of each other, at most ``lookback`` bars apart, with price never
    trading through the pair in between. Stops of everyone who sold the double top
    sit just above it.

    Returns the active pool level (``eqh`` / ``eql``), valid for ``valid_bars`` after
    confirmation, and an id per pool (``eqh_id`` / ``eql_id``) to key sweep detection.
    """
    sw = swing_points(df, k)
    n = len(df)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    a = atr_.to_numpy(float)

    def _scan(vals: np.ndarray, is_high: bool) -> tuple[np.ndarray, np.ndarray]:
        level = np.full(n, np.nan)
        lid = np.full(n, np.nan)
        events = np.flatnonzero(~np.isnan(vals))
        recent: list[tuple[int, float]] = []
        counter = 0
        for i in events:
            v = vals[i]
            piv_i = i - k
            recent = [(j, vj) for j, vj in recent if i - j <= lookback]
            if np.isfinite(a[i]):
                for j, vj in reversed(recent):
                    if abs(v - vj) > tol_atr * a[i]:
                        continue
                    lvl = max(v, vj) if is_high else min(v, vj)
                    between = slice(j - k + 1, piv_i)
                    if is_high:
                        untouched = between.start >= between.stop or h[between].max() <= lvl
                    else:
                        untouched = between.start >= between.stop or lo[between].min() >= lvl
                    if untouched:
                        counter += 1
                        level[i] = lvl
                        lid[i] = counter
                        break
            recent.append((i, v))
        lv = pd.Series(level).ffill(limit=valid_bars).to_numpy()
        ids = pd.Series(lid).ffill(limit=valid_bars).to_numpy()
        return lv, ids

    out = pd.DataFrame(index=df.index)
    out["eqh"], out["eqh_id"] = _scan(sw["swing_high"].to_numpy(float), True)
    out["eql"], out["eql_id"] = _scan(sw["swing_low"].to_numpy(float), False)
    return out


def return_z(close: pd.Series, window: int = 288) -> pd.Series:
    """Log return of the candle in units of the trailing (past-only) return std."""
    r = np.log(close).diff()
    sd = r.rolling(window, min_periods=window // 2).std().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = r / sd
    return z.replace([np.inf, -np.inf], np.nan)


def hourly_seasonality(
    df: pd.DataFrame, df1h: pd.DataFrame, window: str = "120D", min_obs: int = 20
) -> pd.DataFrame:
    """
    Walk-forward hour-of-day seasonality, split weekday / weekend (48 buckets).

    For each candle that closes at the top of an hour, returns the trailing ``window``
    mean and t-stat of the 1h log return (open -> close) of the bucket of the hour that is
    about to START. Only 1h candles that have already CLOSED are used (as-of merge on the
    1h close time), so the value is known when the signal candle closes.
    """
    d = pd.DataFrame({"date": df1h["date"], "r": np.log(df1h["close"] / df1h["open"])})
    d["bucket"] = d["date"].dt.hour + 24 * (d["date"].dt.dayofweek >= 5)
    d = d.set_index("date")
    roll = d.groupby("bucket")["r"].rolling(window)
    stats = pd.DataFrame({"m": roll.mean(), "sd": roll.std(), "n": roll.count()}).reset_index()
    stats.columns = ["bucket", "date", "m", "sd", "n"]
    stats["t"] = stats["m"] / (stats["sd"] / np.sqrt(stats["n"]))
    stats.loc[stats["n"] < min_obs, ["m", "t"]] = np.nan
    stats["avail"] = stats["date"] + pd.Timedelta(hours=1)  # known once the 1h candle closed
    stats = stats.sort_values("avail")[["avail", "bucket", "m", "t"]]

    tf = pd.Timedelta(minutes=infer_tf_minutes(df["date"]) if len(df) > 1 else 5)
    nxt = df["date"] + tf  # close of the current candle = start of the next hour
    left = pd.DataFrame(
        {
            "_i": np.arange(len(df)),
            "avail": nxt,
            "bucket": nxt.dt.hour + 24 * (nxt.dt.dayofweek >= 5),
        }
    ).sort_values("avail")
    merged = pd.merge_asof(left, stats, on="avail", by="bucket", direction="backward")
    merged = merged.sort_values("_i")
    at_hour = (nxt.dt.minute == 0).to_numpy()
    out = pd.DataFrame(index=df.index)
    out["season_mean"] = np.where(at_hour, merged["m"].to_numpy(), np.nan)
    out["season_t"] = np.where(at_hour, merged["t"].to_numpy(), np.nan)
    return out


def htf_trend(
    df4h: pd.DataFrame, dc_len: int = 20, ema_len: int = 200, atr_len: int = 14
) -> pd.DataFrame:
    """4h trend features: Donchian channel, EMA, ATR, efficiency ratio percentile."""
    out = df4h[["date"]].copy()
    out["dc_high"] = df4h["high"].rolling(dc_len, min_periods=dc_len).max()
    out["dc_low"] = df4h["low"].rolling(dc_len, min_periods=dc_len).min()
    out["ema"] = df4h["close"].ewm(span=ema_len, adjust=False, min_periods=ema_len).mean()
    out["atr"] = atr(df4h, atr_len)
    er = efficiency_ratio(df4h["close"], dc_len)
    out["er_pct"] = rolling_pct_rank(er, 540, min_periods=180)
    return out


def cme_weekend_anchor(df: pd.DataFrame) -> pd.DataFrame:
    """
    CME bitcoin futures stop trading Friday 16:00 and reopen Sunday 17:00 (Chicago time).

    ``cme_fri_close``: close of the perp candle ending at the Friday 16:00 CT close,
    carried forward over the weekend. ``cme_reopen``: True on the candle that ends at the
    Sunday 17:00 CT reopen (signal candle -> entry exactly at the reopen).
    """
    local = df["date"].dt.tz_convert(CHICAGO)
    tf = int((df["date"].diff().median()) / pd.Timedelta(minutes=1)) if len(df) > 1 else 5
    end = local + pd.Timedelta(minutes=tf)
    fri = (end.dt.dayofweek == 4) & (end.dt.hour == 16) & (end.dt.minute == 0)
    sun = (end.dt.dayofweek == 6) & (end.dt.hour == 17) & (end.dt.minute == 0)
    fri_close = df["close"].where(fri).ffill(limit=int(60 * 60 / tf))
    out = pd.DataFrame(index=df.index)
    out["cme_fri_close"] = fri_close
    out["cme_reopen"] = sun
    return out
