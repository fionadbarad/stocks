#!/usr/bin/env python3
"""
momentum_screener.py
====================

A modular intraday momentum screener that ranks a universe of liquid US equities
by a composite "Profit Potential Score" built from relative volume, realized
volatility, trend location (VWAP), and momentum (MACD / RSI), then attaches an
explicit risk plan (entry / stop / target / R:R) to each candidate.

--------------------------------------------------------------------------------
READ THIS BEFORE YOU RISK MONEY
--------------------------------------------------------------------------------
This is a *screener*, not a strategy. It tells you which tickers currently satisfy
a set of technical conditions. It does not tell you those conditions are
profitable. Specifically:

1. NO EDGE IS CLAIMED OR DEMONSTRATED. The factor weights below are hand-chosen
   priors, not fitted or validated parameters. The composite score has never been
   backtested. Ranking by it is a heuristic for "this stock is moving with
   volume," which is a *description of the present*, not a *prediction of the
   future*.

2. THE SCREEN SELECTS FOR VARIANCE, NOT RETURN. High ATR% + high RVOL + gap-up is
   the textbook definition of a high-variance name. Screening for it reliably
   increases the dispersion of your outcomes. Whether it increases the *mean* is
   an entirely separate empirical question this code does not answer.

3. MULTIPLE COMPARISONS. Testing ~5 conditions across a few hundred tickers every
   day means some names pass by chance alone. A ticker appearing at the top of
   this dashboard is weak evidence of anything.

4. THE DATA IS NOT REAL-TIME. yfinance serves delayed Yahoo data (commonly
   ~15 minutes on many venues, and the most recent bar is often incomplete or
   zero-volume). Do not route orders off this. The DataProvider class exists so
   you can swap in a real feed (Polygon, Alpaca, Databento, IBKR) behind the same
   interface.

5. COSTS ARE NOT MODELLED. Spread, slippage, borrow, and commission on exactly
   the kind of fast, volatile, high-RVOL names this screen surfaces are the
   largest single reason retail momentum systems that look good on paper lose
   money in practice.

Validate on paper for a statistically meaningful sample before committing capital.
This file is engineering scaffolding and educational material, not financial advice.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python momentum_screener.py                      # default universe, hard filters
    python momentum_screener.py --relax              # rank everything, don't exclude
    python momentum_screener.py --top 5 --timeframe 15m
    python momentum_screener.py --tickers NVDA,AMD,TSLA
    python momentum_screener.py --selftest           # offline indicator unit tests
    python momentum_screener.py --json out.json      # machine-readable output

Requires: python>=3.9, pandas, numpy, yfinance
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dtime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# yfinance is imported lazily inside the provider so that --selftest runs offline.

# The macro/geopolitical layer is OPTIONAL. If macro_news.py is missing the
# screener still runs in pure-technical mode. A news feed being down must never
# take the price screen offline with it.
try:
    import macro_news as mn
    _HAS_MACRO = True
except ImportError:
    mn = None
    _HAS_MACRO = False

# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_OPEN = dtime(9, 30)
REGULAR_CLOSE = dtime(16, 0)


@dataclass
class ScreenerConfig:
    """All tunable parameters in one place.

    Every threshold here is a *choice*, not a discovered truth. They are the
    values named in the original specification. If you change them, you are
    changing the strategy, and you should re-validate.
    """

    # ---- Hard risk filters (enforced unless --relax) -------------------------
    min_price: float = 5.00          # exclude sub-$5 names (penny-stock rule)
    min_dollar_volume: float = 5e6   # 10d avg $ volume floor; liquidity guard
    min_rvol: float = 2.00           # >= 200% of time-adjusted 10d avg volume
    require_above_vwap: bool = True  # price must be above session VWAP
    rsi_low: float = 55.0            # momentum band lower bound
    rsi_high: float = 70.0           # upper bound (avoid already-overbought)
    require_macd_cross: bool = True  # bullish MACD cross within lookback
    macd_cross_lookback: int = 6     # bars since bullish cross, on --timeframe
    min_gap_pct: float = 2.0         # morning gap-up threshold, percent
    require_gap: bool = False        # gap is scored but not mandatory by default

    # ---- Indicator periods --------------------------------------------------
    atr_period: int = 14
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rvol_lookback_days: int = 10
    swing_lookback_bars: int = 20    # for intraday swing high/low levels

    # ---- Composite score weights (must be > 0; normalised internally) --------
    # These encode a view: "volume expansion matters most, then room to move,
    # then trend confirmation, then oscillator position." That view is a prior.
    w_rvol: float = 0.30
    w_atr: float = 0.25
    w_vwap: float = 0.15
    w_macd: float = 0.15
    w_rsi: float = 0.10
    w_gap: float = 0.05

    # ---- Risk plan ----------------------------------------------------------
    atr_target_multiple: float = 2.0   # take-profit = entry + k * ATR (fallback)
    atr_stop_buffer: float = 0.25      # stop placed k*ATR below chosen anchor
    min_acceptable_rr: float = 1.5     # flag setups below this
    risk_per_trade_pct: float = 0.5    # % of account risked, for size suggestion
    # Notional ceiling for the suggested size, as a % of account equity. Risk-
    # based sizing alone is unbounded: as the stop tightens, share count grows
    # without limit. 100 = no leverage. Raise ONLY if you actually have margin
    # and intend to use it.
    max_notional_pct: float = 100.0

    # ---- Fetching -----------------------------------------------------------
    intraday_interval: str = "5m"      # base intraday bar
    intraday_period: str = "1mo"       # history window for intraday pulls
    daily_period: str = "6mo"
    batch_size: int = 25               # tickers per yfinance request
    request_pause: float = 0.6         # seconds between batches (be polite)
    max_retries: int = 3


# A deliberately modest default universe: large, liquid, tight-spread names.
# Liquidity is the single most important thing you can control. Screening the
# Russell 3000 for "maximum profit potential" mostly surfaces illiquid junk that
# you cannot actually get filled in.
DEFAULT_UNIVERSE: List[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    # Semis / hardware
    "MU", "INTC", "QCOM", "TXN", "MRVL", "SMCI", "ARM", "ON", "LRCX", "AMAT",
    # Financials
    "JPM", "BAC", "GS", "MS", "SCHW", "C", "WFC", "COIN", "HOOD", "PYPL",
    # Consumer / industrials
    "WMT", "COST", "HD", "NKE", "SBUX", "DIS", "BA", "CAT", "DE", "UBER",
    # Energy / materials
    "XOM", "CVX", "OXY", "SLB", "FCX", "NEM", "MP",
    # Health
    "LLY", "UNH", "PFE", "MRNA", "ABBV",
    # High-beta / speculative but still liquid
    "PLTR", "SOFI", "RIVN", "LCID", "AFRM", "RBLX", "SNAP", "DKNG", "CVNA", "MSTR",
    # Liquid ETFs for market context (excluded from picks by default)
    "SPY", "QQQ", "IWM",
]

CONTEXT_TICKERS = {"SPY", "QQQ", "IWM"}  # shown as regime context, not ranked


# ==============================================================================
# SECTION 2 — SMALL UTILITIES
# ==============================================================================

def now_market() -> datetime:
    """Current wall-clock time in US market timezone."""
    return datetime.now(MARKET_TZ)


def market_session_state(ts: Optional[datetime] = None) -> str:
    """Classify the current moment as PREMARKET / OPEN / AFTERHOURS / CLOSED.

    Note: this does NOT consult an exchange holiday calendar. For production,
    use `pandas_market_calendars`. Weekend detection only is implemented here.
    """
    ts = ts or now_market()
    if ts.weekday() >= 5:
        return "CLOSED (weekend)"
    t = ts.time()
    if t < dtime(4, 0):
        return "CLOSED"
    if t < REGULAR_OPEN:
        return "PREMARKET"
    if t < REGULAR_CLOSE:
        return "OPEN"
    if t < dtime(20, 0):
        return "AFTERHOURS"
    return "CLOSED"


def _safe_div(a: float, b: float, default: float = float("nan")) -> float:
    """Division that returns `default` instead of raising / producing inf."""
    try:
        if b == 0 or b is None or (isinstance(b, float) and not math.isfinite(b)):
            return default
        out = a / b
        return out if math.isfinite(out) else default
    except (TypeError, ZeroDivisionError):
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    """Bound a value to [lo, hi], passing NaN through as lo."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return lo
    return max(lo, min(hi, x))


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise yfinance output to plain OHLCV columns.

    yfinance returns a MultiIndex (field, ticker) for batch downloads and,
    depending on version/flags, for single downloads too. This collapses both
    shapes to: Open, High, Low, Close, Volume.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Figure out which level holds the ticker symbol.
        lvl0 = set(df.columns.get_level_values(0))
        if ticker in lvl0:
            df = df.xs(ticker, axis=1, level=0)
        else:
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df = df.droplevel(1, axis=1)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[keep].copy()


def _drop_stale_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trailing REGULAR-HOURS bars with zero volume.

    Yahoo frequently appends a still-forming or placeholder bar with volume 0 and
    a flat OHLC. Including it corrupts VWAP, RVOL and the "last price" reading.

    THE REGULAR-HOURS RESTRICTION IS LOAD-BEARING. Yahoo zero-fills the volume
    of EVERY extended-hours bar (see the premarket note in `compute_factors`),
    while still returning real prices for them. An unrestricted walk-back
    therefore does not trim one placeholder bar — it deletes the entire
    after-hours and premarket session. Measured: 12 premarket bars in, 0 out.
    That destroys exactly the premarket high/low range this screener goes out of
    its way to preserve, and it does the most damage when the screener is run
    before the open, which is when a momentum screen is most useful.

    Zero volume inside 09:30-16:00 is a placeholder. Zero volume outside it is a
    feed limitation, and the prices attached to it are still real.
    """
    if df.empty or "Volume" not in df.columns:
        return df
    vol = df["Volume"].fillna(0)
    times = df.index.time
    # Walk back from the end, trimming only zero-volume bars that fall inside
    # the regular session; stop at the first extended-hours bar.
    idx = len(df) - 1
    while idx >= 0:
        in_rth = REGULAR_OPEN <= times[idx] < REGULAR_CLOSE
        if not (in_rth and vol.iloc[idx] == 0):
            break
        idx -= 1
    return df.iloc[: idx + 1]


def _to_market_tz(df: pd.DataFrame, naive_is_calendar_date: bool = False) -> pd.DataFrame:
    """Ensure the DatetimeIndex is tz-aware in market time.

    CAREFUL — this is a real bug source. yfinance returns tz-AWARE timestamps for
    intraday bars but tz-NAIVE midnight timestamps for daily bars. If you localize
    a naive daily index to UTC and convert to New York, `2026-08-07 00:00` becomes
    `2026-08-06 20:00` and every daily bar's calendar date silently shifts back by
    one day. Downstream, "yesterday's close" then resolves to TODAY's close, the
    gap calculation compares today's open against today's close, and the numbers
    look plausible while being wrong.

    So: daily bars carry a calendar DATE, not an instant. Localize them directly
    to market time (`naive_is_calendar_date=True`). Only genuinely naive intraday
    data should be assumed UTC.
    """
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            MARKET_TZ if naive_is_calendar_date else "UTC"
        )
    df.index = df.index.tz_convert(MARKET_TZ)
    return df


def _regular_hours_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter intraday bars to the 09:30–16:00 regular session."""
    if df.empty:
        return df
    t = df.index.time
    mask = (t >= REGULAR_OPEN) & (t < REGULAR_CLOSE)
    return df.loc[mask]


# ==============================================================================
# SECTION 3 — DATA PROVIDER
# ==============================================================================

class DataProvider:
    """Interface boundary between the screener and whatever feed you use.

    Everything downstream depends only on these two methods returning tz-aware
    OHLCV DataFrames. To move to a real-time feed, subclass this and reimplement
    `fetch_intraday` / `fetch_daily`. No other code needs to change.
    """

    def fetch_intraday(self, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def fetch_daily(self, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    """yfinance-backed provider. DELAYED DATA — see module docstring."""

    def __init__(self, cfg: ScreenerConfig, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        import yfinance as yf  # lazy import so --selftest works offline
        self._yf = yf

    # -- internal ------------------------------------------------------------
    def _download(self, tickers: Sequence[str], **kwargs) -> Optional[pd.DataFrame]:
        """yf.download with retry/backoff. Returns None on persistent failure."""
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                df = self._yf.download(
                    list(tickers),
                    progress=False,
                    auto_adjust=False,
                    threads=True,
                    group_by="ticker",
                    **kwargs,
                )
                if df is not None and not df.empty:
                    return df
                last_err = "empty frame"
            except Exception as exc:  # noqa: BLE001 - want any transport error
                last_err = exc
            time.sleep(self.cfg.request_pause * (2 ** attempt))
        if self.verbose:
            print(f"  [warn] download failed for {len(tickers)} tickers: {last_err}",
                  file=sys.stderr)
        return None

    def _batched(
        self, tickers: Sequence[str], label: str,
        naive_is_calendar_date: bool = False, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        batches = [
            list(tickers)[i : i + self.cfg.batch_size]
            for i in range(0, len(tickers), self.cfg.batch_size)
        ]
        for i, batch in enumerate(batches, 1):
            if self.verbose:
                print(f"  fetching {label} batch {i}/{len(batches)} "
                      f"({len(batch)} tickers)...", file=sys.stderr)
            raw = self._download(batch, **kwargs)
            if raw is None:
                continue
            for t in batch:
                try:
                    df = _flatten_columns(raw, t)
                except Exception:  # noqa: BLE001
                    continue
                df = df.dropna(how="all")
                if df.empty:
                    continue
                df = _to_market_tz(df, naive_is_calendar_date)
                out[t] = df
            if i < len(batches):
                time.sleep(self.cfg.request_pause)
        return out

    # -- public --------------------------------------------------------------
    def fetch_intraday(self, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
        """Intraday bars including pre/post so gap + premarket volume are visible."""
        data = self._batched(
            tickers,
            label=f"intraday {self.cfg.intraday_interval}",
            period=self.cfg.intraday_period,
            interval=self.cfg.intraday_interval,
            prepost=True,
        )
        return {t: _drop_stale_tail(df) for t, df in data.items()}

    def fetch_daily(self, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
        return self._batched(
            tickers,
            label="daily",
            naive_is_calendar_date=True,   # see _to_market_tz docstring
            period=self.cfg.daily_period,
            interval="1d",
            prepost=False,
        )


# ==============================================================================
# SECTION 4 — INDICATORS (pure functions, unit-testable, no I/O)
# ==============================================================================

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range: max of the three classic ranges.

    TR = max(H-L, |H - prev_close|, |L - prev_close|)
    The prev_close terms are what make TR gap-aware; plain H-L is not.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder smoothing.

    Wilder's smoother is an EMA with alpha = 1/period. Using ewm(adjust=False)
    matches Wilder's recursion after the warm-up period converges; the first
    `period` values are approximate. Charting platforms seed with an SMA, so
    expect tiny divergence on very short histories.
    """
    tr = true_range(df["High"], df["Low"], df["Close"])
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder).

    Returns 0-100. Note the well-known caveat: in a strong trend RSI can pin
    above 70 for days. "Overbought" is not a sell signal, it is a description.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means no down closes in the window -> RSI is 100 by definition
    out = out.where(avg_loss.ne(0.0), 100.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Returns a DataFrame with columns: macd, signal, hist.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


def bars_since_bullish_cross(hist: pd.Series, max_lookback: int = 50) -> Optional[int]:
    """How many bars ago the MACD histogram crossed from <=0 to >0.

    Returns 0 if the cross is on the most recent bar, None if no cross within
    `max_lookback`. This is the honest way to express "recent bullish crossover":
    a boolean throws away the freshness information, which is the part that
    matters.
    """
    h = hist.dropna()
    if len(h) < 2:
        return None
    vals = h.to_numpy()
    n = len(vals)
    limit = min(max_lookback, n - 1)
    for back in range(limit):
        i = n - 1 - back
        if vals[i] > 0 and vals[i - 1] <= 0:
            return back
    return None


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price, anchored to each session's open.

    Uses typical price (H+L+C)/3 as the per-bar price proxy — this is the
    standard convention and is what most platforms display. Critically, the
    cumulative sums RESET each trading day; a VWAP that runs across sessions is
    a different (and for intraday purposes, useless) statistic.

    Feed this REGULAR-HOURS bars only. Including premarket shifts VWAP toward
    thin, unrepresentative prints and changes the level materially.
    """
    if df.empty:
        return pd.Series(dtype=float)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical * df["Volume"]
    day = pd.Series(df.index.date, index=df.index)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["Volume"].groupby(day).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def time_adjusted_rvol(
    intraday_rth: pd.DataFrame, lookback_days: int = 10
) -> Tuple[float, float, float]:
    """Relative volume, adjusted for time of day.

    THIS IS THE PART MOST SCREENERS GET WRONG. Comparing today's partial-session
    volume against a full-day 10-day average guarantees a reading below 1.0 all
    morning and produces false negatives exactly when you care. The correct
    comparison is:

        today's cumulative volume as of time T
        --------------------------------------
        mean over the last N sessions of cumulative volume as of that same time T

    Returns (rvol, today_cum_volume, baseline_cum_volume).
    """
    if intraday_rth.empty:
        return (float("nan"), 0.0, 0.0)

    df = intraday_rth
    dates = pd.Series(df.index.date, index=df.index)
    unique_days = sorted(dates.unique())
    if len(unique_days) < 2:
        return (float("nan"), 0.0, 0.0)

    today = unique_days[-1]
    today_bars = df.loc[dates == today]
    if today_bars.empty:
        return (float("nan"), 0.0, 0.0)

    today_cum = float(today_bars["Volume"].sum())
    cutoff = today_bars.index[-1].time()

    # Build the baseline: cumulative volume up to the same clock time on each
    # of the prior N sessions.
    prior_days = unique_days[-(lookback_days + 1) : -1]
    baselines: List[float] = []
    for d in prior_days:
        day_bars = df.loc[dates == d]
        if day_bars.empty:
            continue
        upto = day_bars.loc[[t.time() <= cutoff for t in day_bars.index]]
        if upto.empty:
            continue
        baselines.append(float(upto["Volume"].sum()))

    if not baselines:
        return (float("nan"), today_cum, 0.0)

    baseline = float(np.mean(baselines))
    return (_safe_div(today_cum, baseline), today_cum, baseline)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate finer bars into coarser ones (e.g. 5m -> 15m).

    `label='left', closed='left'` matches standard charting convention: a bar
    stamped 09:30 covers [09:30, 09:45).
    """
    if df.empty:
        return df
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["Close"])


def swing_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
    """Most recent swing high / swing low over the last `lookback` bars.

    Deliberately simple (rolling extremum) rather than fractal/pivot detection.
    Pivot algorithms need confirmation bars on both sides, which means the most
    recent — and most relevant — pivot is always missing. For stop placement,
    the rolling low is more useful and more honest about what is known now.
    """
    if df.empty:
        return (float("nan"), float("nan"))
    tail = df.tail(lookback)
    return (float(tail["High"].max()), float(tail["Low"].min()))


# ==============================================================================
# SECTION 5 — FACTOR EXTRACTION
# ==============================================================================

@dataclass
class Factors:
    """Everything measured for one ticker at one moment in time."""

    ticker: str
    ok: bool = True
    reason: str = ""                       # why it was rejected, if rejected

    # Price state
    last: float = float("nan")
    prev_close: float = float("nan")
    day_open: float = float("nan")
    day_high: float = float("nan")
    day_low: float = float("nan")

    # Liquidity
    rvol: float = float("nan")
    today_volume: float = float("nan")
    baseline_volume: float = float("nan")
    avg_dollar_volume: float = float("nan")
    premarket_volume: float = float("nan")   # NaN = feed does not provide it
    premarket_available: bool = False
    premarket_high: float = float("nan")
    premarket_low: float = float("nan")
    premarket_range_pct: float = float("nan")

    # Volatility
    atr_daily: float = float("nan")
    atr_pct: float = float("nan")          # ATR as % of price
    atr_intraday: float = float("nan")

    # Trend / location
    vwap: float = float("nan")
    vwap_dist_pct: float = float("nan")    # (last - vwap) / vwap * 100
    above_vwap: bool = False

    # Momentum
    rsi: float = float("nan")
    macd_hist: float = float("nan")
    macd_bars_since_cross: Optional[int] = None
    macd_bullish: bool = False

    # Gap
    gap_pct: float = float("nan")

    # Structure
    swing_high: float = float("nan")
    swing_low: float = float("nan")

    # Scoring (filled later)
    score: float = 0.0
    subscores: Dict[str, float] = field(default_factory=dict)
    filters_passed: Dict[str, bool] = field(default_factory=dict)


def compute_factors(
    ticker: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    cfg: ScreenerConfig,
    timeframe: str = "5m",
) -> Factors:
    """Turn raw bars into the factor set. Pure: no network, no printing.

    `intraday` should include extended hours (needed for premarket volume and a
    reliable gap calculation); regular-hours filtering happens internally where
    it matters (VWAP, RVOL).
    """
    f = Factors(ticker=ticker)

    # ---- Sanity: do we have enough data to say anything? --------------------
    if intraday is None or intraday.empty or daily is None or daily.empty:
        f.ok, f.reason = False, "no data"
        return f
    if len(daily) < cfg.atr_period + 2:
        f.ok, f.reason = False, f"insufficient daily history ({len(daily)} bars)"
        return f

    rth = _regular_hours_only(intraday)
    if rth.empty:
        f.ok, f.reason = False, "no regular-hours bars"
        return f

    # ---- Optionally coarsen the momentum timeframe --------------------------
    # MACD/RSI are read on the requested timeframe; VWAP/RVOL always use the
    # base 5m series for resolution.
    tf_map = {"5m": None, "15m": "15min", "30m": "30min", "1h": "60min"}
    if timeframe not in tf_map:
        f.ok, f.reason = False, f"unsupported timeframe {timeframe}"
        return f
    rule = tf_map[timeframe]
    momo_df = rth if rule is None else resample_ohlcv(rth, rule)
    if len(momo_df) < cfg.macd_slow + cfg.macd_signal:
        f.ok, f.reason = False, f"insufficient {timeframe} bars ({len(momo_df)})"
        return f

    # ---- Price state --------------------------------------------------------
    dates = pd.Series(rth.index.date, index=rth.index)
    today = sorted(dates.unique())[-1]
    today_rth = rth.loc[dates == today]

    f.last = float(rth["Close"].iloc[-1])
    f.day_open = float(today_rth["Open"].iloc[0])
    f.day_high = float(today_rth["High"].max())
    f.day_low = float(today_rth["Low"].min())

    # Prior close from the DAILY series. Using the intraday series for this is a
    # common bug: intraday history may not extend far enough, and the last
    # intraday bar of the prior session != the official close.
    daily_dates = [d.date() for d in daily.index]
    prior_rows = [i for i, d in enumerate(daily_dates) if d < today]
    if prior_rows:
        f.prev_close = float(daily["Close"].iloc[prior_rows[-1]])
    else:
        f.prev_close = float("nan")

    # Completed daily bars only (today's bar is still forming while the market
    # is open). Used for ATR and average dollar volume so that neither is
    # biased downward by a partially-realised session.
    daily_completed = daily.iloc[prior_rows] if prior_rows else daily
    if len(daily_completed) < cfg.atr_period + 1:
        daily_completed = daily

    # ---- Gap ----------------------------------------------------------------
    f.gap_pct = _safe_div(f.day_open - f.prev_close, f.prev_close, float("nan")) * 100.0

    # ---- Premarket ----------------------------------------------------------
    # Extended-hours bars from 04:00 to 09:30 today.
    #
    # IMPORTANT DATA CAVEAT: Yahoo returns extended-hours PRICE bars but
    # zero-fills their VOLUME at every interval (1m/5m/15m alike). So premarket
    # volume is genuinely UNAVAILABLE from this feed, not "zero". Reporting 0
    # would be worse than reporting nothing, because 0 reads as a real
    # observation of no premarket interest. We distinguish the two: NaN means
    # unknown. The premarket price RANGE is real and is captured instead, since
    # it still tells you whether the name was active overnight.
    ext = intraday
    ext_dates = pd.Series(ext.index.date, index=ext.index)
    pm_mask = (ext_dates == today) & pd.Series(
        [t.time() < REGULAR_OPEN for t in ext.index], index=ext.index
    )
    pm = ext.loc[pm_mask]
    if pm.empty:
        f.premarket_volume = float("nan")
        f.premarket_available = False
    else:
        pm_vol = float(pm["Volume"].sum())
        f.premarket_available = pm_vol > 0
        f.premarket_volume = pm_vol if pm_vol > 0 else float("nan")
        f.premarket_high = float(pm["High"].max())
        f.premarket_low = float(pm["Low"].min())
        f.premarket_range_pct = _safe_div(
            f.premarket_high - f.premarket_low, f.prev_close, float("nan")
        ) * 100.0

    # ---- Liquidity ----------------------------------------------------------
    f.rvol, f.today_volume, f.baseline_volume = time_adjusted_rvol(
        rth, cfg.rvol_lookback_days
    )
    dv = (daily_completed["Close"] * daily_completed["Volume"]).tail(
        cfg.rvol_lookback_days
    )
    f.avg_dollar_volume = float(dv.mean()) if not dv.empty else float("nan")

    # ---- Volatility ---------------------------------------------------------
    # ATR is computed on COMPLETED sessions (see daily_completed above): ATR is
    # meant to describe a typical full-day range, and is used here to size stops.
    daily_atr = atr(daily_completed, cfg.atr_period)
    f.atr_daily = float(daily_atr.iloc[-1]) if not daily_atr.dropna().empty else float("nan")
    f.atr_pct = _safe_div(f.atr_daily, f.last, float("nan")) * 100.0

    intra_atr = atr(momo_df, cfg.atr_period)
    f.atr_intraday = (
        float(intra_atr.iloc[-1]) if not intra_atr.dropna().empty else float("nan")
    )

    # ---- VWAP ---------------------------------------------------------------
    vwap_series = session_vwap(today_rth)
    if not vwap_series.dropna().empty:
        f.vwap = float(vwap_series.iloc[-1])
        f.vwap_dist_pct = _safe_div(f.last - f.vwap, f.vwap, float("nan")) * 100.0
        f.above_vwap = bool(f.last > f.vwap)

    # ---- Momentum -----------------------------------------------------------
    rsi_series = rsi(momo_df["Close"], cfg.rsi_period)
    if not rsi_series.dropna().empty:
        f.rsi = float(rsi_series.iloc[-1])

    macd_df = macd(momo_df["Close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    if not macd_df["hist"].dropna().empty:
        f.macd_hist = float(macd_df["hist"].iloc[-1])
        f.macd_bars_since_cross = bars_since_bullish_cross(macd_df["hist"])
        f.macd_bullish = (
            f.macd_bars_since_cross is not None
            and f.macd_bars_since_cross <= cfg.macd_cross_lookback
        )

    # ---- Structure ----------------------------------------------------------
    f.swing_high, f.swing_low = swing_levels(momo_df, cfg.swing_lookback_bars)

    return f


# ==============================================================================
# SECTION 6 — FILTERS AND SCORING
# ==============================================================================

def apply_filters(f: Factors, cfg: ScreenerConfig) -> Dict[str, bool]:
    """Evaluate every hard rule and record pass/fail individually.

    Recording each rule separately (rather than short-circuiting) means the
    dashboard can show you *why* nothing passed — which on most days is the
    genuinely useful output, because the conjunction of all six conditions is
    rare.
    """
    checks = {
        "price>$5": bool(f.last > cfg.min_price),
        "liquidity": bool(
            math.isfinite(f.avg_dollar_volume)
            and f.avg_dollar_volume >= cfg.min_dollar_volume
        ),
        "rvol>=%.1fx" % cfg.min_rvol: bool(
            math.isfinite(f.rvol) and f.rvol >= cfg.min_rvol
        ),
        "above_vwap": bool(f.above_vwap) if cfg.require_above_vwap else True,
        "rsi_%g-%g" % (cfg.rsi_low, cfg.rsi_high): bool(
            math.isfinite(f.rsi) and cfg.rsi_low <= f.rsi <= cfg.rsi_high
        ),
        "macd_cross": bool(f.macd_bullish) if cfg.require_macd_cross else True,
    }
    if cfg.require_gap:
        checks["gap>%.1f%%" % cfg.min_gap_pct] = bool(
            math.isfinite(f.gap_pct) and f.gap_pct >= cfg.min_gap_pct
        )
    return checks


def _score_rvol(rvol: float) -> float:
    """Map RVOL to 0-1. Saturating: 5x is not 2.5x as good as 2x.

    log-scaled because volume distributions are lognormal-ish, and because the
    marginal information in going from 5x to 10x is small (and often signals a
    news event whose direction the screener cannot know).
    """
    if not math.isfinite(rvol) or rvol <= 0:
        return 0.0
    return _clamp(math.log(1.0 + rvol) / math.log(1.0 + 6.0), 0.0, 1.0)


def _score_atr(atr_pct: float) -> float:
    """Map ATR% to 0-1, peaking around 4-6% daily range.

    Intentionally NOT monotonic. The prompt asks to "prioritise high ATR", but
    unbounded preference for volatility selects for names that are untradeable:
    the stop distance required grows faster than the realistic target, and the
    spread widens. A 20% ATR name is not a better trade than a 5% ATR name; it
    is a different, worse-risk-adjusted one.
    """
    if not math.isfinite(atr_pct) or atr_pct <= 0:
        return 0.0
    if atr_pct <= 5.0:
        return _clamp(atr_pct / 5.0, 0.0, 1.0)
    # Decay above 5%: reaches ~0.4 at 12%, ~0.2 at 20%.
    return _clamp(1.0 / (1.0 + 0.12 * (atr_pct - 5.0)), 0.0, 1.0)


def _score_vwap(dist_pct: float) -> float:
    """Reward being above VWAP, but penalise being extended far above it.

    Best score near +0.3% above VWAP: trend confirmed, entry not chased.
    Above ~2% you are buying from whoever bought at VWAP, with your stop
    correspondingly far away.
    """
    if not math.isfinite(dist_pct):
        return 0.0
    if dist_pct <= 0:
        return 0.0
    if dist_pct <= 0.5:
        return _clamp(0.6 + 0.8 * dist_pct, 0.0, 1.0)
    return _clamp(1.0 / (1.0 + 0.6 * (dist_pct - 0.5)), 0.0, 1.0)


def _score_macd(bars_since: Optional[int], hist: float, lookback: int) -> float:
    """Fresher crossover scores higher; no crossover scores 0.

    A cross 1 bar ago and a cross 30 bars ago are completely different states,
    and the second is usually already exhausted.
    """
    if bars_since is None:
        return 0.0
    if bars_since > lookback:
        return 0.0
    freshness = 1.0 - (bars_since / (lookback + 1.0))
    strength = 1.0 if (math.isfinite(hist) and hist > 0) else 0.0
    return _clamp(0.7 * freshness + 0.3 * strength, 0.0, 1.0)


def _score_rsi(value: float, lo: float, hi: float) -> float:
    """Triangular preference peaking at the midpoint of the target band."""
    if not math.isfinite(value):
        return 0.0
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if half <= 0:
        return 0.0
    return _clamp(1.0 - abs(value - mid) / half, 0.0, 1.0)


def _score_gap(gap_pct: float, min_gap: float) -> float:
    """Reward a gap up, cap the reward. Huge gaps are news-driven coin flips."""
    if not math.isfinite(gap_pct) or gap_pct <= 0:
        return 0.0
    return _clamp(gap_pct / (2.0 * max(min_gap, 0.5)), 0.0, 1.0)


def score_factors(f: Factors, cfg: ScreenerConfig) -> Factors:
    """Compute weighted composite score in 0-100.

    The weights are normalised so the score is interpretable regardless of how
    you set them. It is still an arbitrary index — comparable across tickers on
    the same day, NOT comparable across days or across universes.
    """
    subs = {
        "rvol": _score_rvol(f.rvol),
        "atr": _score_atr(f.atr_pct),
        "vwap": _score_vwap(f.vwap_dist_pct),
        "macd": _score_macd(f.macd_bars_since_cross, f.macd_hist,
                            cfg.macd_cross_lookback),
        "rsi": _score_rsi(f.rsi, cfg.rsi_low, cfg.rsi_high),
        "gap": _score_gap(f.gap_pct, cfg.min_gap_pct),
    }
    weights = {
        "rvol": cfg.w_rvol, "atr": cfg.w_atr, "vwap": cfg.w_vwap,
        "macd": cfg.w_macd, "rsi": cfg.w_rsi, "gap": cfg.w_gap,
    }
    total_w = sum(weights.values()) or 1.0
    f.subscores = subs
    f.score = 100.0 * sum(subs[k] * weights[k] for k in subs) / total_w
    return f


# ==============================================================================
# SECTION 7 — RISK PLAN
# ==============================================================================

@dataclass
class RiskPlan:
    """A concrete, checkable trade plan. All levels are prices, not opinions."""

    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    stop_basis: str = ""
    target_basis: str = ""
    risk_per_share: float = float("nan")
    reward_per_share: float = float("nan")
    rr: float = float("nan")
    stop_pct: float = float("nan")
    target_pct: float = float("nan")
    residual_range: float = float("nan")   # typical daily range still unspent
    range_used_pct: float = float("nan")
    target_feasible: bool = True
    feasible_target: float = float("nan")
    rr_feasible: float = float("nan")      # R:R capped to reachable range
    shares_for_1pct_account: float = float("nan")   # risk-derived, AFTER cap
    shares_uncapped: float = float("nan")  # what pure risk sizing asked for
    notional: float = float("nan")         # shares * entry
    notional_capped: bool = False          # True if the ceiling bound the size
    acceptable: bool = False
    notes: List[str] = field(default_factory=list)


def build_risk_plan(
    f: Factors, cfg: ScreenerConfig, account_size: float = 25_000.0
) -> RiskPlan:
    """Derive entry / stop / target and the resulting reward-to-risk ratio.

    Stop logic — take the TIGHTEST sensible anchor that still sits below
    meaningful structure:
      candidate A: recent swing low  - buffer*ATR_intraday
      candidate B: session VWAP      - buffer*ATR_intraday
    We use the HIGHER of the two (tighter stop) when both are below price,
    because the nearer level is the one that, if broken, invalidates the setup
    first. Placing the stop at the further level just pays more to learn the
    same thing.

    Target logic:
      preferred: the nearest overhead structure (recent swing high / day high)
                 if it offers meaningful room
      fallback:  entry + k * ATR_daily

    A note on what R:R actually means: it is a ratio of DISTANCES, not of
    probabilities. A 3:1 setup is only good if it hits more than ~25% of the
    time. This function cannot estimate that hit rate, and neither can any of
    the indicators above. Do not read a high R:R as a high expected value.
    """
    plan = RiskPlan()

    if not math.isfinite(f.last) or f.last <= 0:
        plan.notes.append("no valid last price")
        return plan

    plan.entry = f.last
    buf = f.atr_intraday if math.isfinite(f.atr_intraday) else 0.0
    buf *= cfg.atr_stop_buffer

    # ---- Stop --------------------------------------------------------------
    candidates: List[Tuple[float, str]] = []
    if math.isfinite(f.swing_low) and f.swing_low < plan.entry:
        candidates.append((f.swing_low - buf, "swing low - 0.25*ATR"))
    if math.isfinite(f.vwap) and f.vwap < plan.entry:
        candidates.append((f.vwap - buf, "VWAP - 0.25*ATR"))

    if candidates:
        # Tightest valid stop = the highest candidate still below entry.
        valid = [c for c in candidates if c[0] < plan.entry]
        if valid:
            plan.stop, plan.stop_basis = max(valid, key=lambda c: c[0])
    if not math.isfinite(plan.stop):
        # Last resort: pure volatility stop.
        atr_ref = f.atr_daily if math.isfinite(f.atr_daily) else plan.entry * 0.02
        plan.stop = plan.entry - 1.0 * atr_ref
        plan.stop_basis = "entry - 1.0*ATR (no structure available)"
        plan.notes.append("no structural stop available; using volatility stop")

    # ---- Target ------------------------------------------------------------
    atr_ref = f.atr_daily if math.isfinite(f.atr_daily) else plan.entry * 0.02
    atr_target = plan.entry + cfg.atr_target_multiple * atr_ref

    overhead = [
        (lvl, name)
        for lvl, name in ((f.swing_high, "recent swing high"),
                          (f.day_high, "day high"))
        if math.isfinite(lvl) and lvl > plan.entry * 1.001
    ]
    if overhead:
        lvl, name = min(overhead, key=lambda c: c[0])
        # Only use structure as the target if it isn't trivially close.
        if (lvl - plan.entry) >= 0.5 * (plan.entry - plan.stop):
            plan.target, plan.target_basis = lvl, name
        else:
            plan.target, plan.target_basis = (
                atr_target,
                f"entry + {cfg.atr_target_multiple:g}*ATR "
                f"(structure too close: {name} @ {lvl:.2f})",
            )
    else:
        plan.target, plan.target_basis = (
            atr_target, f"entry + {cfg.atr_target_multiple:g}*ATR (no overhead level)"
        )

    # ---- Derived numbers ----------------------------------------------------
    # ---- Feasibility: is the target reachable in the time remaining? --------
    #
    # This check exists because R:R is trivially gameable by pushing the target
    # further out. A 2x-DAILY-ATR target attached to a trade entered at 13:00,
    # on a name that has already traveled its full typical daily range, produces
    # a headline "10:1 R:R" that is arithmetically true and practically
    # meaningless — the price simply will not get there before the close.
    #
    # Budget: a typical session delivers about one ATR of range. Subtract what
    # today has already spent. What remains is roughly the room still available.
    # If the target needs more than that, say so loudly rather than letting the
    # ratio flatter the setup.
    realized_range = (
        f.day_high - f.day_low
        if math.isfinite(f.day_high) and math.isfinite(f.day_low)
        else float("nan")
    )
    if math.isfinite(realized_range) and math.isfinite(atr_ref):
        plan.residual_range = max(0.0, atr_ref - realized_range)
        plan.range_used_pct = _safe_div(realized_range, atr_ref, float("nan")) * 100.0

    plan.risk_per_share = plan.entry - plan.stop
    plan.reward_per_share = plan.target - plan.entry
    plan.rr = _safe_div(plan.reward_per_share, plan.risk_per_share, float("nan"))

    if math.isfinite(plan.residual_range):
        if plan.reward_per_share > plan.residual_range:
            plan.target_feasible = False
            plan.notes.append(
                f"target needs ${plan.reward_per_share:.2f} but only "
                f"${plan.residual_range:.2f} of typical daily range remains "
                f"({_fmt(plan.range_used_pct, '.0f')}% already used) — "
                f"unlikely to fill today"
            )
            # Report the honest, range-constrained alternative alongside it.
            feasible_target = plan.entry + plan.residual_range
            plan.rr_feasible = _safe_div(
                feasible_target - plan.entry, plan.risk_per_share, float("nan")
            )
            plan.feasible_target = feasible_target
        else:
            plan.target_feasible = True
            plan.rr_feasible = plan.rr
            plan.feasible_target = plan.target
    plan.stop_pct = _safe_div(plan.risk_per_share, plan.entry, float("nan")) * 100.0
    plan.target_pct = _safe_div(plan.reward_per_share, plan.entry, float("nan")) * 100.0
    # Judge acceptability on the REACHABLE ratio. Using the headline ratio would
    # let an unreachable target launder a poor setup into a great-looking one.
    judge_rr = plan.rr_feasible if math.isfinite(plan.rr_feasible) else plan.rr
    plan.acceptable = bool(math.isfinite(judge_rr) and judge_rr >= cfg.min_acceptable_rr)

    # ---- Position size ------------------------------------------------------
    # Risk-based sizing answers "how many shares put cfg.risk_per_trade_pct of
    # the account at risk if the stop fills?" — and on its own it is UNBOUNDED.
    # Share count scales as 1/stop_distance, so a setup hugging VWAP with a 5c
    # stop asks for 2,272 shares of a $100 stock: $227k of exposure on a $25k
    # account, 9x equity, with no warning printed. That number is arithmetically
    # correct and operationally nonsense — it is unbuyable, and if it were
    # buyable a single gap through the stop (precisely what the catalyst layer
    # exists to warn about) would exceed the whole account many times over.
    #
    # The stop-distance premise is also weakest exactly when it matters most:
    # tight stops assume continuous price, and a 5c stop on a delayed feed will
    # not fill at 5c. So cap the notional and say so.
    dollars_at_risk = account_size * (cfg.risk_per_trade_pct / 100.0)
    raw_shares = math.floor(_safe_div(dollars_at_risk, plan.risk_per_share, 0.0))
    plan.shares_uncapped = raw_shares

    max_notional = account_size * (cfg.max_notional_pct / 100.0)
    cap_shares = math.floor(_safe_div(max_notional, plan.entry, 0.0))
    shares = min(raw_shares, cap_shares)
    plan.notional_capped = bool(raw_shares > cap_shares)
    plan.shares_for_1pct_account = shares
    plan.notional = shares * plan.entry

    if plan.notional_capped:
        plan.notes.append(
            f"size capped by notional ceiling: risk sizing asked for "
            f"{raw_shares:,.0f} sh (${raw_shares * plan.entry:,.0f}, "
            f"{_fmt(raw_shares * plan.entry / account_size, '.1f')}x account) "
            f"on a ${_fmt(plan.risk_per_share)}/sh stop — reduced to "
            f"{shares:,.0f} sh. A stop this tight will not survive slippage; "
            f"treat the R:R as optimistic."
        )

    if plan.stop_pct > 5.0:
        plan.notes.append(
            f"wide stop ({plan.stop_pct:.1f}%) — position size must shrink accordingly"
        )
    if not plan.acceptable:
        plan.notes.append(
            f"reachable R:R {_fmt(judge_rr, '.2f')} below minimum "
            f"{cfg.min_acceptable_rr:g}"
        )
    return plan


# ==============================================================================
# SECTION 8 — SCREENER ORCHESTRATION
# ==============================================================================

@dataclass
class ScreenResult:
    factors: Factors
    plan: Optional[RiskPlan] = None
    passed_all: bool = False
    n_passed: int = 0
    n_checks: int = 0
    catalyst: object = None            # macro_news.CatalystFlag when enabled
    size_multiplier: float = 1.0       # macro regime x catalyst risk
    adjusted_shares: float = float("nan")
    required_rr: float = float("nan")  # base minimum x catalyst uplift


def apply_risk_overlay(
    results: List[ScreenResult],
    cfg: ScreenerConfig,
    macro_regime=None,
    ticker_news: Optional[Dict[str, list]] = None,
    account_size: float = 25_000.0,
) -> None:
    """Fold macro regime and per-ticker catalyst risk into position sizing.

    This is where the geopolitical layer becomes ACTIONABLE. Both inputs adjust
    the same two levers, and both adjust them in the SAME direction (down):

        size_multiplier      = macro_regime x catalyst_risk
        required_R:R         = base_minimum x catalyst_uplift

    Note what is NOT happening: neither input alters the Profit Potential Score
    or the ranking. That is deliberate. Macro stress and news presence tell you
    HOW MUCH to risk, not WHAT to buy. Letting them tilt the ranking would be
    smuggling an unvalidated directional view into a technical screen.

    Multiplying (rather than taking the minimum) is intentional: an elevated
    macro regime AND a stock-specific breaking catalyst are independent sources
    of gap risk, and their effects compound.
    """
    macro_mult = 1.0
    if macro_regime is not None and math.isfinite(
        getattr(macro_regime, "size_multiplier", float("nan"))
    ):
        macro_mult = macro_regime.size_multiplier

    for r in results:
        cat = None
        if ticker_news is not None and _HAS_MACRO:
            heads = ticker_news.get(r.factors.ticker, [])
            # Pass RVOL so headline noise on quiet names is discounted.
            cat = mn.assess_catalyst_risk(
                r.factors.ticker, heads, rvol=r.factors.rvol
            )
        r.catalyst = cat

        cat_mult = cat.size_multiplier if cat else 1.0
        cat_rr = cat.rr_requirement_multiplier if cat else 1.0

        r.size_multiplier = macro_mult * cat_mult
        r.required_rr = cfg.min_acceptable_rr * cat_rr

        if r.plan and math.isfinite(r.plan.shares_for_1pct_account):
            r.adjusted_shares = math.floor(
                r.plan.shares_for_1pct_account * r.size_multiplier
            )
            # Re-judge acceptability against the RAISED bar. A setup that clears
            # 1.5:1 in calm conditions may not clear 2.25:1 into a live catalyst.
            judge = (r.plan.rr_feasible if math.isfinite(r.plan.rr_feasible)
                     else r.plan.rr)
            if math.isfinite(judge) and judge < r.required_rr:
                r.plan.acceptable = False
                if cat_rr > 1.0:
                    r.plan.notes.append(
                        f"R:R bar raised to {r.required_rr:.2f} by catalyst risk; "
                        f"reachable {judge:.2f} does not clear it"
                    )


def run_screen(
    tickers: Sequence[str],
    cfg: ScreenerConfig,
    provider: DataProvider,
    timeframe: str = "5m",
    relax: bool = False,
    account_size: float = 25_000.0,
    verbose: bool = True,
) -> Tuple[List[ScreenResult], List[Factors], Dict[str, Factors]]:
    """Fetch, compute, filter, score, rank.

    Returns (ranked_results, rejected_factors, context_factors).
    `relax=True` keeps every ticker in the ranking regardless of filter failures,
    which is usually what you want, because the strict conjunction returns an
    empty set on most days.
    """
    tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order
    if verbose:
        print(f"  universe: {len(tickers)} tickers", file=sys.stderr)

    intraday = provider.fetch_intraday(tickers)
    daily = provider.fetch_daily(tickers)

    results: List[ScreenResult] = []
    rejected: List[Factors] = []
    context: Dict[str, Factors] = {}

    for t in tickers:
        f = compute_factors(t, intraday.get(t), daily.get(t), cfg, timeframe)
        if not f.ok:
            rejected.append(f)
            continue

        f.filters_passed = apply_filters(f, cfg)
        f = score_factors(f, cfg)

        n_pass = sum(1 for v in f.filters_passed.values() if v)
        n_checks = len(f.filters_passed)
        passed_all = n_pass == n_checks

        if t in CONTEXT_TICKERS:
            context[t] = f
            continue

        if passed_all or relax:
            results.append(
                ScreenResult(
                    factors=f,
                    plan=build_risk_plan(f, cfg, account_size),
                    passed_all=passed_all,
                    n_passed=n_pass,
                    n_checks=n_checks,
                )
            )
        else:
            f.reason = "failed: " + ", ".join(
                k for k, v in f.filters_passed.items() if not v
            )
            rejected.append(f)

    # Rank: fully-qualified candidates first, then by composite score.
    results.sort(key=lambda r: (r.passed_all, r.factors.score), reverse=True)
    return results, rejected, context


# ==============================================================================
# SECTION 9 — TERMINAL DASHBOARD
# ==============================================================================

W = 78  # dashboard width


def _rule(ch: str = "─") -> str:
    return ch * W


def _fmt(x: float, spec: str = ".2f", dash: str = "—") -> str:
    """Format a float, degrading gracefully to an em-dash on NaN."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return dash
    return format(x, spec)


def _bar(value01: float, width: int = 12) -> str:
    """Tiny ASCII meter for a 0-1 subscore."""
    v = _clamp(value01, 0.0, 1.0)
    filled = int(round(v * width))
    return "█" * filled + "·" * (width - filled)


def _human_volume(v: float) -> str:
    if not math.isfinite(v):
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v/div:.1f}{unit}"
    return f"{v:.0f}"


def print_header(cfg: ScreenerConfig, timeframe: str, universe_n: int,
                 relax: bool) -> None:
    ts = now_market()
    state = market_session_state(ts)
    print()
    print(_rule("═"))
    print("  INTRADAY MOMENTUM SCREENER".ljust(W - 20)
          + f"{ts:%Y-%m-%d %H:%M} ET")
    print(_rule("═"))
    print(f"  Session: {state:<22} Timeframe: {timeframe:<8} "
          f"Universe: {universe_n}")
    print(f"  Mode:    {'RANK-ALL (filters advisory)' if relax else 'STRICT (all filters enforced)'}")
    if state != "OPEN":
        print("  ⚠ Market is not in regular session — VWAP/RVOL reflect the last")
        print("    completed session and are not actionable intraday readings.")
    print("  ⚠ Data source is DELAYED. Not for order routing. Not advice.")
    print(_rule())


def print_context(context: Dict[str, Factors]) -> None:
    """Market regime line. A long setup in a tape that is broadly selling off is
    a different proposition from the same setup in a bid market."""
    if not context:
        return
    parts = []
    for t in ("SPY", "QQQ", "IWM"):
        f = context.get(t)
        if not f or not math.isfinite(f.last):
            continue
        chg = _safe_div(f.last - f.prev_close, f.prev_close, float("nan")) * 100.0
        arrow = "▲" if (math.isfinite(chg) and chg >= 0) else "▼"
        vw = "above VWAP" if f.above_vwap else "below VWAP"
        parts.append(f"{t} {arrow}{_fmt(abs(chg), '.2f')}% ({vw})")
    if parts:
        print("  MARKET CONTEXT: " + " │ ".join(parts))
        print(_rule())


def print_candidate(rank: int, res: ScreenResult, cfg: ScreenerConfig) -> None:
    f, p = res.factors, res.plan
    flag = "✓ ALL FILTERS PASS" if res.passed_all else \
           f"◐ {res.n_passed}/{res.n_checks} filters"

    print()
    print(f"  #{rank}  {f.ticker:<6}  ${_fmt(f.last)}   "
          f"SCORE {_fmt(f.score, '.1f')}/100   [{flag}]")
    print("  " + _rule("-")[: W - 4])

    # --- Factor readings ----------------------------------------------------
    print(f"    RVOL      {_fmt(f.rvol, '.2f')}x  "
          f"({_human_volume(f.today_volume)} vs {_human_volume(f.baseline_volume)} "
          f"typical by this time)")
    print(f"    ATR       ${_fmt(f.atr_daily)}  ({_fmt(f.atr_pct, '.2f')}% of price)"
          f"   |  intraday ATR ${_fmt(f.atr_intraday)}")
    print(f"    VWAP      ${_fmt(f.vwap)}  "
          f"({'+' if (math.isfinite(f.vwap_dist_pct) and f.vwap_dist_pct >= 0) else ''}"
          f"{_fmt(f.vwap_dist_pct, '.2f')}%)   "
          f"{'ABOVE' if f.above_vwap else 'BELOW'}")
    cross = (f"{f.macd_bars_since_cross} bars ago"
             if f.macd_bars_since_cross is not None else "none in lookback")
    print(f"    MACD      hist {_fmt(f.macd_hist, '+.4f')}   bullish cross: {cross}")
    print(f"    RSI       {_fmt(f.rsi, '.1f')}   "
          f"(target band {cfg.rsi_low:g}–{cfg.rsi_high:g})")
    pm_txt = (f"premkt vol {_human_volume(f.premarket_volume)}"
              if f.premarket_available
              else "premkt vol n/a (feed limitation)")
    pm_rng = (f"   premkt range {_fmt(f.premarket_range_pct, '.2f')}%"
              if math.isfinite(f.premarket_range_pct) else "")
    print(f"    GAP       {_fmt(f.gap_pct, '+.2f')}%   {pm_txt}{pm_rng}")
    print(f"              prev close ${_fmt(f.prev_close)}   "
          f"open ${_fmt(f.day_open)}   "
          f"day range ${_fmt(f.day_low)}-${_fmt(f.day_high)}")

    # --- Score decomposition ------------------------------------------------
    print("    " + "-" * (W - 8))
    print("    SCORE BREAKDOWN")
    labels = {"rvol": "Rel Volume", "atr": "Volatility", "vwap": "VWAP loc.",
              "macd": "MACD", "rsi": "RSI", "gap": "Gap"}
    weights = {"rvol": cfg.w_rvol, "atr": cfg.w_atr, "vwap": cfg.w_vwap,
               "macd": cfg.w_macd, "rsi": cfg.w_rsi, "gap": cfg.w_gap}
    for k, lab in labels.items():
        v = f.subscores.get(k, 0.0)
        print(f"      {lab:<12} {_bar(v)}  {v:.2f}  (w={weights[k]:.2f})")

    # --- Failed filters -----------------------------------------------------
    fails = [k for k, v in f.filters_passed.items() if not v]
    if fails:
        print(f"    FAILING:  {', '.join(fails)}")

    # --- Risk plan ----------------------------------------------------------
    if p is None:
        return
    print("    " + "-" * (W - 8))
    print("    TRADE PLAN (hypothetical long)")
    print(f"      Entry     ${_fmt(p.entry)}")
    print(f"      Stop      ${_fmt(p.stop)}   "
          f"(-{_fmt(p.stop_pct, '.2f')}%)   basis: {p.stop_basis}")
    print(f"      Target    ${_fmt(p.target)}   "
          f"(+{_fmt(p.target_pct, '.2f')}%)   basis: {p.target_basis}")
    verdict = "OK" if p.acceptable else "BELOW THRESHOLD"
    if not p.target_feasible and math.isfinite(p.rr_feasible):
        print(f"      R:R       {_fmt(p.rr, '.2f')} : 1 nominal  →  "
              f"{_fmt(p.rr_feasible, '.2f')} : 1 reachable today   [{verdict}]")
        print(f"                reachable target ${_fmt(p.feasible_target)} "
              f"({_fmt(p.range_used_pct, '.0f')}% of daily ATR already used)")
    else:
        print(f"      R:R       {_fmt(p.rr, '.2f')} : 1   [{verdict}]")
    print(f"      Size      {_fmt(p.shares_for_1pct_account, '.0f')} sh "
          f"to risk {cfg.risk_per_trade_pct:g}% of account "
          f"(${_fmt(p.risk_per_share)}/sh risk)")
    print_size_overlay(res, cfg)
    for n in p.notes:
        print(f"      ! {n}")
    print_catalyst_block(res)


def print_diagnostics(rejected: List[Factors], cfg: ScreenerConfig) -> None:
    """Why the screen came back thin. Usually the most informative panel."""
    if not rejected:
        return
    print()
    print(_rule())
    print("  DIAGNOSTICS — why candidates were excluded")
    print(_rule())

    tally: Dict[str, int] = {}
    no_data = 0
    for f in rejected:
        if not f.filters_passed:
            no_data += 1
            continue
        for k, v in f.filters_passed.items():
            if not v:
                tally[k] = tally.get(k, 0) + 1

    if no_data:
        print(f"    {no_data} ticker(s) excluded for missing/insufficient data")
    for k, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>3} failed {k}")
    print()
    print("    An empty or thin result set is a NORMAL outcome. The conjunction")
    print("    of RVOL≥2x, above-VWAP, RSI 55–70 and a fresh MACD cross is rare")
    print("    by construction. Re-run with --relax to see the full ranking.")


def print_footer() -> None:
    print()
    print(_rule("═"))
    print("  This output ranks CURRENT technical conditions. It does not forecast")
    print("  returns. The composite score is an unvalidated heuristic with hand-set")
    print("  weights and no backtest behind it. Screening for high RVOL and high")
    print("  ATR reliably increases outcome variance; whether it increases expected")
    print("  return is unknown and untested here. Paper-trade before risking capital.")
    print()
    print("  The macro panel measures stress ALREADY PRICED by markets — it sizes")
    print("  risk, it does not forecast. Headlines are used only as a gap-risk veto;")
    print("  no sentiment is scored, because a scraper cannot know what is priced in.")
    print(_rule("═"))
    print()


def print_dashboard(
    results: List[ScreenResult],
    rejected: List[Factors],
    context: Dict[str, Factors],
    cfg: ScreenerConfig,
    timeframe: str,
    universe_n: int,
    relax: bool,
    top_n: int = 3,
    macro_regime=None,
    geo_headlines=None,
) -> None:
    """The single entry point for rendering everything to the terminal."""
    print_header(cfg, timeframe, universe_n, relax)
    print_context(context)
    print_macro_panel(macro_regime)

    if not results:
        print()
        print("  NO CANDIDATES matched the screen.")
    else:
        qualified = sum(1 for r in results if r.passed_all)
        print(f"  {len(results)} ranked candidate(s); "
              f"{qualified} passed every hard filter.")
        for i, res in enumerate(results[:top_n], 1):
            print_candidate(i, res, cfg)

    if geo_headlines:
        print_geo_headlines(geo_headlines)
    print_diagnostics(rejected, cfg)
    print_footer()


def results_to_json(results: List[ScreenResult]) -> List[dict]:
    """Serialise for logging / downstream analysis.

    Log every run. Without a persistent record of what the screen said and what
    subsequently happened, you can never evaluate whether it works — and that
    evaluation is the only thing that would justify trading it.
    """
    out = []
    for r in results:
        d = asdict(r.factors)
        d["plan"] = asdict(r.plan) if r.plan else None
        d["passed_all"] = r.passed_all
        d["timestamp"] = now_market().isoformat()

        # The risk overlay MUST be in the record. `plan.shares_for_1pct_account`
        # is the pre-overlay number; what you would actually have traded is
        # `adjusted_shares`, after the macro regime and catalyst multipliers.
        # Logging only the pre-overlay figure means a later review reconstructs
        # a position you never took, and cannot tell whether the sizing overlay
        # helped or hurt — which is the one question a run log exists to answer.
        d["size_multiplier"] = r.size_multiplier
        d["adjusted_shares"] = r.adjusted_shares
        d["required_rr"] = r.required_rr
        cat = r.catalyst
        d["catalyst"] = None if cat is None else {
            "risk_level": cat.risk_level,
            "n_relevant": cat.n_relevant,
            "newest_age_hours": cat.newest_age_hours,
            "categories": list(cat.categories),
            "size_multiplier": cat.size_multiplier,
            "rr_requirement_multiplier": cat.rr_requirement_multiplier,
            "warnings": list(cat.warnings),
            # Titles + source + link only, matching macro_news.py's policy of
            # never persisting article bodies.
            "headlines": [
                {"title": h.title, "source": h.source, "url": h.url,
                 "age_hours": h.age_hours, "categories": list(h.categories)}
                for h in cat.headlines
            ],
        }
        out.append(d)
    return out


# ==============================================================================
# SECTION 10 — SELF TEST (offline; no network required)
# ==============================================================================

def _synthetic_intraday(days: int = 12, seed: int = 7) -> pd.DataFrame:
    """Build deterministic 5m regular-hours bars for testing."""
    rng = np.random.default_rng(seed)
    frames = []
    price = 100.0
    start_day = pd.Timestamp("2025-01-06", tz=MARKET_TZ)  # a Monday
    d = 0
    made = 0
    while made < days:
        day = start_day + pd.Timedelta(days=d)
        d += 1
        if day.weekday() >= 5:
            continue
        made += 1
        idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                            periods=78, freq="5min", tz=MARKET_TZ)
        rets = rng.normal(0.0002, 0.0025, len(idx))
        closes = price * np.cumprod(1 + rets)
        highs = closes * (1 + np.abs(rng.normal(0, 0.0012, len(idx))))
        lows = closes * (1 - np.abs(rng.normal(0, 0.0012, len(idx))))
        opens = np.concatenate([[closes[0]], closes[:-1]])
        vols = rng.integers(40_000, 90_000, len(idx)).astype(float)
        frames.append(pd.DataFrame(
            {"Open": opens, "High": np.maximum(highs, np.maximum(opens, closes)),
             "Low": np.minimum(lows, np.minimum(opens, closes)),
             "Close": closes, "Volume": vols}, index=idx))
        price = closes[-1]
    return pd.concat(frames)


def _synthetic_daily(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-08-01", periods=n, tz=MARKET_TZ)
    rets = rng.normal(0.0005, 0.018, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(2_000_000, 6_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": np.maximum(high, np.maximum(open_, close)),
                         "Low": np.minimum(low, np.minimum(open_, close)),
                         "Close": close, "Volume": vol}, index=idx)


def run_selftest() -> int:
    """Verify indicator math against known cases. Returns process exit code."""
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("\n  SELF TEST — indicator correctness (offline)\n" + "  " + "-" * 60)

    # --- RSI boundary behaviour ---------------------------------------------
    up = pd.Series(np.arange(1, 60, dtype=float))          # monotonic up
    down = pd.Series(np.arange(60, 1, -1, dtype=float))    # monotonic down
    r_up, r_dn = rsi(up).iloc[-1], rsi(down).iloc[-1]
    check("RSI = 100 on monotonic advance", abs(r_up - 100.0) < 1e-6, f"got {r_up}")
    check("RSI = 0 on monotonic decline", r_dn < 1e-6, f"got {r_dn}")

    flat = pd.Series([50.0] * 60)
    check("RSI on flat series is not NaN-crashing",
          math.isfinite(rsi(flat).iloc[-1]) or math.isnan(rsi(flat).iloc[-1]))

    # --- True Range gap awareness -------------------------------------------
    gap_df = pd.DataFrame({
        "High":  [10.0, 20.0],
        "Low":   [ 9.0, 19.0],
        "Close": [ 9.5, 19.5],
    })
    tr = true_range(gap_df["High"], gap_df["Low"], gap_df["Close"]).iloc[-1]
    # prev close 9.5, high 20 -> |20-9.5| = 10.5 dominates H-L = 1.0
    check("True Range captures gaps", abs(tr - 10.5) < 1e-9, f"got {tr}")

    # --- ATR sanity ----------------------------------------------------------
    d = _synthetic_daily()
    a = atr(d, 14).iloc[-1]
    check("ATR is positive and finite", math.isfinite(a) and a > 0, f"got {a}")
    check("ATR is a sane fraction of price", 0 < a / d['Close'].iloc[-1] < 0.25)

    # --- MACD crossover detection -------------------------------------------
    # V-shaped series: forced down then forced up => guaranteed bullish cross.
    v = pd.Series(np.concatenate([np.linspace(100, 80, 60), np.linspace(80, 110, 40)]))
    m = macd(v)
    bsc = bars_since_bullish_cross(m["hist"])
    check("MACD bullish cross detected on V-reversal", bsc is not None, "none found")
    check("MACD cross is in the recovery leg",
          bsc is not None and bsc < 40, f"bars_since={bsc}")
    mono = macd(pd.Series(np.linspace(100, 50, 100)))
    check("No bullish cross on monotonic decline",
          bars_since_bullish_cross(mono["hist"]) is None)

    # --- VWAP ---------------------------------------------------------------
    idx = pd.date_range("2025-01-06 09:30", periods=4, freq="5min", tz=MARKET_TZ)
    vd = pd.DataFrame({"High": [10, 10, 10, 10], "Low": [10, 10, 10, 10],
                       "Close": [10, 10, 10, 10], "Volume": [100, 100, 100, 100]},
                      index=idx).astype(float)
    check("VWAP of constant price equals that price",
          abs(session_vwap(vd).iloc[-1] - 10.0) < 1e-9)

    # VWAP must reset across sessions.
    two = _synthetic_intraday(days=2)
    vw = session_vwap(two)
    day2 = pd.Series(two.index.date, index=two.index)
    first_of_day2 = vw.loc[day2 == sorted(day2.unique())[1]].iloc[0]
    open_of_day2 = two.loc[day2 == sorted(day2.unique())[1]]
    tp = (open_of_day2["High"].iloc[0] + open_of_day2["Low"].iloc[0]
          + open_of_day2["Close"].iloc[0]) / 3
    check("VWAP resets each session", abs(first_of_day2 - tp) < 1e-6,
          f"{first_of_day2} vs {tp}")

    # --- Time-adjusted RVOL --------------------------------------------------
    base = _synthetic_intraday(days=11, seed=11)
    rv, today_v, base_v = time_adjusted_rvol(base, 10)
    check("RVOL near 1.0 for homogeneous synthetic volume",
          math.isfinite(rv) and 0.7 < rv < 1.4, f"got {rv:.3f}")

    # Triple today's volume -> RVOL should roughly triple.
    boosted = base.copy()
    dts = pd.Series(boosted.index.date, index=boosted.index)
    last_day = sorted(dts.unique())[-1]
    boosted.loc[dts == last_day, "Volume"] *= 3.0
    rv2, _, _ = time_adjusted_rvol(boosted, 10)
    check("RVOL scales with volume", abs(rv2 / rv - 3.0) < 0.01,
          f"{rv:.3f} -> {rv2:.3f}")

    # Partial day must NOT depress RVOL (the bug this design exists to avoid).
    partial = base.loc[
        ~((dts == last_day) & pd.Series([t.time() > dtime(11, 0)
                                         for t in base.index], index=base.index))
    ]
    rv3, _, _ = time_adjusted_rvol(partial, 10)
    check("RVOL is time-of-day neutral (partial session)",
          math.isfinite(rv3) and 0.7 < rv3 < 1.4, f"got {rv3:.3f} at 11:00")

    # --- Resampling ----------------------------------------------------------
    r15 = resample_ohlcv(base, "15min")
    check("15m resample yields ~1/3 the bars",
          abs(len(r15) - len(base) / 3) <= 3, f"{len(base)} -> {len(r15)}")
    check("Resample preserves total volume",
          abs(r15["Volume"].sum() - base["Volume"].sum()) < 1.0)

    # --- Scoring monotonicity / shape ---------------------------------------
    check("RVOL score increases with RVOL", _score_rvol(4) > _score_rvol(2) > _score_rvol(1))
    check("RVOL score saturates", _score_rvol(50) - _score_rvol(10) < 0.15)
    check("ATR score peaks mid-range, penalises extremes",
          _score_atr(5) > _score_atr(1) and _score_atr(5) > _score_atr(25))
    check("VWAP score zero at/below VWAP", _score_vwap(-0.5) == 0.0 and _score_vwap(0) == 0.0)
    check("VWAP score penalises extension", _score_vwap(0.4) > _score_vwap(4.0))
    check("RSI score peaks at band midpoint",
          _score_rsi(62.5, 55, 70) > _score_rsi(56, 55, 70) > _score_rsi(50, 55, 70))
    check("MACD score rewards freshness", _score_macd(0, 0.1, 6) > _score_macd(5, 0.1, 6))
    check("MACD score zero without cross", _score_macd(None, 0.1, 6) == 0.0)

    # --- End-to-end factor pipeline -----------------------------------------
    f = compute_factors("TEST", _synthetic_intraday(days=12), _synthetic_daily(),
                        ScreenerConfig(), "5m")
    check("compute_factors completes on synthetic data", f.ok, f.reason)
    check("all key factors finite",
          all(math.isfinite(x) for x in (f.last, f.rvol, f.atr_pct, f.vwap, f.rsi)))

    f = score_factors(f, ScreenerConfig())
    check("score in [0,100]", 0.0 <= f.score <= 100.0, f"got {f.score}")

    plan = build_risk_plan(f, ScreenerConfig())
    check("stop is strictly below entry", plan.stop < plan.entry)
    check("target is strictly above entry", plan.target > plan.entry)
    check("R:R is finite and positive", math.isfinite(plan.rr) and plan.rr > 0)

    # --- REGRESSION: daily calendar dates must not shift a day ---------------
    # This bug shipped once. A naive daily index localized to UTC then converted
    # to New York moves every bar back one calendar day, which silently makes
    # "previous close" resolve to TODAY's close and corrupts every gap figure.
    naive_daily = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
         "Close": [1.0, 2.0], "Volume": [1.0, 1.0]},
        index=pd.to_datetime(["2026-08-06", "2026-08-07"]),
    )
    fixed = _to_market_tz(naive_daily.copy(), naive_is_calendar_date=True)
    check("daily calendar dates survive tz conversion",
          [d.date().isoformat() for d in fixed.index] == ["2026-08-06", "2026-08-07"],
          f"got {[d.date().isoformat() for d in fixed.index]}")
    wrong = _to_market_tz(naive_daily.copy(), naive_is_calendar_date=False)
    check("the UTC assumption is what shifts dates (documents the bug)",
          wrong.index[0].date().isoformat() == "2026-08-05")

    # --- REGRESSION: gap uses PRIOR close, never today's close ---------------
    intr = _synthetic_intraday(days=12)
    dly = _synthetic_daily()
    # Force the daily series to align its last date with the intraday last date.
    last_day = sorted(pd.Series(intr.index.date, index=intr.index).unique())[-1]
    dly2 = dly.copy()
    new_idx = list(dly2.index[:-1]) + [pd.Timestamp(last_day, tz=MARKET_TZ)]
    dly2.index = pd.DatetimeIndex(new_idx)
    ff = compute_factors("GAPTEST", intr, dly2, ScreenerConfig(), "5m")
    check("prev_close differs from today's daily close",
          ff.prev_close != float(dly2["Close"].iloc[-1]),
          "prev_close picked up today's own bar")
    expected_gap = (ff.day_open - ff.prev_close) / ff.prev_close * 100.0
    check("gap matches (open - prev_close)/prev_close",
          abs(ff.gap_pct - expected_gap) < 1e-9)

    # --- Premarket volume is NaN, never a misleading 0 ----------------------
    zero_pm = intr.copy()
    check("absent premarket reported as unknown, not zero",
          not ff.premarket_available and math.isnan(ff.premarket_volume))

    # --- REGRESSION: position size was unbounded by notional ----------------
    # Risk sizing scales as 1/stop_distance. A setup hugging VWAP produced 2,272
    # shares of a $100 stock — $227k, 9x a $25k account — with no note printed.
    tight = Factors(ticker="TIGHT")
    tight.last, tight.vwap, tight.swing_low = 100.0, 99.95, 99.90
    tight.atr_intraday, tight.atr_daily = 0.02, 3.0
    tight.day_high, tight.day_low, tight.swing_high = 100.2, 99.5, 104.0
    tp = build_risk_plan(tight, ScreenerConfig(), account_size=25_000.0)
    check("tight stop no longer implies leveraged size",
          tp.notional <= 25_000.0 + 1e-6,
          f"${tp.notional:,.0f} notional on a $25,000 account")
    check("notional cap is flagged, not applied silently", tp.notional_capped)
    check("the uncapped request is still recorded for review",
          tp.shares_uncapped > tp.shares_for_1pct_account,
          f"{tp.shares_uncapped} vs {tp.shares_for_1pct_account}")
    check("capping explains itself in the notes",
          any("capped by notional" in n for n in tp.notes))

    # A normal-width stop must be untouched by the cap.
    wide = Factors(ticker="WIDE")
    wide.last, wide.vwap, wide.swing_low = 100.0, 97.0, 96.0
    wide.atr_intraday, wide.atr_daily = 0.5, 3.0
    wide.day_high, wide.day_low, wide.swing_high = 101.0, 96.5, 106.0
    wp = build_risk_plan(wide, ScreenerConfig(), account_size=25_000.0)
    check("ordinary stop distance is not capped", not wp.notional_capped)
    check("uncapped size equals risk-derived size",
          wp.shares_for_1pct_account == wp.shares_uncapped)
    check("risk-derived size still risks ~the configured %",
          abs(wp.shares_for_1pct_account * wp.risk_per_share - 125.0) <= 5.0,
          f"${wp.shares_for_1pct_account * wp.risk_per_share:.2f} vs $125")

    # --- REGRESSION: stale-tail trim deleted the extended session ------------
    # Yahoo zero-fills extended-hours volume. An unrestricted walk-back removed
    # every premarket bar, destroying the premarket range the screener reports.
    # 04:00-09:00 inclusive — entirely before the 09:30 open.
    pm_idx = pd.date_range("2025-01-06 04:00", periods=11, freq="30min",
                           tz=MARKET_TZ)
    pm_only = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                            "Close": 100.5, "Volume": 0.0}, index=pm_idx)
    check("zero-volume premarket bars are preserved",
          len(_drop_stale_tail(pm_only)) == len(pm_only),
          f"{len(pm_only)} in -> {len(_drop_stale_tail(pm_only))} out")

    # ...while a still-forming REGULAR-HOURS placeholder is still trimmed.
    rth_idx = pd.date_range("2025-01-06 09:30", periods=5, freq="5min",
                            tz=MARKET_TZ)
    rth_tail = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                             "Close": 100.5, "Volume": [10.0, 10.0, 10.0, 0.0, 0.0]},
                            index=rth_idx)
    check("still-forming regular-hours bars are still trimmed",
          len(_drop_stale_tail(rth_tail)) == 3,
          f"got {len(_drop_stale_tail(rth_tail))}")

    # --- REGRESSION: JSON log dropped the risk overlay ------------------------
    # The pre-overlay share count is not the position you would have taken.
    rec = ScreenResult(factors=score_factors(
        compute_factors("TEST", _synthetic_intraday(days=12), _synthetic_daily(),
                        ScreenerConfig(), "5m"), ScreenerConfig()))
    rec.plan = build_risk_plan(rec.factors, ScreenerConfig())
    rec.size_multiplier, rec.adjusted_shares, rec.required_rr = 0.375, 12.0, 2.25
    js = results_to_json([rec])[0]
    check("JSON records the applied size multiplier",
          js.get("size_multiplier") == 0.375)
    check("JSON records the shares actually implied after the overlay",
          js.get("adjusted_shares") == 12.0)
    check("JSON records the raised R:R bar", js.get("required_rr") == 2.25)
    check("JSON round-trips through the serialiser",
          json.loads(json.dumps(js, default=str))["ticker"] == "TEST")

    # --- Degenerate inputs ---------------------------------------------------
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    bad = compute_factors("EMPTY", empty, empty, ScreenerConfig(), "5m")
    check("empty input rejected cleanly, no exception", not bad.ok)

    print("  " + "-" * 60)
    if failures:
        print(f"  {len(failures)} FAILURE(S): {', '.join(failures)}\n")
        return 1
    print("  All checks passed.\n")
    return 0


# ==============================================================================
# SECTION 11 — CLI
# ==============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Intraday momentum screener (RVOL / ATR / VWAP / MACD / RSI).",
        epilog="Screens current conditions. Does not predict returns. Not advice.",
    )
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated universe override, e.g. NVDA,AMD,TSLA")
    p.add_argument("--timeframe", choices=["5m", "15m", "30m", "1h"], default="5m",
                   help="Bar size for MACD/RSI (default: 5m)")
    p.add_argument("--top", type=int, default=3, help="Candidates to display")
    p.add_argument("--relax", action="store_true",
                   help="Rank all tickers; treat filters as advisory")
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--min-rvol", type=float, default=None)
    p.add_argument("--require-gap", action="store_true",
                   help="Make the gap-up threshold a hard filter")
    p.add_argument("--account", type=float, default=25_000.0,
                   help="Account size for position-size suggestion")
    p.add_argument("--json", type=str, default=None, help="Write results to JSON")
    p.add_argument("--live", action="store_true",
                   help="Continuously refresh until Ctrl-C")
    p.add_argument("--interval", type=int, default=60,
                   help="Live refresh interval in seconds (default 60)")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Stop live mode after N cycles (for testing)")
    p.add_argument("--no-news", action="store_true",
                   help="Disable the news layer (macro regime still runs)")
    p.add_argument("--no-macro", action="store_true",
                   help="Disable macro + news entirely; pure technical screen")
    p.add_argument("--selftest", action="store_true",
                   help="Run offline indicator tests and exit")
    p.add_argument("--quiet", action="store_true", help="Suppress fetch progress")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.selftest:
        return run_selftest()

    cfg = ScreenerConfig()
    if args.min_price is not None:
        cfg.min_price = args.min_price
    if args.min_rvol is not None:
        cfg.min_rvol = args.min_rvol
    if args.require_gap:
        cfg.require_gap = True

    universe = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else list(DEFAULT_UNIVERSE)
    )
    # Always include context ETFs so the regime line can render.
    for t in CONTEXT_TICKERS:
        if t not in universe:
            universe.append(t)

    verbose = not args.quiet
    try:
        provider = YFinanceProvider(cfg, verbose=verbose)
    except ImportError:
        print("yfinance is not installed. Run: pip install yfinance", file=sys.stderr)
        return 2

    enable_macro = _HAS_MACRO and not args.no_macro
    enable_news = enable_macro and not args.no_news
    if args.no_macro is False and not _HAS_MACRO:
        print("  [info] macro_news.py not found — running pure technical screen",
              file=sys.stderr)

    if args.live:
        return run_live(
            universe, cfg, provider, args.timeframe, args.top, args.relax,
            args.account, args.interval, enable_news,
            max_iterations=args.max_iterations, verbose=verbose,
        )

    cache = LiveCache()
    if enable_macro:
        refresh_macro_layer(cache, universe, enable_news, verbose=verbose)

    try:
        results, rejected, context = run_screen(
            universe, cfg, provider,
            timeframe=args.timeframe, relax=args.relax,
            account_size=args.account, verbose=verbose,
        )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    apply_risk_overlay(
        results, cfg,
        macro_regime=cache.macro_regime,
        ticker_news=cache.ticker_news if enable_news else None,
        account_size=args.account,
    )

    n_ranked = len([t for t in universe if t not in CONTEXT_TICKERS])
    print_dashboard(results, rejected, context, cfg, args.timeframe,
                    n_ranked, args.relax, top_n=args.top,
                    macro_regime=cache.macro_regime,
                    geo_headlines=cache.geo_headlines if enable_news else None)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results_to_json(results), fh, indent=2, default=str)
        print(f"  wrote {args.json}\n")

    return 0



# ==============================================================================
# SECTION 12 — MACRO / GEOPOLITICAL DASHBOARD PANELS
# ==============================================================================

def print_macro_panel(regime) -> None:
    """Market-priced geopolitical / macro stress.

    This panel answers "what is the market ALREADY pricing", which is a far more
    reliable read on geopolitics than any headline scraper. If gold, crude, bond
    vol and defense equities are all bid together, that IS the geopolitical
    signal — and it is continuous, timestamped, and free of interpretation.
    """
    if regime is None:
        return
    print()
    print(_rule())
    if not math.isfinite(regime.stress_index):
        print("  MACRO / GEOPOLITICAL REGIME — unavailable")
        for n in regime.notes:
            print(f"    ! {n}")
        print(_rule())
        return

    idx = regime.stress_index
    meter = _bar(idx / 100.0, 20)
    # Denominator is the number of SCORED gauges, derived rather than hardcoded.
    # It read "/6" against 7 scored instruments, understating the denominator and
    # making breadth look stronger than it was — and it would silently drift
    # again the moment MACRO_INSTRUMENTS changed.
    n_gauges = len(mn.scored_instruments()) if _HAS_MACRO else 0
    print(f"  MACRO / GEOPOLITICAL STRESS   {meter}  {idx:.1f}/100")
    print(f"  Regime: {regime.stress_label:<38} "
          f"confirming gauges: {regime.breadth}/{n_gauges}")
    if regime.size_multiplier < 1.0:
        print(f"  → position size scaled to {regime.size_multiplier:.0%} "
              f"of normal while this regime persists")
    print("  " + "-" * (W - 4))

    for sym, r in regime.readings.items():
        if not r.available:
            print(f"    {r.label:<22} unavailable")
            continue
        z = _fmt(r.zscore, "+.2f")
        pct = _fmt(r.percentile, ".0f")
        chg = _fmt(r.change_pct, "+.2f")
        # Levels are shown raw; ratios (defense) are small decimals.
        lvl = f"{r.last:,.4f}" if r.last < 1 else f"{r.last:,.2f}"
        marker = " ←" if (math.isfinite(r.zscore) and abs(r.zscore) >= 2.0) else ""
        print(f"    {r.label:<22} {lvl:>10}  {chg:>7}%  "
              f"z={z:>6}  pct={pct:>3}{marker}")
    print()
    print("    z = how unusual TODAY'S move is (vs 60d).  "
          "pct = where the LEVEL sits (0-100).")
    print("    Gauges marked ← are 2+ sigma moves. A high index on ONE gauge is")
    print("    usually idiosyncratic; genuine geopolitical stress shows up in")
    print("    several unrelated markets at once, which is what breadth counts.")
    for n in regime.notes:
        print(f"    ! {n}")
    print(_rule())


def print_geo_headlines(headlines, limit: int = 8) -> None:
    """Broad geopolitical headline panel. CONTEXT ONLY — deliberately unscored.

    Every item here is displayed as title + source + age. No sentiment is
    computed and nothing on this panel influences the ranking or the trade
    plans. See macro_news.py's module docstring for why: a headline scraper
    cannot assess whether news is already priced in, which is the only question
    that matters for trading it.
    """
    if not headlines:
        return
    print()
    print(_rule())
    print("  GEOPOLITICAL HEADLINES (last 24h)  —  context only, not scored")
    print(_rule())

    tally = mn.summarize_geo_categories(headlines) if _HAS_MACRO else {}
    if tally:
        print("  Themes: " + "  ".join(f"{k}={v}" for k, v in tally.items()))
        print()

    for h in headlines[:limit]:
        age = f"{h.age_hours:4.1f}h" if math.isfinite(h.age_hours) else "  ? "
        cats = ",".join(h.categories)[:18]
        title = h.title if len(h.title) <= 58 else h.title[:55] + "..."
        print(f"    {age}  [{h.source[:12]:<12}] {cats:<18} {title}")

    print()
    print("    Keyword classification is crude: it cannot handle negation")
    print("    ('sanctions LIFTED' tags identically to 'sanctions IMPOSED'),")
    print("    nor metaphor ('price war'). Treat tags as a rough index, and")
    print("    read the source before acting on anything here.")
    print(_rule())


def print_catalyst_block(res: ScreenResult) -> None:
    """Per-candidate news-risk block, printed inside the candidate card."""
    cat = res.catalyst
    if cat is None:
        return
    print("    " + "-" * (W - 8))
    if cat.n_relevant == 0:
        print("    NEWS      no ticker-relevant headlines in 48h")
        return

    badge = {"high": "⚠ HIGH", "moderate": "◐ MODERATE", "none": "· low"}.get(
        cat.risk_level, cat.risk_level
    )
    age = (f"{cat.newest_age_hours:.1f}h" if math.isfinite(cat.newest_age_hours)
           else "?")
    print(f"    NEWS RISK {badge}   {cat.n_relevant} relevant headline(s), "
          f"newest {age}")
    if cat.categories:
        print(f"              themes: {', '.join(cat.categories)}")
    for h in cat.headlines[:3]:
        a = f"{h.age_hours:4.1f}h" if math.isfinite(h.age_hours) else "  ? "
        t = h.title if len(h.title) <= 50 else h.title[:47] + "..."
        print(f"              {a} [{h.source[:12]:<12}] {t}")
    for wmsg in cat.warnings:
        print(f"      ! {wmsg}")


def print_size_overlay(res: ScreenResult, cfg: ScreenerConfig) -> None:
    """Show how macro + catalyst risk changed the position size."""
    if res.size_multiplier >= 0.999 or not math.isfinite(res.adjusted_shares):
        return
    base = res.plan.shares_for_1pct_account if res.plan else float("nan")
    print(f"      Size adj  {_fmt(base, '.0f')} sh → "
          f"{_fmt(res.adjusted_shares, '.0f')} sh "
          f"({res.size_multiplier:.0%} of normal: macro regime x news risk)")
    if math.isfinite(res.required_rr) and res.required_rr > cfg.min_acceptable_rr:
        print(f"      R:R bar   raised {cfg.min_acceptable_rr:.2f} → "
              f"{res.required_rr:.2f} by catalyst risk")


# ==============================================================================
# SECTION 13 — LIVE MODE
# ==============================================================================

@dataclass
class LiveCache:
    """Holds slow-moving data between fast price refreshes.

    Prices, macro data and news change on completely different timescales, and
    every feed here is rate-limited. Re-pulling company names and RSS feeds on a
    30-second price loop would get the client throttled or blocked within an
    hour, and would add nothing: RSS feeds update in minutes, not seconds.
    """

    macro_regime: object = None
    macro_fetched: float = 0.0
    geo_headlines: list = field(default_factory=list)
    geo_fetched: float = 0.0
    ticker_news: Dict[str, list] = field(default_factory=dict)
    news_fetched: float = 0.0
    company_names: Dict[str, str] = field(default_factory=dict)
    names_fetched: float = 0.0

    macro_ttl: float = 300.0     # 5 min
    geo_ttl: float = 600.0       # 10 min
    news_ttl: float = 300.0      # 5 min
    names_ttl: float = 86_400.0  # 24 h — company names essentially never change

    def stale(self, which: str) -> bool:
        now = time.time()
        return {
            "macro": now - self.macro_fetched > self.macro_ttl,
            "geo": now - self.geo_fetched > self.geo_ttl,
            "news": now - self.news_fetched > self.news_ttl,
            "names": now - self.names_fetched > self.names_ttl,
        }[which]


def refresh_macro_layer(
    cache: LiveCache,
    tickers: Sequence[str],
    enable_news: bool,
    verbose: bool = True,
) -> None:
    """Refresh macro + news into the cache, respecting TTLs.

    Every branch is individually guarded. A dead RSS feed, a throttled news
    endpoint or a macro download failure degrades that one panel and leaves the
    price screen fully functional. Nothing here is allowed to raise.
    """
    if not _HAS_MACRO:
        return

    if cache.stale("macro"):
        if verbose:
            print("  refreshing macro regime...", file=sys.stderr)
        try:
            cache.macro_regime = mn.fetch_macro_regime(verbose=verbose)
            cache.macro_fetched = time.time()
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [warn] macro refresh failed: {type(exc).__name__}",
                      file=sys.stderr)

    if not enable_news:
        return

    if cache.stale("geo"):
        if verbose:
            print("  refreshing geopolitical headlines...", file=sys.stderr)
        try:
            cache.geo_headlines = mn.fetch_geopolitical_headlines()
            cache.geo_fetched = time.time()
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [warn] geo feed failed: {type(exc).__name__}",
                      file=sys.stderr)

    if cache.stale("names"):
        try:
            cache.company_names = mn.get_company_names(tickers)
            cache.names_fetched = time.time()
        except Exception:  # noqa: BLE001
            pass

    if cache.stale("news"):
        if verbose:
            print(f"  refreshing headlines for {len(tickers)} tickers...",
                  file=sys.stderr)
        fresh: Dict[str, list] = {}
        for t in tickers:
            try:
                fresh[t] = mn.fetch_ticker_headlines(
                    t, cache.company_names.get(t, "")
                )
            except Exception:  # noqa: BLE001
                fresh[t] = []
        cache.ticker_news = fresh
        cache.news_fetched = time.time()


def run_live(
    universe: List[str],
    cfg: ScreenerConfig,
    provider: DataProvider,
    timeframe: str,
    top_n: int,
    relax: bool,
    account_size: float,
    interval: int,
    enable_news: bool,
    max_iterations: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """Continuously re-run the screen until interrupted.

    A WORD ON WHAT "LIVE" MEANS HERE. This loop refreshes on a timer. It does
    NOT make the underlying data real-time — yfinance quotes remain delayed, and
    polling faster does not change that. A tighter interval gets you the same
    stale data more often, plus a higher chance of being rate-limited.

    If you need genuinely real-time prices you need a streaming vendor feed
    (Polygon, Alpaca, Databento, IBKR). Subclass DataProvider and the rest of
    this file works unchanged — that is what the abstraction is for.
    """
    cache = LiveCache()
    iteration = 0
    n_ranked = len([t for t in universe if t not in CONTEXT_TICKERS])

    print("\n  Starting live mode. Ctrl-C to stop.\n", file=sys.stderr)
    try:
        while True:
            iteration += 1
            cycle_start = time.time()

            refresh_macro_layer(cache, universe, enable_news, verbose=verbose)

            results, rejected, context = run_screen(
                universe, cfg, provider, timeframe=timeframe, relax=relax,
                account_size=account_size, verbose=verbose,
            )
            apply_risk_overlay(
                results, cfg,
                macro_regime=cache.macro_regime,
                ticker_news=cache.ticker_news if enable_news else None,
                account_size=account_size,
            )

            # ANSI clear + home. Harmless if the terminal ignores it.
            print("\033[2J\033[H", end="")
            print_dashboard(
                results, rejected, context, cfg, timeframe, n_ranked, relax,
                top_n=top_n, macro_regime=cache.macro_regime,
                geo_headlines=cache.geo_headlines if enable_news else None,
            )

            elapsed = time.time() - cycle_start
            print(f"  cycle {iteration} completed in {elapsed:.1f}s   "
                  f"next refresh in {interval}s   (Ctrl-C to stop)")
            if elapsed > interval:
                print(f"  ! a cycle takes longer ({elapsed:.0f}s) than the "
                      f"{interval}s interval — raise --interval to avoid "
                      f"back-to-back requests and rate limiting")

            if max_iterations and iteration >= max_iterations:
                return 0
            time.sleep(max(1.0, interval - elapsed))

    except KeyboardInterrupt:
        print("\n\n  Live mode stopped.\n", file=sys.stderr)
        return 0


# ==============================================================================
# ENTRY POINT — must stay LAST. `main()` references LiveCache and run_live,
# which are defined in the macro/live sections above it; invoking main() any
# earlier in the file raises NameError at import time.
# ==============================================================================

if __name__ == "__main__":
    sys.exit(main())
