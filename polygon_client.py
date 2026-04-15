"""
Polygon.io API client — replaces yfinance throughout the codebase.

All data functions read the API key from the environment variable:
    export MASSIVE_API_KEY="o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"

Endpoints used (free tier, no options):
    /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}  — daily OHLCV
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ── API key resolution ────────────────────────────────────────────────
_DEFAULT_KEY = "o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"

def _api_key() -> str:
    return os.getenv("MASSIVE_API_KEY", _DEFAULT_KEY)


# ════════════════════════════════════════════════════════════════════
# Core fetch
# ════════════════════════════════════════════════════════════════════

def fetch_daily_ohlcv(
    ticker:  str,
    start:   str,        # "YYYY-MM-DD"
    end:     str,        # "YYYY-MM-DD"
    adjusted: bool = True,
    timeout:  int  = 30,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Polygon.io for `ticker` between
    `start` and `end` (inclusive).

    Returns a DataFrame indexed by date with columns:
        open, high, low, close, volume, vwap

    Handles Polygon's 50 000-row limit by paginating automatically.
    """
    key    = _api_key()
    url    = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort":     "asc",
        "limit":    50_000,
        "apiKey":   key,
    }

    rows: list[dict] = []
    while url:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")
        if status not in ("OK", "DELAYED"):
            raise RuntimeError(
                f"Polygon error for {ticker!r}: status={status!r}  "
                f"message={data.get('message','')}"
            )

        rows.extend(data.get("results", []))

        # Pagination (next_url replaces url + clears params)
        url    = data.get("next_url")
        params = {"apiKey": key} if url else {}

    if not rows:
        raise RuntimeError(
            f"Polygon returned no data for {ticker!r} "
            f"between {start} and {end}"
        )

    df = pd.DataFrame(rows)
    # Polygon timestamp is Unix ms → convert to date index
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap",
    })
    df.index = pd.DatetimeIndex(df.index)
    return df[["open", "high", "low", "close", "volume"]]


# ════════════════════════════════════════════════════════════════════
# Convenience wrappers
# ════════════════════════════════════════════════════════════════════

def fetch_close(ticker: str, start: str, end: str) -> pd.Series:
    """Return daily closing prices as a Series indexed by date."""
    df = fetch_daily_ohlcv(ticker, start, end)
    return df["close"].rename(ticker)


def fetch_log_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Return daily log-returns (first obs dropped)."""
    close = fetch_close(ticker, start, end)
    return np.log(close / close.shift(1)).dropna()


# ════════════════════════════════════════════════════════════════════
# Market IV proxy  (replaces spot VIX — not authorized on free tier)
# ════════════════════════════════════════════════════════════════════

def compute_trailing_hv(
    log_returns: pd.Series,
    window:      int   = 21,
    ann_factor:  float = 252.0,
) -> pd.Series:
    """
    21-day trailing realised volatility, annualised.

    This is the standard proxy for implied vol used in academic
    backtests when a live VIX feed is unavailable.

    σ_HV(t) = sqrt(252) · std(log_ret[t-window : t])

    Values are available from index `window` onward; earlier entries
    are NaN and are dropped by callers.
    """
    hv = log_returns.rolling(window).std() * np.sqrt(ann_factor)
    return hv.dropna()


def fetch_spy_with_hv(
    start:  str,
    end:    str,
    window: int = 21,
) -> tuple[pd.Series, pd.Series]:
    """
    Fetch SPY prices and compute trailing HV in one call.

    Returns
    -------
    close  : pd.Series  — daily SPY close prices
    hv     : pd.Series  — 21-day trailing annualised realised vol
             (proxy for market implied vol; NaN for first `window` days)
    """
    # Fetch extra history so rolling window is primed from `start`
    extra_start = (
        pd.Timestamp(start) - timedelta(days=int(window * 1.5))
    ).strftime("%Y-%m-%d")

    close = fetch_close("SPY", extra_start, end)
    log_ret = np.log(close / close.shift(1)).dropna()
    hv = compute_trailing_hv(log_ret, window=window)

    # Trim back to the requested window
    close = close[start:]
    hv    = hv[start:]

    return close, hv
