"""
analytics/indicators.py

Technical indicators for the market intelligence scanner.
Pure numpy/pandas — no yfinance or network calls.
"""

import math
import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI. Returns series same length as input, NaN for first `period` values."""
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, dtype=float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothed moving average (EWM with adjust=False matches standard RSI)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100 - (100 / (1 + rs))

    # Force first `period` values to NaN to match docstring expectation
    result.iloc[:period] = np.nan
    return result


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram). All same length as input."""
    if close is None or len(close) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower). middle = SMA(period)."""
    if close is None or len(close) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def bollinger_width(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """(upper - lower) / middle. Normalized band width. Low = volatility compression."""
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)

    upper, middle, lower = bollinger(close, period=period, std_dev=std_dev)
    # Avoid division by zero
    width = (upper - lower) / middle.replace(0, np.nan)
    return width


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range. TR = max(H-L, |H-Cprev|, |L-Cprev|)."""
    if any(s is None or len(s) == 0 for s in (high, low, close)):
        return pd.Series(dtype=float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing (same as RSI)
    result = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    return result


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    return close.ewm(span=period, adjust=False).mean()


def pct_from_high(close: pd.Series, lookback: int = 252) -> pd.Series:
    """% below rolling `lookback`-period high. Negative = below. 0 = at high."""
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)

    rolling_high = close.rolling(window=lookback, min_periods=1).max()
    return (close / rolling_high - 1) * 100


def pct_from_low(close: pd.Series, lookback: int = 252) -> pd.Series:
    """% above rolling `lookback`-period low."""
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)

    rolling_low = close.rolling(window=lookback, min_periods=1).min()
    return (close / rolling_low - 1) * 100


def relative_strength(
    asset: pd.Series,
    benchmark: pd.Series,
    period: int = 63,
) -> float:
    """(asset_return_N - benchmark_return_N) where N=period days. Simple excess return."""
    if asset is None or benchmark is None:
        return float("nan")
    if len(asset) < period + 1 or len(benchmark) < period + 1:
        return float("nan")

    asset_return = asset.iloc[-1] / asset.iloc[-period - 1] - 1
    bench_return = benchmark.iloc[-1] / benchmark.iloc[-period - 1] - 1
    return float(asset_return - bench_return)


def vol_adjusted_momentum(
    close: pd.Series,
    skip_recent: int = 21,
    lookback: int = 252,
) -> Optional[float]:
    """
    Risk-adjusted momentum: (close[-skip_recent] / close[-lookback] - 1) / annualized_vol.
    annualized_vol = daily_returns.std() * sqrt(252) over the lookback window.
    Returns None if insufficient data. Used by Screen #3 (Barroso-Santa-Clara 2015).
    """
    if close is None or len(close) < lookback:
        return None

    # Slice the lookback window
    window = close.iloc[-lookback:]
    daily_returns = window.pct_change().dropna()

    if len(daily_returns) < 2:
        return None

    ann_vol = daily_returns.std() * math.sqrt(252)
    if ann_vol == 0 or math.isnan(ann_vol):
        return None

    # Momentum: return from lookback start to skip_recent ago
    start_price = close.iloc[-lookback]
    end_price = close.iloc[-skip_recent]
    momentum = end_price / start_price - 1

    return float(momentum / ann_vol)


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std over `window` periods."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    roll_mean = series.rolling(window=window, min_periods=window).mean()
    roll_std = series.rolling(window=window, min_periods=window).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# compute_all — aggregated convenience function
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns [Open, High, Low, Close, Volume],
    compute and return a dict with keys:
      rsi_14, macd_line, macd_signal, macd_hist,
      bb_upper, bb_middle, bb_lower, bb_width,
      atr_14, sma_20, sma_50, sma_200, ema_12, ema_26,
      pct_from_52wh, pct_from_52wl,
      vol_adj_mom,         # scalar float or None
      latest_rsi,          # scalar float (last RSI value)
      latest_close,        # scalar float
      above_50dma,         # bool: latest close > sma_50
      above_200dma,        # bool: latest close > sma_200
      golden_cross,        # bool: sma_50 > sma_200 (last value)
      volume_ratio,        # float: today's volume / 20d avg volume
    Returns {} if df has fewer than 30 rows.
    """
    try:
        if df is None or len(df) < 30:
            return {}

        # Normalise column names to title-case
        df = df.copy()
        df.columns = [c.title() for c in df.columns]

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return {}

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        # RSI
        rsi_14 = rsi(close, period=14)
        latest_rsi_val = float(rsi_14.dropna().iloc[-1]) if rsi_14.dropna().shape[0] > 0 else float("nan")

        # MACD
        macd_line, macd_signal, macd_hist = macd(close)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger(close)
        bb_width_series = bollinger_width(close)

        # ATR
        atr_14 = atr(high, low, close)

        # SMAs
        sma_20 = sma(close, 20)
        sma_50 = sma(close, 50)
        sma_200 = sma(close, 200)

        # EMAs
        ema_12 = ema(close, 12)
        ema_26 = ema(close, 26)

        # 52-week range metrics (252 trading days)
        pct_from_52wh = pct_from_high(close, lookback=252)
        pct_from_52wl = pct_from_low(close, lookback=252)

        # Vol-adjusted momentum
        vol_adj_mom_val = vol_adjusted_momentum(close)

        # Latest close
        latest_close_val = float(close.iloc[-1])

        # Derived booleans
        last_sma_50 = sma_50.iloc[-1]
        last_sma_200 = sma_200.iloc[-1]

        above_50dma = bool(latest_close_val > last_sma_50) if not math.isnan(last_sma_50) else False
        above_200dma = bool(latest_close_val > last_sma_200) if not math.isnan(last_sma_200) else False
        golden_cross = (
            bool(last_sma_50 > last_sma_200)
            if not math.isnan(last_sma_50) and not math.isnan(last_sma_200)
            else False
        )

        # Volume ratio: today's volume / 20-day average volume
        avg_vol_20 = volume.rolling(window=20, min_periods=1).mean().iloc[-1]
        volume_ratio = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else float("nan")

        return {
            "rsi_14": rsi_14,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "bb_width": bb_width_series,
            "atr_14": atr_14,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "pct_from_52wh": pct_from_52wh,
            "pct_from_52wl": pct_from_52wl,
            "vol_adj_mom": vol_adj_mom_val,
            "latest_rsi": latest_rsi_val,
            "latest_close": latest_close_val,
            "above_50dma": above_50dma,
            "above_200dma": above_200dma,
            "golden_cross": golden_cross,
            "volume_ratio": volume_ratio,
        }

    except Exception:
        return {}
