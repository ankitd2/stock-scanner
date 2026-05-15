"""
analytics/indicators.py — Technical indicator helpers.

compute_all(df) -> dict with keys:
    rsi_14, latest_rsi, above_50dma, above_200dma, golden_cross,
    pct_from_52wh, pct_from_52wl, vol_adj_mom, bb_width, volume_ratio, latest_close
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def rsi(prices, period: int = 14) -> float:
    """Compute RSI for a price series. Returns float or None."""
    if not _AVAILABLE:
        return None
    try:
        s = pd.Series(prices).dropna()
        if len(s) < period + 1:
            return None
        delta = s.diff().dropna()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))
    except Exception:
        return None


def sma(prices, window: int) -> float:
    """Simple moving average of last n values."""
    if not _AVAILABLE:
        return None
    try:
        s = pd.Series(prices).dropna()
        if len(s) < window:
            return None
        return float(s.tail(window).mean())
    except Exception:
        return None


def ema(prices, span: int) -> float:
    """Exponential moving average."""
    if not _AVAILABLE:
        return None
    try:
        s = pd.Series(prices).dropna()
        if len(s) < 2:
            return None
        return float(s.ewm(span=span).mean().iloc[-1])
    except Exception:
        return None


def compute_all(df: "pd.DataFrame") -> dict:
    """
    Compute all technical indicators from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with at minimum a 'Close' column. Optionally 'Volume'.
        Index should be date-ordered (oldest first).

    Returns
    -------
    dict with keys:
        rsi_14        : pd.Series of RSI values (full series)
        latest_rsi    : float — most recent RSI(14) value
        above_50dma   : bool — close > 50-day SMA
        above_200dma  : bool — close > 200-day SMA
        golden_cross  : bool — 50dma > 200dma
        pct_from_52wh : float — % from 52-week high (negative = below high)
        pct_from_52wl : float — % above 52-week low (positive = above low)
        vol_adj_mom   : float — volatility-adjusted 12-1 month momentum
        bb_width      : float — Bollinger Band width (20d, 2 std) as fraction
        volume_ratio  : float — today's volume / 20d avg volume (None if no Volume col)
        latest_close  : float — most recent closing price
    """
    result = {
        "rsi_14": None,
        "latest_rsi": None,
        "above_50dma": False,
        "above_200dma": False,
        "golden_cross": False,
        "pct_from_52wh": None,
        "pct_from_52wl": None,
        "vol_adj_mom": None,
        "bb_width": None,
        "volume_ratio": None,
        "latest_close": None,
    }

    if not _AVAILABLE or df is None or df.empty:
        return result

    try:
        close = df["Close"].dropna()
        if len(close) < 20:
            return result

        latest = float(close.iloc[-1])
        result["latest_close"] = latest

        # --- RSI(14) ---
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(com=13, min_periods=14).mean()
            avg_loss = loss.ewm(com=13, min_periods=14).mean()
            # Avoid division-by-zero
            rs = avg_gain / avg_loss.replace(0, float("nan"))
            rsi_series = 100 - 100 / (1 + rs)
            result["rsi_14"] = rsi_series
            result["latest_rsi"] = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

        # --- Moving averages ---
        sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
        result["above_50dma"] = bool(sma50 is not None and latest > sma50)
        result["above_200dma"] = bool(sma200 is not None and latest > sma200)
        result["golden_cross"] = bool(
            sma50 is not None and sma200 is not None and sma50 > sma200
        )

        # --- 52-week range ---
        year_close = close.tail(252) if len(close) >= 252 else close
        hi52 = float(year_close.max())
        lo52 = float(year_close.min())
        result["pct_from_52wh"] = float((latest - hi52) / hi52 * 100) if hi52 else None
        result["pct_from_52wl"] = float((latest - lo52) / lo52 * 100) if lo52 else None

        # --- Vol-adjusted momentum (Barroso-Santa-Clara 2015) ---
        # 12-1 month return (roughly 252 days ago to 21 days ago)
        if len(close) >= 252:
            price_12m_ago = float(close.iloc[-252])
            price_1m_ago = float(close.iloc[-21])
            raw_mom = (price_1m_ago - price_12m_ago) / price_12m_ago
            # Realized volatility over same window (daily returns std)
            returns = close.iloc[-252:-1].pct_change().dropna()
            realized_vol = float(returns.std()) * (252 ** 0.5)  # annualised
            if realized_vol > 0:
                result["vol_adj_mom"] = float(raw_mom / realized_vol)
            else:
                result["vol_adj_mom"] = float(raw_mom)
        elif len(close) >= 63:
            # Use 3-month as fallback
            price_3m_ago = float(close.iloc[-63])
            price_1m_ago = float(close.iloc[-21]) if len(close) >= 21 else latest
            raw_mom = (price_1m_ago - price_3m_ago) / price_3m_ago
            returns = close.iloc[-63:-1].pct_change().dropna()
            realized_vol = float(returns.std()) * (252 ** 0.5) if len(returns) > 1 else 0
            if realized_vol > 0:
                result["vol_adj_mom"] = float(raw_mom / realized_vol)
            else:
                result["vol_adj_mom"] = float(raw_mom)

        # --- Bollinger Band width (20d, 2 std) ---
        if len(close) >= 20:
            rolling_mean = close.tail(20).mean()
            rolling_std = close.tail(20).std()
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
            if rolling_mean != 0:
                result["bb_width"] = float((upper_band - lower_band) / rolling_mean)

        # --- Volume ratio (today vs 20d avg) ---
        if "Volume" in df.columns:
            vol = df["Volume"].dropna()
            if len(vol) >= 20:
                today_vol = float(vol.iloc[-1])
                avg_vol_20d = float(vol.tail(20).mean())
                if avg_vol_20d > 0:
                    result["volume_ratio"] = float(today_vol / avg_vol_20d)

    except Exception:
        pass

    return result
