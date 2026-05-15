"""
analytics/indicators.py — Technical indicator helpers.
"""

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
