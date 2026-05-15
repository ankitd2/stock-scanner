"""
data/aaii.py — AAII Investor Sentiment Survey fetcher (stub / real implementation).

latest_aaii() → {bullish, bearish, neutral, bull_bear_spread, bull_bear_zscore_5y}
"""

from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")

# AAII publishes a CSV at this URL (updated weekly)
_AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"


def latest_aaii() -> Dict[str, Optional[float]]:
    """
    Fetch the latest AAII sentiment survey.
    Returns bullish/bearish/neutral percentages and the bull-bear spread z-score.
    Falls back gracefully if the fetch fails.
    """
    result: Dict[str, Optional[float]] = {
        "bullish": None,
        "bearish": None,
        "neutral": None,
        "bull_bear_spread": None,
        "bull_bear_zscore_5y": None,
    }
    try:
        import pandas as pd
        import numpy as np
        import requests
        from io import BytesIO

        resp = requests.get(_AAII_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_excel(BytesIO(resp.content), skiprows=3, engine="xlrd")
        # Expected columns: Date, Bullish, Neutral, Bearish, ...
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.dropna(subset=["bullish", "bearish"])
        df = df[pd.to_numeric(df["bullish"], errors="coerce").notna()]
        df["bullish"] = pd.to_numeric(df["bullish"], errors="coerce")
        df["bearish"] = pd.to_numeric(df["bearish"], errors="coerce")
        df["neutral"] = pd.to_numeric(df.get("neutral", pd.Series(dtype=float)), errors="coerce")
        df = df.tail(260)  # ~5 years of weekly data

        if df.empty:
            return result

        latest = df.iloc[-1]
        bullish = float(latest["bullish"])
        bearish = float(latest["bearish"])
        neutral = float(latest.get("neutral", 0.0)) if not pd.isna(latest.get("neutral", float("nan"))) else None

        spread = bullish - bearish
        spread_series = df["bullish"] - df["bearish"]
        mean = float(spread_series.mean())
        std = float(spread_series.std())
        zscore = float((spread - mean) / std) if std > 0 else 0.0

        result.update({
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "bull_bear_spread": spread,
            "bull_bear_zscore_5y": zscore,
        })
    except Exception:
        pass
    return result
