"""
data/fred.py — FRED economic data fetcher (stub / real implementation).

latest_fred() → {hy_oas: float, t10y2y: float, dgs10: float}
fred_zscore(series, window) → float
get_fred_series() → {name: pd.Series}
"""

from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")

# FRED series IDs
_SERIES = {
    "hy_oas": "BAMLH0A0HYM2",   # ICE BofA US HY OAS
    "t10y2y": "T10Y2Y",          # 10Y - 2Y Treasury spread
    "dgs10":  "DGS10",           # 10-Year Treasury Rate
}

_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&vintage_date={date}"


def _fetch_series(series_id: str, limit_rows: int = 300) -> Optional["pd.Series"]:
    """Fetch a FRED series via public CSV endpoint (no API key required)."""
    try:
        import pandas as pd
        import requests
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), parse_dates=[0], index_col=0)
        df.columns = ["value"]
        s = pd.to_numeric(df["value"], errors="coerce").dropna()
        return s.tail(limit_rows)
    except Exception:
        return None


def latest_fred() -> Dict[str, Optional[float]]:
    """Return latest values for HY OAS, 10Y-2Y spread, and 10Y Treasury yield."""
    result: Dict[str, Optional[float]] = {"hy_oas": None, "t10y2y": None, "dgs10": None}
    for key, series_id in _SERIES.items():
        s = _fetch_series(series_id)
        if s is not None and not s.empty:
            result[key] = float(s.iloc[-1])
    return result


def fred_zscore(series: "pd.Series", window: int = 260) -> float:
    """Compute z-score of the latest value vs trailing window."""
    try:
        import numpy as np
        tail = series.dropna().tail(window)
        if len(tail) < 20:
            return 0.0
        mean = float(tail.mean())
        std = float(tail.std())
        if std == 0:
            return 0.0
        return float((tail.iloc[-1] - mean) / std)
    except Exception:
        return 0.0


def get_fred_series() -> Dict[str, "pd.Series"]:
    """Return all FRED series as a dict of pd.Series."""
    result = {}
    for key, series_id in _SERIES.items():
        s = _fetch_series(series_id)
        if s is not None:
            result[key] = s
    return result
