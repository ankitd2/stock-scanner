"""
data/fred.py — FRED macroeconomic time series fetcher.

Uses pandas-datareader to pull series from the Federal Reserve Economic Data
(FRED) API.  No API key required for most series.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

# Guard the import: pandas_datareader may be broken on Python 3.14+ installs.
# Tests inject a stub into sys.modules before importing this module, so this
# try/except lets both production and test code work without raising at import.
try:
    import pandas_datareader.data as web  # type: ignore[import]
except Exception as _pdr_exc:  # noqa: BLE001
    print(
        f"[fred] WARNING: could not import pandas_datareader — {_pdr_exc}",
        file=sys.stderr,
    )
    web = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Series registry
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "hy_oas": "BAMLH0A0HYM2",  # BofA HY option-adjusted spread (bps)
    "t10y2y": "T10Y2Y",         # 10Y-2Y Treasury spread (pct)
    "dgs10":  "DGS10",          # 10-Year Treasury yield (pct)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fred_series(
    series_ids: dict[str, str] = FRED_SERIES,
    lookback_days: int = 400,
) -> dict[str, pd.Series]:
    """
    Fetch each FRED series for the last ``lookback_days`` calendar days.

    Returns
    -------
    dict[str, pd.Series]
        Mapping of friendly name → pd.Series with a DatetimeIndex.
        Values are floats.  NaN rows are dropped.  Series are sorted
        ascending by date (most-recent value last).
        Returns ``{}`` on any failure and prints a warning to stderr.
    """
    if web is None:
        print("[fred] WARNING: pandas_datareader unavailable; returning {}", file=sys.stderr)
        return {}

    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    result: dict[str, pd.Series] = {}

    for name, fred_id in series_ids.items():
        try:
            raw = web.DataReader(fred_id, "fred", start, end)
            series = raw[fred_id].dropna().sort_index()
            result[name] = series
        except Exception as exc:  # noqa: BLE001
            print(
                f"[fred] WARNING: could not fetch '{fred_id}' ({name}): {exc}",
                file=sys.stderr,
            )

    return result


def latest_fred(series_ids: dict[str, str] = FRED_SERIES) -> dict[str, float]:
    """
    Return the most-recent value for each series.

    Returns
    -------
    dict[str, float]
        ``{name: float}`` — only series that were successfully fetched are
        included.  Returns ``{}`` if all fetches fail.
    """
    all_series = get_fred_series(series_ids)
    return {
        name: float(series.iloc[-1])
        for name, series in all_series.items()
        if len(series) > 0
    }


def fred_zscore(series: pd.Series, window: int = 252) -> float | None:
    """
    Z-score of the most-recent observation vs the trailing ``window`` obs.

    Formula: ``(last - mean) / std`` over the most-recent ``window`` rows.

    Returns
    -------
    float or None
        ``None`` if there are fewer than ``window // 2`` observations or if
        the standard deviation is zero.
    """
    if series is None or len(series) < window // 2:
        return None

    tail = series.iloc[-window:]
    mean = float(tail.mean())
    std = float(tail.std())

    if std == 0:
        return None

    return (float(series.iloc[-1]) - mean) / std
