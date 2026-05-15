"""
data/fred.py — FRED macroeconomic time series fetcher.

Two fetch paths:

1. ``pandas_datareader.data.DataReader(..., "fred", ...)`` — full-featured
   but fragile across pandas/datareader version pairs. On the GHA runners
   we see ``deprecate_kwarg() missing 1 required positional argument:
   'new_arg_name'`` because pandas-datareader 0.10.0 calls a removed
   pandas internal. This is also why Python 3.14 installs fail locally.

2. Direct CSV download from ``https://fred.stlouisfed.org/graph/fredgraph.csv``
   — no API key, no extra deps, just ``requests`` + ``pd.read_csv``. Used
   automatically when pandas-datareader is unavailable or its DataReader
   call raises.

Warnings are de-duplicated per process so callers that invoke us twice
in a single run (scanner.py + analytics/market_state.py) don't double-
log the same failure.
"""

import io
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

# Guard the import: pandas_datareader can be broken on Python 3.14+ installs
# AND on Python 3.11 + pandas>=3.0 (where the `deprecate_kwarg` helper it
# imports has been removed). Tests inject a stub into sys.modules before
# importing this module, so this try/except lets both production and test
# code work without raising at import.
_PDR_IMPORT_ERROR: Exception | None = None
try:
    import pandas_datareader.data as web  # type: ignore[import]
except Exception as _pdr_exc:  # noqa: BLE001
    _PDR_IMPORT_ERROR = _pdr_exc
    web = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Series registry
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "hy_oas": "BAMLH0A0HYM2",  # BofA HY option-adjusted spread (bps)
    "t10y2y": "T10Y2Y",         # 10Y-2Y Treasury spread (pct)
    "dgs10":  "DGS10",          # 10-Year Treasury yield (pct)
}

# Public FRED CSV download endpoint — no API key required.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_TIMEOUT = 15  # seconds

# FRED's CDN actively rate-limits / silently times out requests that send a
# browser-style User-Agent (verified 2026-05-15: a Chrome UA gets read-timed
# out at 15s while a default python-requests UA or curl UA returns 200 in
# <200ms). So unlike AAII, we DO NOT spoof a browser here. A minimal Accept
# header is enough; we let requests send its default UA. Tested on macOS
# local + GitHub Actions Linux runners.
_FRED_HEADERS = {
    "Accept": "text/csv,text/plain,*/*;q=0.8",
}

# Per-process warning cache (see data/aaii.py for the same pattern).
_WARN_CACHE: dict[str, bool] = {}


def _warn_once(message: str) -> None:
    """Emit ``message`` to stderr, but only the first time per process."""
    key = f"warn:{message}"
    if _WARN_CACHE.get(key):
        return
    _WARN_CACHE[key] = True
    print(message, file=sys.stderr)


# ---------------------------------------------------------------------------
# Direct-CSV fallback
# ---------------------------------------------------------------------------

def _fetch_fred_direct(series_id: str, lookback_days: int = 400) -> pd.Series | None:
    """
    Fetch a single FRED series via the public fredgraph.csv endpoint.

    Returns
    -------
    pd.Series | None
        DatetimeIndex Series of floats (NaN rows dropped, sorted ascending),
        or ``None`` on any failure.
    """
    try:
        resp = requests.get(
            FRED_CSV_URL,
            params={"id": series_id},
            headers=_FRED_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _warn_once(
            f"[fred] WARNING: direct CSV fetch failed for '{series_id}': {exc}"
        )
        return None

    text = resp.text or ""
    if not text.strip():
        _warn_once(
            f"[fred] WARNING: direct CSV for '{series_id}' returned empty body"
        )
        return None

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        _warn_once(
            f"[fred] WARNING: could not parse direct CSV for '{series_id}': {exc}"
        )
        return None

    if df.empty or len(df.columns) < 2:
        _warn_once(
            f"[fred] WARNING: direct CSV for '{series_id}' has no usable rows"
        )
        return None

    # FRED CSVs are shaped: DATE,<series_id>
    # The data column is usually the second column. Use that to be robust
    # to header naming differences (some series ship as observation_date).
    date_col = df.columns[0]
    value_col = df.columns[1] if df.columns[1] != date_col else df.columns[-1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # FRED uses '.' for missing observations.
    df[value_col] = pd.to_numeric(
        df[value_col].replace(".", pd.NA), errors="coerce"
    )

    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if df.empty:
        return None

    # Trim to lookback window for parity with the pandas-datareader path.
    if lookback_days and lookback_days > 0:
        cutoff = datetime.today() - timedelta(days=lookback_days)
        df = df[df[date_col] >= cutoff]

    if df.empty:
        return None

    series = pd.Series(
        df[value_col].astype(float).values,
        index=df[date_col].values,
        name=series_id,
    )
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fred_series(
    series_ids: dict[str, str] = FRED_SERIES,
    lookback_days: int = 400,
) -> dict[str, pd.Series]:
    """
    Fetch each FRED series for the last ``lookback_days`` calendar days.

    Tries pandas-datareader first when available; falls back to the public
    fredgraph.csv endpoint on import failure or per-series fetch failure.

    Returns
    -------
    dict[str, pd.Series]
        Mapping of friendly name → pd.Series with a DatetimeIndex.
        Values are floats.  NaN rows are dropped.  Series are sorted
        ascending by date (most-recent value last).
        Returns ``{}`` if every series fetch fails.
    """
    if web is None:
        # pandas-datareader couldn't be imported at module load. Log the
        # underlying reason once (so CI logs explain the root cause once,
        # not the silent "unavailable" downstream message). Then fall
        # straight through to the direct CSV path.
        if _PDR_IMPORT_ERROR is not None:
            _warn_once(
                f"[fred] WARNING: pandas_datareader import failed "
                f"({type(_PDR_IMPORT_ERROR).__name__}: {_PDR_IMPORT_ERROR}); "
                f"falling back to direct CSV fetch"
            )
        else:
            _warn_once(
                "[fred] WARNING: pandas_datareader unavailable; "
                "falling back to direct CSV fetch"
            )

    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    result: dict[str, pd.Series] = {}

    for name, fred_id in series_ids.items():
        series: pd.Series | None = None

        if web is not None:
            try:
                raw = web.DataReader(fred_id, "fred", start, end)
                series = raw[fred_id].dropna().sort_index()
            except Exception as exc:  # noqa: BLE001
                _warn_once(
                    f"[fred] WARNING: pandas_datareader failed for "
                    f"'{fred_id}' ({name}): {exc}; trying direct CSV"
                )
                series = None

        if series is None or len(series) == 0:
            series = _fetch_fred_direct(fred_id, lookback_days=lookback_days)

        if series is not None and len(series) > 0:
            result[name] = series

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
