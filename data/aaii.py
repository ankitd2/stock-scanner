"""
data/aaii.py — AAII Investor Sentiment Survey fetcher.

AAII publishes a weekly sentiment CSV (XLS-named) at a public URL.
The file is used as a contrarian indicator: ±2σ extremes in the
bull-bear spread are actionable signals.
"""

import io
import sys
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"

_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(columns: list[str], keyword: str) -> str | None:
    """
    Case-insensitive, whitespace-stripped column lookup.
    Returns the first column name whose lower-stripped form contains ``keyword``.
    """
    keyword_lower = keyword.lower()
    for col in columns:
        if keyword_lower in str(col).lower().strip():
            return col
    return None


def _parse_response(content: bytes, text: str) -> pd.DataFrame | None:
    """
    Attempt to parse the raw AAII response.  Tries Excel first, then CSV.
    Returns a raw DataFrame or None on failure.
    """
    # --- attempt 1: Excel ---
    try:
        df = pd.read_excel(io.BytesIO(content), engine="xlrd")
        if df is not None and not df.empty:
            return df
    except Exception:  # noqa: BLE001
        pass

    # --- attempt 2: openpyxl (xlsx) ---
    try:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        if df is not None and not df.empty:
            return df
    except Exception:  # noqa: BLE001
        pass

    # --- attempt 3: CSV ---
    try:
        df = pd.read_csv(io.StringIO(text))
        if df is not None and not df.empty:
            return df
    except Exception:  # noqa: BLE001
        pass

    return None


def _extract_sentiment_columns(raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    From a raw DataFrame (unknown column names), extract date, bullish,
    bearish, neutral columns using flexible name matching.

    Returns a clean DataFrame with those four columns or None if any
    required column cannot be located.
    """
    cols = list(raw.columns)

    date_col = _find_column(cols, "date")
    bull_col = _find_column(cols, "bullish")
    bear_col = _find_column(cols, "bearish")
    neut_col = _find_column(cols, "neutral")

    missing = [
        name for name, col in [
            ("date", date_col),
            ("bullish", bull_col),
            ("bearish", bear_col),
            ("neutral", neut_col),
        ]
        if col is None
    ]
    if missing:
        print(
            f"[aaii] WARNING: could not locate columns {missing} in raw data. "
            f"Available columns: {cols[:10]}",
            file=sys.stderr,
        )
        return None

    out = raw[[date_col, bull_col, bear_col, neut_col]].copy()
    out.columns = ["date", "bullish", "bearish", "neutral"]

    # Coerce types
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("bullish", "bearish", "neutral"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop rows with missing date or all-NaN sentiment
    out = out.dropna(subset=["date"])
    out = out.dropna(subset=["bullish", "bearish", "neutral"], how="all")

    # Normalise to 0-100 percentages if stored as 0-1 fractions
    for col in ("bullish", "bearish", "neutral"):
        if out[col].dropna().max() <= 1.0:
            out[col] = out[col] * 100

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_aaii_sentiment(n_weeks: int = 260) -> pd.DataFrame:
    """
    Fetch the last ``n_weeks`` of AAII weekly sentiment data.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, bullish, bearish, neutral, bull_bear_spread]``.
        All percentage values in 0-100 range.  Sorted ascending by date.
        Returns an empty DataFrame on any failure.
    """
    empty = pd.DataFrame(
        columns=["date", "bullish", "bearish", "neutral", "bull_bear_spread"]
    )

    try:
        resp = requests.get(AAII_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[aaii] WARNING: HTTP request failed: {exc}",
            file=sys.stderr,
        )
        return empty

    raw = _parse_response(resp.content, resp.text)
    if raw is None:
        print("[aaii] WARNING: failed to parse AAII response as Excel or CSV.", file=sys.stderr)
        return empty

    clean = _extract_sentiment_columns(raw)
    if clean is None:
        return empty

    clean = clean.sort_values("date").reset_index(drop=True)
    clean["bull_bear_spread"] = clean["bullish"] - clean["bearish"]

    # Keep only the most-recent n_weeks rows
    clean = clean.tail(n_weeks).reset_index(drop=True)

    return clean


def latest_aaii() -> dict:
    """
    Return the most-recent AAII sentiment week as a dictionary.

    Returns
    -------
    dict
        Keys: ``date`` (ISO string), ``bullish``, ``bearish``, ``neutral``,
        ``bull_bear_spread``, ``bull_bear_zscore_5y``.
        Returns ``{}`` on failure.
    """
    df = get_aaii_sentiment(n_weeks=260)

    if df.empty:
        return {}

    last = df.iloc[-1]

    # Z-score of most-recent bull_bear_spread vs full 5-year (260-week) window
    spread_series = df["bull_bear_spread"].dropna()
    if len(spread_series) >= 10:
        mean = float(spread_series.mean())
        std = float(spread_series.std())
        zscore = (float(last["bull_bear_spread"]) - mean) / std if std != 0 else 0.0
    else:
        zscore = 0.0

    return {
        "date": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
        "bullish": float(last["bullish"]),
        "bearish": float(last["bearish"]),
        "neutral": float(last["neutral"]),
        "bull_bear_spread": float(last["bull_bear_spread"]),
        "bull_bear_zscore_5y": round(zscore, 4),
    }
