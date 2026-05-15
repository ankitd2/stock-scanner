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

# AAII's CDN (Imperva) returns HTTP 403 / a JS-challenge page to requests
# with the default ``python-requests/...`` User-Agent or with a UA-only
# header set. Sending a richer browser-like header set (UA, Accept,
# Accept-Language, Referer, and the Sec-Fetch-* hints) is what actually
# gets us the real XLS file consistently. Tested 2026-05-15.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.aaii.com/sentimentsurvey",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
}


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


def _looks_like_sentiment_header(cols: list) -> bool:
    """Return True if a column list contains date/bull/bear/neutral markers."""
    lowered = [str(c).lower().strip() for c in cols]
    has_date = any("date" in c for c in lowered)
    has_bull = any("bullish" in c for c in lowered)
    has_bear = any("bearish" in c for c in lowered)
    has_neut = any("neutral" in c for c in lowered)
    return has_date and has_bull and has_bear and has_neut


def _parse_response(content: bytes, text: str) -> pd.DataFrame | None:
    """
    Attempt to parse the raw AAII response.  Tries Excel first, then CSV.
    Returns a raw DataFrame or None on failure.

    The real AAII workbook has a few metadata rows above the column
    headers (e.g. "American Association of Individual Investors ...")
    and the actual header row sits a few rows down. We probe a handful
    of plausible header row indices and pick the first one that exposes
    date / bullish / bearish / neutral columns.
    """
    # Candidate (engine, sheet_name) attempts. ``None`` sheet_name lets
    # pandas pick the first sheet (helpful when the workbook has no
    # explicit "SENTIMENT" sheet).
    excel_attempts = [
        ("xlrd", "SENTIMENT"),
        ("xlrd", None),
        ("openpyxl", "SENTIMENT"),
        ("openpyxl", None),
    ]

    for engine, sheet in excel_attempts:
        for header_row in (0, 1, 2, 3, 4):
            try:
                kwargs = {"engine": engine, "header": header_row}
                if sheet is not None:
                    kwargs["sheet_name"] = sheet
                df = pd.read_excel(io.BytesIO(content), **kwargs)
                if df is None or df.empty:
                    continue
                if _looks_like_sentiment_header(list(df.columns)):
                    return df
            except Exception:  # noqa: BLE001
                continue

    # Fallback: take the first successful Excel parse even if header
    # detection didn't find a sentiment match — `_extract_sentiment_columns`
    # will surface a clear warning instead of us silently returning None.
    for engine, sheet in excel_attempts:
        try:
            kwargs = {"engine": engine}
            if sheet is not None:
                kwargs["sheet_name"] = sheet
            df = pd.read_excel(io.BytesIO(content), **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception:  # noqa: BLE001
            continue

    # --- final attempt: CSV ---
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
        resp = requests.get(AAII_URL, headers=_HEADERS, timeout=_TIMEOUT)
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
