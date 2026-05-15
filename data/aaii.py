"""
data/aaii.py — AAII Investor Sentiment Survey fetcher.

AAII publishes a weekly sentiment CSV (XLS-named) at a public URL.
The file is used as a contrarian indicator: ±2σ extremes in the
bull-bear spread are actionable signals.

Fetch chain:
1. XLS endpoint (https://www.aaii.com/files/surveys/sentiment.xls)
   — gives full multi-year history; preferred when reachable.
2. HTML scrape fallback (https://www.aaii.com/sentimentsurvey/sent_results)
   — single most-recent week only; used when the XLS endpoint serves
   an Imperva JS-challenge page or otherwise fails to parse.

Both endpoints sit behind Imperva, so either can return a "Pardon Our
Interruption" HTML challenge page even with a browser User-Agent. The
fallback chain still tries both — if the HTML results page is also
blocked we accept the failure, return an empty DataFrame, and emit a
single warning (cached per-process so callers that invoke us twice in
the same run don't double-log).
"""

import io
import re
import sys
from datetime import date
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
AAII_HTML_URL = "https://www.aaii.com/sentimentsurvey/sent_results"

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

# Imperva "Pardon Our Interruption" challenge page markers. Any of these
# in a response body means we got blocked, not the real data.
_IMPERVA_MARKERS = (
    "Pardon Our Interruption",
    "window.onProtectionInitialized",
    "reeseSkipExpirationCheck",
)

# Per-process cache so we don't spam stderr when callers invoke
# get_aaii_sentiment / latest_aaii multiple times in a single run.
# Keys: "warn:<message>" — values: True once logged.
_WARN_CACHE: dict[str, bool] = {}


def _warn_once(message: str) -> None:
    """Emit ``message`` to stderr, but only the first time per process."""
    key = f"warn:{message}"
    if _WARN_CACHE.get(key):
        return
    _WARN_CACHE[key] = True
    print(message, file=sys.stderr)


def _looks_like_imperva(text: str) -> bool:
    """Heuristic: does the response body look like an Imperva JS-challenge page?"""
    if not text:
        return False
    # Only sniff the first ~4KB to keep this cheap.
    head = text[:4096]
    return any(marker in head for marker in _IMPERVA_MARKERS)


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
        _warn_once(
            f"[aaii] WARNING: could not locate columns {missing} in raw data. "
            f"Available columns: {cols[:10]}"
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
# HTML scrape fallback
# ---------------------------------------------------------------------------

# Match a label (Bullish/Bearish/Neutral) followed by anything up to the
# next percent number. Non-greedy. The labels on the live page appear
# both as the section heading ("Bullish") and inline with the value
# ("Bullish 39.32%") OR split across markup ("<span>Bullish</span>
# <span>39.32%</span>"). The pattern allows up to ~200 chars of any
# content (including tags / whitespace / punctuation) between the label
# and the percentage — non-greedy so the closest number wins.
_PCT_PATTERN = r"{label}\b[^\d]{{0,200}}?(\d{{1,3}}(?:\.\d{{1,3}})?)\s*%"


def _extract_pct(html: str, label: str) -> float | None:
    """Find the first ``<label> ... NN.NN%`` match in ``html``."""
    pattern = _PCT_PATTERN.format(label=label)
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _extract_date(html: str) -> str | None:
    """
    Try to find a survey date in the HTML. AAII typically labels weekly
    results with a "Week of <Mon DD, YYYY>" or "Reported Date: ..." line.
    Returns ISO-formatted date string or None.
    """
    # Pattern 1: "Week of November 6, 2025" / "Week Ending Nov 6, 2025"
    # Use a non-letter, non-digit separator class so the [A-Za-z]+ month
    # name isn't gobbled up by the in-between match.
    m = re.search(
        r"(?:Week\s+(?:of|Ending)|Reported(?:\s+Date)?)[^A-Za-z\d]{0,30}"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        html, flags=re.IGNORECASE,
    )
    if m:
        try:
            return pd.to_datetime(m.group(1)).strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            pass

    # Pattern 2: ISO date "2025-11-06"
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", html)
    if m:
        return m.group(1)

    return None


def _fetch_aaii_html_fallback() -> dict | None:
    """
    Scrape the publicly visible AAII sentiment results page.

    Returns
    -------
    dict | None
        ``{date, bullish, bearish, neutral}`` (single most-recent week)
        or None if the page is unreachable / blocked by Imperva / lacks
        the expected sentiment numbers.
    """
    try:
        resp = requests.get(AAII_HTML_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"[aaii] WARNING: HTML fallback HTTP failed: {exc}")
        return None

    text = resp.text or ""
    if _looks_like_imperva(text):
        # Both AAII endpoints are behind Imperva. From some networks
        # (e.g. corporate proxies, GitHub Actions IPs) both the XLS and
        # the HTML results page are served as JS-challenge pages. There
        # is no clean way around this without paid services, so we
        # accept the limitation and fail cleanly.
        _warn_once(
            "[aaii] WARNING: HTML fallback also returned Imperva challenge page; "
            "AAII data unavailable from this network."
        )
        return None

    bullish = _extract_pct(text, "bullish")
    bearish = _extract_pct(text, "bearish")
    neutral = _extract_pct(text, "neutral")

    if bullish is None or bearish is None or neutral is None:
        _warn_once(
            "[aaii] WARNING: HTML fallback page did not contain expected "
            "Bullish/Bearish/Neutral percentages."
        )
        return None

    iso_date = _extract_date(text) or date.today().strftime("%Y-%m-%d")

    return {
        "date": iso_date,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_aaii_sentiment(n_weeks: int = 260) -> pd.DataFrame:
    """
    Fetch the last ``n_weeks`` of AAII weekly sentiment data.

    Tries the XLS endpoint first. If that returns an Imperva JS-challenge
    page or fails to parse as Excel/CSV, falls back to scraping the
    public HTML results page (which gives only the most-recent week).

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

    # --- Primary: XLS endpoint -------------------------------------------
    xls_text = ""
    xls_content = b""
    xls_ok = False
    try:
        resp = requests.get(AAII_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        xls_content = resp.content or b""
        xls_text = resp.text or ""
        xls_ok = True
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"[aaii] WARNING: HTTP request failed: {exc}")

    if xls_ok:
        # If the "XLS" body is actually an Imperva challenge HTML page,
        # skip the parse attempts and jump straight to the HTML fallback.
        if _looks_like_imperva(xls_text):
            _warn_once(
                "[aaii] WARNING: XLS endpoint served an Imperva challenge page; "
                "trying HTML scrape fallback."
            )
        else:
            raw = _parse_response(xls_content, xls_text)
            if raw is not None:
                clean = _extract_sentiment_columns(raw)
                if clean is not None:
                    clean = clean.sort_values("date").reset_index(drop=True)
                    clean["bull_bear_spread"] = clean["bullish"] - clean["bearish"]
                    clean = clean.tail(n_weeks).reset_index(drop=True)
                    return clean
            _warn_once(
                "[aaii] WARNING: failed to parse AAII response as Excel or CSV; "
                "trying HTML scrape fallback."
            )

    # --- Fallback: scrape the public HTML results page -------------------
    fallback = _fetch_aaii_html_fallback()
    if fallback is None:
        return empty

    df = pd.DataFrame([{
        "date": pd.to_datetime(fallback["date"], errors="coerce"),
        "bullish": float(fallback["bullish"]),
        "bearish": float(fallback["bearish"]),
        "neutral": float(fallback["neutral"]),
    }])
    df = df.dropna(subset=["date"])
    if df.empty:
        return empty
    df["bull_bear_spread"] = df["bullish"] - df["bearish"]
    df.attrs["source"] = "html_fallback"  # Hint to latest_aaii() for z-score handling
    return df.reset_index(drop=True)


def latest_aaii() -> dict:
    """
    Return the most-recent AAII sentiment week as a dictionary.

    Returns
    -------
    dict
        Keys: ``date`` (ISO string), ``bullish``, ``bearish``, ``neutral``,
        ``bull_bear_spread``, ``bull_bear_zscore_5y``.
        If only the single-week HTML scrape succeeded, ``bull_bear_zscore_5y``
        is set to 0.0 and a ``note`` field documents the limitation.
        Returns ``{}`` on failure.
    """
    df = get_aaii_sentiment(n_weeks=260)

    if df.empty:
        return {}

    last = df.iloc[-1]
    is_html_fallback = df.attrs.get("source") == "html_fallback" or len(df) < 10

    # Z-score of most-recent bull_bear_spread vs full 5-year (260-week) window.
    # If we only have the single-week HTML scrape we cannot compute a real
    # z-score — fall back to 0 (neutral) with a documenting note.
    spread_series = df["bull_bear_spread"].dropna()
    note: str | None = None
    if len(spread_series) >= 10:
        mean = float(spread_series.mean())
        std = float(spread_series.std())
        zscore = (float(last["bull_bear_spread"]) - mean) / std if std != 0 else 0.0
    else:
        zscore = 0.0
        if is_html_fallback:
            note = "z-score unavailable from HTML scrape — using neutral baseline"

    result = {
        "date": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
        "bullish": float(last["bullish"]),
        "bearish": float(last["bearish"]),
        "neutral": float(last["neutral"]),
        "bull_bear_spread": float(last["bull_bear_spread"]),
        "bull_bear_zscore_5y": round(zscore, 4),
    }
    if note is not None:
        result["note"] = note
    return result
