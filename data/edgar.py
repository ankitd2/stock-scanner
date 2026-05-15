"""
data/edgar.py — SEC EDGAR Form 4 (insider transaction) ingestion.

Pulls recent open-market insider purchase data and aggregates per-ticker
over a rolling window for use by Screen #7 (Insider Cluster Buys,
Lakonishok-Lee 2001).

Two data paths are supported:

  1. Primary: openinsider.com — an established aggregator that scrapes EDGAR
     Form 4 filings and serves them as plain HTML tables. Free, no key, fast
     (single HTTP request for the whole dataset). robots.txt allows polite
     scraping; we identify ourselves via a clear User-Agent.

  2. Fallback: SEC EDGAR full-index files. Slower (must fetch + parse one
     XML per filing) and rate-limited to ~10 req/sec by SEC policy, but is
     the canonical source. Only used if openinsider is unreachable.

Both paths fail gracefully — any exception returns ``{}`` with a stderr
warning so callers (Screen 7) can simply skip the screen.

SEC EDGAR notes
---------------
- Identify your tool via User-Agent: "Stock Scanner scanner@example.com"
- Max 10 requests/second
- CIK → ticker mapping: https://www.sec.gov/files/company_tickers.json
"""

import re
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_HEADERS = {
    # SEC requires identification; include a contact string.
    "User-Agent": "Stock Scanner scanner@stockscanner.example",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

OPENINSIDER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; StockScanner/1.0; "
        "+scanner@stockscanner.example)"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

EDGAR_BASE = "https://www.sec.gov"
OPENINSIDER_BASE = "http://openinsider.com"
CIK_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

RATE_LIMIT_DELAY = 0.11  # ~9 req/sec — safely under SEC's 10/sec cap
HTTP_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Module-level caches (populated lazily, reset per process)
# ---------------------------------------------------------------------------

_CIK_TICKER_MAP: Optional[dict[str, str]] = None  # cik (10-digit str) -> ticker
_PURCHASES_CACHE: dict[tuple, dict[str, list[dict]]] = {}


def _reset_caches() -> None:
    """Clear all module-level caches. Intended for tests."""
    global _CIK_TICKER_MAP
    _CIK_TICKER_MAP = None
    _PURCHASES_CACHE.clear()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _rate_limited_get(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = HTTP_TIMEOUT,
) -> Optional[requests.Response]:
    """
    GET with rate limiting + a single retry on 5xx. Returns ``None`` on any
    permanent failure (timeout, connection error, non-2xx after retry).
    """
    headers = headers or EDGAR_HEADERS
    try:
        time.sleep(RATE_LIMIT_DELAY)
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code >= 500:
            # one retry on transient server errors
            time.sleep(RATE_LIMIT_DELAY * 2)
            resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(
                f"[edgar] WARNING: GET {url} -> HTTP {resp.status_code}",
                file=sys.stderr,
            )
            return None
        return resp
    except Exception as exc:  # noqa: BLE001
        print(f"[edgar] WARNING: GET {url} failed — {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CIK / ticker mapping
# ---------------------------------------------------------------------------

def _load_cik_ticker_map() -> dict[str, str]:
    """
    Load the SEC's CIK → ticker mapping JSON. Cached at module level.

    Returns ``{}`` on failure (network/parse error). Keys are zero-padded
    10-digit CIK strings, values are uppercase ticker symbols.
    """
    global _CIK_TICKER_MAP
    if _CIK_TICKER_MAP is not None:
        return _CIK_TICKER_MAP

    resp = _rate_limited_get(
        CIK_TICKERS_URL,
        headers={**EDGAR_HEADERS, "Host": "www.sec.gov"},
    )
    if resp is None:
        _CIK_TICKER_MAP = {}
        return _CIK_TICKER_MAP

    try:
        data = resp.json()
        mapping: dict[str, str] = {}
        # File shape: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
        for entry in data.values():
            cik = entry.get("cik_str")
            ticker = entry.get("ticker")
            if cik is None or not ticker:
                continue
            cik_padded = str(cik).zfill(10)
            mapping[cik_padded] = str(ticker).upper()
        _CIK_TICKER_MAP = mapping
        return _CIK_TICKER_MAP
    except Exception as exc:  # noqa: BLE001
        print(
            f"[edgar] WARNING: could not parse CIK/ticker mapping — {exc}",
            file=sys.stderr,
        )
        _CIK_TICKER_MAP = {}
        return _CIK_TICKER_MAP


# ---------------------------------------------------------------------------
# EDGAR full-index path (canonical, slow)
# ---------------------------------------------------------------------------

def fetch_recent_form4_filings(lookback_days: int = 30) -> list[dict]:
    """
    Fetch a list of recent Form 4 filings from the EDGAR quarterly form index.

    Returns
    -------
    list of dict
        Each dict: {
          "ticker":      str | None,
          "cik":         str (zero-padded 10-digit),
          "company":     str,
          "filing_date": "YYYY-MM-DD",
          "accession":   str,
          "url":         str,
        }
        Filings older than ``lookback_days`` are excluded. Sorted newest-first.
        Returns ``[]`` on any error.
    """
    today = date.today()
    cutoff = today - timedelta(days=lookback_days)

    # Determine which quarters to scan (a 30d window may span a quarter boundary).
    quarters = _quarters_in_window(cutoff, today)
    cik_map = _load_cik_ticker_map()

    rows: list[dict] = []
    for year, quarter in quarters:
        url = f"{EDGAR_BASE}/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"
        resp = _rate_limited_get(url)
        if resp is None:
            continue
        try:
            for entry in _parse_form_idx(resp.text, form_type="4"):
                try:
                    fdate = datetime.strptime(entry["filing_date"], "%Y-%m-%d").date()
                except Exception:
                    continue
                if fdate < cutoff or fdate > today:
                    continue
                cik_padded = str(entry["cik"]).zfill(10)
                ticker = cik_map.get(cik_padded)
                rows.append({
                    "ticker": ticker,
                    "cik": cik_padded,
                    "company": entry["company"],
                    "filing_date": entry["filing_date"],
                    "accession": entry["accession"],
                    "url": entry["url"],
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[edgar] WARNING: parse form.idx — {exc}", file=sys.stderr)
            continue

    rows.sort(key=lambda r: r["filing_date"], reverse=True)
    return rows


def _quarters_in_window(start: date, end: date) -> list[tuple[int, int]]:
    """Yield (year, quarter) tuples that cover [start, end]."""
    out: list[tuple[int, int]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        q = (cur.month - 1) // 3 + 1
        pair = (cur.year, q)
        if pair not in out:
            out.append(pair)
        # advance one month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def _parse_form_idx(text: str, form_type: str = "4") -> list[dict]:
    """
    Parse the fixed-column SEC ``form.idx`` listing.

    The file has a header block then space-separated columns:
        Form Type | Company Name | CIK | Date Filed | Filename
    """
    out: list[dict] = []
    started = False
    for line in text.splitlines():
        if not started:
            if line.startswith("-----"):
                started = True
            continue
        if not line.strip():
            continue

        # Form type is the first whitespace-delimited token.
        parts = line.split(None, 1)
        if not parts or parts[0] != form_type:
            continue
        # Walk back from the right to get filename, date, cik.
        try:
            # The filename is the last whitespace-separated token.
            # Date is the second-to-last; cik is third-from-last (numeric).
            tokens = line.split()
            filename = tokens[-1]
            filed = tokens[-2]
            cik = tokens[-3]
            # Company name is whatever is between form_type and CIK.
            ft_end = line.find(parts[0]) + len(parts[0])
            cik_pos = line.rfind(cik)
            company = line[ft_end:cik_pos].strip()

            # Accession from filename: edgar/data/{cik}/{accession}-index.htm
            # Filenames look like "edgar/data/320193/0000320193-25-000123.txt"
            m = re.search(r"(\d{10}-\d{2}-\d{6})", filename)
            accession = m.group(1) if m else ""
            url = f"{EDGAR_BASE}/Archives/{filename}"

            out.append({
                "form_type": parts[0],
                "company": company,
                "cik": cik,
                "filing_date": filed,
                "accession": accession,
                "url": url,
            })
        except Exception:
            continue
    return out


def parse_form4_xml(accession: str, cik: str) -> Optional[dict]:
    """
    Fetch and parse a single Form 4 XML filing.

    Returns a dict with insider name/title and a list of ``transactions``
    (filtered to open-market purchases, transaction code "P"). Returns ``None``
    on any error.
    """
    if not accession or not cik:
        return None

    # Strip dashes from accession + leading zeros from CIK for the archive path:
    #   /Archives/edgar/data/{cik_no_leading_zeros}/{accession_nd}/{file}.xml
    accession_nd = accession.replace("-", "")
    try:
        cik_int = str(int(cik))
    except (TypeError, ValueError):
        return None

    # Fetch the filing index page first; it lists the file names in the filing.
    index_url = (
        f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{accession_nd}/"
        f"{accession}-index.htm"
    )
    resp = _rate_limited_get(index_url)
    if resp is None:
        return None

    # Look for an .xml link in the index page.
    m = re.search(r'href="([^"]+\.xml)"', resp.text, flags=re.IGNORECASE)
    if not m:
        return None
    xml_href = m.group(1)
    if xml_href.startswith("/"):
        xml_url = f"{EDGAR_BASE}{xml_href}"
    else:
        xml_url = (
            f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/"
            f"{accession_nd}/{xml_href}"
        )

    resp = _rate_limited_get(xml_url)
    if resp is None:
        return None

    try:
        return _parse_form4_xml_text(resp.text)
    except Exception as exc:  # noqa: BLE001
        print(f"[edgar] WARNING: parse Form 4 XML — {exc}", file=sys.stderr)
        return None


def _parse_form4_xml_text(text: str) -> Optional[dict]:
    """
    Parse a Form 4 XML body string. Namespaces are messy in EDGAR XBRL; we
    use ``local-name`` style matching by stripping any namespace prefix.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None

    def _ln(elem: ET.Element) -> str:
        # Strip XML namespace: "{ns}tag" -> "tag"
        tag = elem.tag
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def _direct_text(elem: ET.Element) -> Optional[str]:
        """Return the first non-empty text inside an element (depth-first)."""
        t = (elem.text or "").strip()
        if t:
            return t
        for sub in elem.iter():
            if sub is elem:
                continue
            t = (sub.text or "").strip()
            if t:
                return t
        return None

    def _find_text(node: ET.Element, name: str) -> Optional[str]:
        for child in node.iter():
            if _ln(child) == name:
                t = _direct_text(child)
                if t:
                    return t
        return None

    ticker = _find_text(root, "issuerTradingSymbol") or ""
    company = _find_text(root, "issuerName") or ""
    insider_name = _find_text(root, "rptOwnerName") or ""
    insider_title = _find_text(root, "officerTitle") or ""

    transactions: list[dict] = []
    # Both nonDerivativeTransaction (most common) and derivativeTransaction blocks.
    for txn in root.iter():
        ln = _ln(txn)
        if ln not in ("nonDerivativeTransaction", "derivativeTransaction"):
            continue
        date_val = _find_text(txn, "transactionDate")
        code_val = _find_text(txn, "transactionCode")
        shares_val = _find_text(txn, "transactionShares")
        price_val = _find_text(txn, "transactionPricePerShare")
        if not code_val:
            continue
        # Only open-market purchases ("P") matter for the cluster signal.
        if code_val.upper() != "P":
            continue
        try:
            shares = int(float(shares_val)) if shares_val else 0
            price = float(price_val) if price_val else 0.0
        except Exception:
            continue
        if shares <= 0 or price <= 0:
            continue
        transactions.append({
            "date": (date_val or "")[:10],
            "code": code_val.upper(),
            "shares": shares,
            "price": price,
            "value": float(shares) * float(price),
        })

    return {
        "ticker": ticker.upper(),
        "company": company,
        "insider_name": insider_name,
        "insider_title": insider_title,
        "transactions": transactions,
    }


# ---------------------------------------------------------------------------
# openinsider.com path (primary, fast aggregator)
# ---------------------------------------------------------------------------

def _fetch_openinsider_html(lookback_days: int = 30) -> Optional[str]:
    """Fetch the openinsider screener page for the last `lookback_days` days."""
    # ``fd`` = days back, ``xp=1`` = include only purchases (transaction code P).
    url = (
        f"{OPENINSIDER_BASE}/screener?"
        f"s=&o=&pl=&ph=&ll=&lh=&fd={lookback_days}&fdr=&td=0&tdr=&"
        f"fdlyl=&fdlyh=&daysago=&xp=1&xs=&"
        f"vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&"
        f"grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&"
        f"oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
    )
    resp = _rate_limited_get(url, headers=OPENINSIDER_HEADERS, timeout=HTTP_TIMEOUT)
    if resp is None:
        return None
    return resp.text


# Regex pattern to pull one openinsider table row. We match a <tr>...</tr>
# block and extract individual <td> contents in order. openinsider's columns
# (as of 2025) are:
#   X | Filing Date | Trade Date | Ticker | Insider Name | Title |
#   Trade Type | Price | Qty | Owned | DOwn | Value
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    """Strip HTML tags and decode common entities."""
    s = _TAG_RE.sub("", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return s.strip()


def _parse_openinsider_html(html: str) -> list[dict]:
    """
    Parse openinsider HTML and return a list of purchase rows:

        [{ "ticker", "insider", "title", "date", "shares", "price", "value" }, ...]

    Only rows with a "P" trade type (open-market purchase) are returned.
    Robust to extra rows / header rows / malformed cells.
    """
    out: list[dict] = []
    for row_match in _ROW_RE.finditer(html):
        body = row_match.group(1)
        cells = [_strip_html(c) for c in _CELL_RE.findall(body)]
        if len(cells) < 12:
            continue
        # Heuristic: expected column order
        #   [0]X [1]Filing Date [2]Trade Date [3]Ticker [4]Company
        #   [5]Insider [6]Title [7]Trade Type [8]Price [9]Qty [10]Owned
        #   [11]DOwn [12]Value
        # We anchor on the trade-type cell ("P - Purchase" / "P") so the
        # parser is resilient to optional/missing columns.
        trade_type_idx = None
        for i, c in enumerate(cells[:11]):
            if c.startswith("P -") or c == "P" or c.startswith("P-Purchase"):
                trade_type_idx = i
                break
        if trade_type_idx is None or trade_type_idx < 4:
            continue

        try:
            trade_date = cells[trade_type_idx - 5]
            ticker = cells[trade_type_idx - 4].upper().strip()
            # cells[trade_type_idx - 3] is company name (unused)
            insider = cells[trade_type_idx - 2]
            title = cells[trade_type_idx - 1]
            price_str = cells[trade_type_idx + 1] if trade_type_idx + 1 < len(cells) else ""
            qty_str = cells[trade_type_idx + 2] if trade_type_idx + 2 < len(cells) else ""
            value_str = cells[trade_type_idx + 5] if trade_type_idx + 5 < len(cells) else ""
        except IndexError:
            continue

        if not ticker or not re.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", ticker):
            continue

        price = _parse_money(price_str)
        shares = _parse_int(qty_str)
        value = _parse_money(value_str)
        # Some rows have negative qty for "owned change"; we need shares > 0.
        if shares is None or shares <= 0:
            continue
        if value is None or value <= 0:
            if price and shares:
                value = float(price) * float(shares)
            else:
                continue
        if price is None or price <= 0:
            price = value / shares if shares else 0.0

        # Normalize date — accept YYYY-MM-DD; reject anything we can't parse.
        m = re.search(r"(\d{4}-\d{2}-\d{2})", trade_date)
        if not m:
            continue
        out.append({
            "ticker": ticker,
            "insider": insider,
            "title": title,
            "date": m.group(1),
            "shares": int(shares),
            "price": float(price),
            "value": float(value),
            "code": "P",
        })
    return out


def _parse_money(s: str) -> Optional[float]:
    """Parse strings like ``$1,234.56`` / ``$2,345`` / ``+$50,000`` to float."""
    if not s:
        return None
    cleaned = s.replace(",", "").replace("$", "").replace("+", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int(s: str) -> Optional[int]:
    """Parse strings like ``1,234`` / ``+500`` to int."""
    if not s:
        return None
    cleaned = s.replace(",", "").replace("+", "").strip()
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public aggregator
# ---------------------------------------------------------------------------

def get_insider_purchases(
    lookback_days: int = 30,
    use_cache: bool = True,
) -> dict[str, list[dict]]:
    """
    Aggregate insider open-market purchases per ticker over the lookback window.

    Returns
    -------
    dict
        ``{ticker: [{"insider", "title", "date", "shares", "price", "value"}]}``

    Returns ``{}`` on any failure with a stderr warning. The result is cached
    at module level (keyed by ``lookback_days``) so repeated calls in the same
    process don't re-hit the network.
    """
    cache_key = ("openinsider", int(lookback_days))
    if use_cache and cache_key in _PURCHASES_CACHE:
        return _PURCHASES_CACHE[cache_key]

    html = _fetch_openinsider_html(lookback_days=lookback_days)
    if html is None:
        if use_cache:
            _PURCHASES_CACHE[cache_key] = {}
        return {}

    try:
        rows = _parse_openinsider_html(html)
    except Exception as exc:  # noqa: BLE001
        print(f"[edgar] WARNING: openinsider parse — {exc}", file=sys.stderr)
        rows = []

    grouped: dict[str, list[dict]] = {}
    cutoff = date.today() - timedelta(days=lookback_days)
    for r in rows:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if d < cutoff:
            continue
        grouped.setdefault(r["ticker"], []).append({
            "insider": r["insider"],
            "title": r["title"],
            "date": r["date"],
            "shares": int(r["shares"]),
            "price": float(r["price"]),
            "value": float(r["value"]),
        })

    if use_cache:
        _PURCHASES_CACHE[cache_key] = grouped
    return grouped


# ---------------------------------------------------------------------------
# Cluster-detection logic (pure function — fully unit testable)
# ---------------------------------------------------------------------------

def cluster_buy_signal(
    insider_purchases: dict[str, list[dict]],
    min_insiders: int = 3,
    min_value: float = 500_000.0,
    window_days: int = 30,
    insider_sales: Optional[dict[str, list[dict]]] = None,
) -> list[dict]:
    """
    Identify tickers passing the Lakonishok-Lee cluster-buy criteria.

    A ticker qualifies when, within the most recent ``window_days``:
      * at least ``min_insiders`` distinct insider names purchased shares
      * total purchase value >= ``min_value``
      * no insider selling (only checked if ``insider_sales`` is provided)

    Parameters
    ----------
    insider_purchases : dict
        ``{ticker: [purchase, ...]}`` from :func:`get_insider_purchases`.
    insider_sales : dict, optional
        Same shape, listing sell transactions. If omitted, the ``no_selling``
        criterion is assumed to be True (cannot verify without the data).

    Returns
    -------
    list of dict
        Sorted by ``total_value`` descending:
        ``[{ticker, n_insiders, total_value, earliest_date, latest_date,
            insider_names, no_selling}, ...]``
    """
    if not insider_purchases:
        return []

    today = date.today()
    cutoff = today - timedelta(days=window_days)
    sales_lookup = insider_sales or {}

    results: list[dict] = []
    for ticker, purchases in insider_purchases.items():
        if not purchases:
            continue
        windowed = [p for p in purchases if _in_window(p.get("date"), cutoff, today)]
        if not windowed:
            continue
        names = []
        seen_names = set()
        for p in windowed:
            n = (p.get("insider") or "").strip()
            if n and n.lower() not in seen_names:
                seen_names.add(n.lower())
                names.append(n)
        if len(seen_names) < min_insiders:
            continue
        total_value = sum(float(p.get("value") or 0) for p in windowed)
        if total_value < min_value:
            continue

        # No-selling check
        no_selling = True
        ticker_sales = sales_lookup.get(ticker, [])
        for s in ticker_sales:
            if _in_window(s.get("date"), cutoff, today):
                no_selling = False
                break

        dates = sorted([p.get("date") for p in windowed if p.get("date")])
        results.append({
            "ticker": ticker,
            "n_insiders": len(seen_names),
            "total_value": total_value,
            "earliest_date": dates[0] if dates else "",
            "latest_date": dates[-1] if dates else "",
            "insider_names": names[:5],
            "no_selling": no_selling,
        })

    results.sort(key=lambda r: r["total_value"], reverse=True)
    return results


def _in_window(date_str: Optional[str], cutoff: date, today: date) -> bool:
    """Return True if ``date_str`` (YYYY-MM-DD) is inside [cutoff, today]."""
    if not date_str:
        return False
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    except Exception:
        return False
    return cutoff <= d <= today


# ---------------------------------------------------------------------------
# Convenience: single-ticker cluster lookup (used by older Screen 7 stub)
# ---------------------------------------------------------------------------

def get_insider_cluster(
    ticker: str,
    days: int = 30,
    min_insiders: int = 3,
    min_value: float = 500_000.0,
) -> Optional[dict]:
    """
    Return cluster metadata for a single ticker, or ``None`` if it does not
    pass the cluster criteria. Convenience wrapper retained for backwards
    compatibility with the existing Screen 7 stub call signature.
    """
    if not ticker:
        return None
    purchases = get_insider_purchases(lookback_days=days)
    if not purchases:
        return None
    ticker = ticker.upper()
    sub = {ticker: purchases.get(ticker, [])}
    signals = cluster_buy_signal(
        sub,
        min_insiders=min_insiders,
        min_value=min_value,
        window_days=days,
    )
    if not signals:
        # Return minimal info even if it fails the criteria so callers can
        # apply their own thresholds.
        windowed = purchases.get(ticker, [])
        if not windowed:
            return None
        names = {(p.get("insider") or "").strip().lower() for p in windowed}
        names.discard("")
        return {
            "ticker": ticker,
            "n_buyers": len(names),
            "total_value": sum(float(p.get("value") or 0) for p in windowed),
            "has_sells": False,  # we don't track sells in this path
        }
    sig = signals[0]
    return {
        "ticker": ticker,
        "n_buyers": sig["n_insiders"],
        "total_value": sig["total_value"],
        "has_sells": not sig["no_selling"],
        "insider_names": sig["insider_names"],
        "earliest_date": sig["earliest_date"],
        "latest_date": sig["latest_date"],
    }
