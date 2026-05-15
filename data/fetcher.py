"""
data/fetcher.py — central yfinance wrapper.

All analytics modules call this; nothing calls yfinance directly except this file.
"""

import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf


# ── Retry helper ─────────────────────────────────────────────────────────────

def _retry(fn, *args, attempts: int = 3, backoff: float = 1.5, **kwargs):
    """Call fn(*args, **kwargs), retry up to `attempts` times on exception.

    Waits backoff^attempt seconds between tries (attempt starts at 1).
    Returns None on all failures.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                wait = backoff ** attempt
                time.sleep(wait)
    print(f"[warn] _retry: all {attempts} attempts failed — {last_exc}", file=sys.stderr)
    return None


# ── Bulk history ─────────────────────────────────────────────────────────────

def bulk_history(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV history for all tickers in one yf.download() call.

    Returns {ticker: DataFrame(Open, High, Low, Close, Volume)} — only tickers
    with data. Handles yfinance's multi-level column output (group_by='ticker').
    Uses retry logic. Logs failures to stderr, does NOT raise.
    """
    if not tickers:
        return {}

    def _download():
        return yf.download(
            tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )

    raw = _retry(_download, attempts=3, backoff=1.5)
    if raw is None or raw.empty:
        print(
            f"[warn] bulk_history: download returned empty for {tickers}",
            file=sys.stderr,
        )
        return {}

    result: dict[str, pd.DataFrame] = {}

    # yfinance returns a multi-level column DataFrame when multiple tickers are
    # requested: top level = field (Open/High/Low/Close/Volume),
    # second level = ticker. When a single ticker is passed it collapses to a
    # flat DataFrame instead.
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker path — group_by='ticker' puts ticker at level 0
        # but the actual layout depends on yfinance version.
        # Try ticker-first (group_by='ticker') layout.
        top_level = raw.columns.get_level_values(0).unique().tolist()
        ohlcv = {"Open", "High", "Low", "Close", "Volume"}

        # Determine whether top-level = tickers or fields
        if ohlcv.issuperset(set(top_level)):
            # Top level is fields (field, ticker) layout — transpose logic
            for ticker in tickers:
                try:
                    df = raw.xs(ticker, axis=1, level=1)[
                        ["Open", "High", "Low", "Close", "Volume"]
                    ].dropna(how="all")
                    if not df.empty:
                        result[ticker] = df
                except Exception as e:
                    print(
                        f"[warn] bulk_history: slice failed for {ticker}: {e}",
                        file=sys.stderr,
                    )
        else:
            # Top level is tickers (ticker, field) layout
            for ticker in tickers:
                try:
                    df = raw[ticker][
                        ["Open", "High", "Low", "Close", "Volume"]
                    ].dropna(how="all")
                    if not df.empty:
                        result[ticker] = df
                except Exception as e:
                    print(
                        f"[warn] bulk_history: slice failed for {ticker}: {e}",
                        file=sys.stderr,
                    )
    else:
        # Single-ticker path — flat columns
        if len(tickers) == 1:
            try:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(
                    how="all"
                )
                if not df.empty:
                    result[tickers[0]] = df
            except Exception as e:
                print(
                    f"[warn] bulk_history: single-ticker slice failed for "
                    f"{tickers[0]}: {e}",
                    file=sys.stderr,
                )

    return result


# ── Recommendations helper ────────────────────────────────────────────────────

_BUY_GRADES = {
    "buy", "strong buy", "overweight", "outperform",
    "market outperform", "sector outperform", "positive",
    "accumulate", "add",
}
_SELL_GRADES = {
    "sell", "strong sell", "underweight", "underperform",
    "market underperform", "sector underperform", "negative", "reduce",
}
_HOLD_GRADES = {
    "hold", "neutral", "market perform", "sector perform",
    "peer perform", "equal-weight", "equal weight", "in-line",
    "inline", "fair value",
}


def _parse_recommendations(t: "yf.Ticker") -> tuple[int, int, int, float]:
    """Return (buy_count, hold_count, sell_count, buy_pct) from last 30 rows."""
    try:
        recs = t.recommendations
        if recs is None or recs.empty:
            return 0, 0, 0, 0.0
        recs = recs.tail(30)

        buy_col = None
        # Modern yfinance: columns like strongBuy, buy, hold, sell, strongSell
        modern_cols = {"strongBuy", "buy", "hold", "sell", "strongSell"}
        if modern_cols.issubset(set(recs.columns)):
            buy_count = int(recs["strongBuy"].sum() + recs["buy"].sum())
            sell_count = int(recs["strongSell"].sum() + recs["sell"].sum())
            hold_count = int(recs["hold"].sum())
        elif "To Grade" in recs.columns:
            # Legacy schema: one row per analyst action with a "To Grade" column
            grades = recs["To Grade"].str.lower().fillna("")
            buy_count = int(grades.apply(lambda g: any(b in g for b in _BUY_GRADES)).sum())
            sell_count = int(grades.apply(lambda g: any(s in g for s in _SELL_GRADES)).sum())
            hold_count = int(grades.apply(lambda g: any(h in g for h in _HOLD_GRADES)).sum())
        else:
            return 0, 0, 0, 0.0

        total = buy_count + hold_count + sell_count
        buy_pct = (buy_count / total * 100) if total > 0 else 0.0
        return buy_count, hold_count, sell_count, buy_pct
    except Exception:
        return 0, 0, 0, 0.0


def _parse_earnings_dt(t: "yf.Ticker") -> "date | None":
    """Return next future earnings date from t.calendar, or None."""
    try:
        cal = t.calendar
        today = date.today()
        if cal is None:
            return None
        # Modern yfinance: dict with key 'Earnings Date'
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed is None:
                return None
            if isinstance(ed, (list, tuple)):
                dates = [d for d in ed if d is not None]
            else:
                dates = [ed]
        elif isinstance(cal, pd.DataFrame):
            # Legacy: DataFrame with 'Earnings Date' in index or columns
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"]
                dates = list(val) if hasattr(val, "__iter__") else [val]
            elif "Earnings Date" in cal.columns:
                dates = list(cal["Earnings Date"].dropna())
            else:
                return None
        else:
            return None

        future = []
        for d in dates:
            try:
                if isinstance(d, (datetime, pd.Timestamp)):
                    d = d.date()
                elif not isinstance(d, date):
                    d = pd.Timestamp(d).date()
                if d >= today:
                    future.append(d)
            except Exception:
                continue
        return min(future) if future else None
    except Exception:
        return None


# ── Single-ticker fundamentals ─────────────────────────────────────────────

def get_ticker_info(ticker: str) -> dict:
    """Fetch fundamental data for a single ticker via yf.Ticker(ticker).info.

    Returns dict with keys (use .get() with None fallback for all):
      price, hi52, lo52, pct_today, pct_ytd,
      tgt_mean, tgt_high, tgt_low, n_analysts,
      buy_count, hold_count, sell_count, buy_pct,
      rev_growth, gross_margin, op_margin,
      pe_trailing, pe_forward, peg, mcap,
      name, sector, industry, earnings_dt

    Returns {} on any exception.
    """
    try:
        t = yf.Ticker(ticker)
        info = _retry(lambda: t.info, attempts=3, backoff=1.5)
        if not info:
            print(f"[warn] {ticker}: info returned empty", file=sys.stderr)
            return {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")

        pct_today: float | None = None
        if price and prev_close:
            pct_today = (price - prev_close) / prev_close * 100

        # YTD: use 52wk high/low as proxy when open is unavailable; prefer
        # regularMarketOpen for same-day reference, else fall back.
        year_open = info.get("52WeekChange")  # fractional
        pct_ytd: float | None = None
        if year_open is not None:
            pct_ytd = year_open * 100

        buy_count, hold_count, sell_count, buy_pct = _parse_recommendations(t)
        earnings_dt = _parse_earnings_dt(t)

        rev_growth_raw = info.get("revenueGrowth")
        rev_growth = rev_growth_raw * 100 if rev_growth_raw is not None else None

        gross_margin_raw = info.get("grossMargins")
        gross_margin = gross_margin_raw * 100 if gross_margin_raw is not None else None

        op_margin_raw = info.get("operatingMargins")
        op_margin = op_margin_raw * 100 if op_margin_raw is not None else None

        return {
            "price": price,
            "hi52": info.get("fiftyTwoWeekHigh"),
            "lo52": info.get("fiftyTwoWeekLow"),
            "pct_today": pct_today,
            "pct_ytd": pct_ytd,
            "tgt_mean": info.get("targetMeanPrice"),
            "tgt_high": info.get("targetHighPrice"),
            "tgt_low": info.get("targetLowPrice"),
            "n_analysts": info.get("numberOfAnalystOpinions"),
            "buy_count": buy_count,
            "hold_count": hold_count,
            "sell_count": sell_count,
            "buy_pct": buy_pct,
            "rev_growth": rev_growth,
            "gross_margin": gross_margin,
            "op_margin": op_margin,
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "peg": info.get("pegRatio"),
            "mcap": info.get("marketCap"),
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "earnings_dt": earnings_dt,
        }
    except Exception as e:
        print(f"[warn] {ticker}: get_ticker_info failed — {e}", file=sys.stderr)
        return {}


# ── News ──────────────────────────────────────────────────────────────────────

def get_news(ticker: str, n: int = 3) -> list[dict]:
    """Return up to n news items: [{"title": str, "url": str}].

    Handles both old yfinance schema (item["link"]) and new schema
    (item["content"]["canonicalUrl"]["url"]).
    Returns [] on any exception.
    """
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news
        if not raw_news:
            return []

        items = []
        for item in raw_news[:n]:
            try:
                title = item.get("title") or ""

                # New schema: nested content dict
                url = None
                content = item.get("content")
                if isinstance(content, dict):
                    canonical = content.get("canonicalUrl")
                    if isinstance(canonical, dict):
                        url = canonical.get("url")
                    if not url:
                        url = content.get("url")

                # Old schema fallback
                if not url:
                    url = item.get("link") or item.get("url") or ""

                if title or url:
                    items.append({"title": title, "url": url or ""})
            except Exception:
                continue

        return items
    except Exception as e:
        print(f"[warn] {ticker}: get_news failed — {e}", file=sys.stderr)
        return []


# ── Index / macro data ────────────────────────────────────────────────────────

_INDEX_SYMBOLS = [
    "^GSPC", "^IXIC", "^VIX", "^VIX3M", "^VIX9D", "^VVIX",
    "^TNX", "GC=F", "HG=F", "SPY", "RSP", "SPHB", "SPLV",
]


def get_index_data(symbols: list[str] | None = None) -> dict[str, dict]:
    """Fetch indices (default: S&P500, Nasdaq, VIX variants, yields, metals, ETFs).

    Returns {symbol: {"price": float, "pct_change": float, "prev_close": float}}
    Uses bulk_history internally (period='5d') to get recent prices.
    Returns {} on any exception.
    """
    if symbols is None:
        symbols = _INDEX_SYMBOLS

    if not symbols:
        return {}

    try:
        hist = bulk_history(symbols, period="5d", interval="1d")
        result: dict[str, dict] = {}

        for sym in symbols:
            df = hist.get(sym)
            if df is None or df.empty or len(df) < 2:
                if df is not None and len(df) == 1:
                    # Only today's data — no prev close available
                    price = float(df["Close"].iloc[-1])
                    result[sym] = {
                        "price": price,
                        "pct_change": None,
                        "prev_close": None,
                    }
                else:
                    print(
                        f"[warn] get_index_data: insufficient data for {sym}",
                        file=sys.stderr,
                    )
                continue

            price = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            pct_change = (price - prev_close) / prev_close * 100 if prev_close else None

            result[sym] = {
                "price": price,
                "pct_change": pct_change,
                "prev_close": prev_close,
            }

        return result
    except Exception as e:
        print(f"[warn] get_index_data failed — {e}", file=sys.stderr)
        return {}
