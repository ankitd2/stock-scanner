"""
data/fetcher.py — yfinance-backed data fetcher.

get_ticker_info(ticker) -> dict with keys:
    price, hi52, lo52, pct_today, tgt_mean, tgt_high, n_analysts,
    buy_pct, sell_count, rev_growth, gross_margin, pe_forward, peg,
    mcap, earnings_dt, name

bulk_history(tickers, period, interval) -> {ticker: DataFrame(OHLCV)}
get_index_data(symbols) -> {symbol: {price, pct_change, prev_close}}
"""

import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    import pandas as pd
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


def _get_rec_counts(t) -> tuple:
    """Parse buy/hold/sell from yfinance recommendations. Returns (buy, hold, sell)."""
    buy = hold = sell = 0
    try:
        recs = t.recommendations
        if recs is None or recs.empty:
            return buy, hold, sell
        for col in recs.tail(30).columns:
            cl = col.lower()
            val = int(recs.tail(30)[col].sum())
            if "buy" in cl:
                buy += val
            elif "hold" in cl or "neutral" in cl:
                hold += val
            elif "sell" in cl:
                sell += val
    except Exception:
        pass
    return buy, hold, sell


def _get_earnings_date(t) -> Optional[date]:
    """Extract next earnings date from yfinance calendar."""
    try:
        cal = t.calendar
        if not isinstance(cal, dict):
            return None
        dates = cal.get("Earnings Date", [])
        for d in (dates if hasattr(dates, "__iter__") else [dates]):
            dt = pd.Timestamp(d).date()
            if dt >= date.today():
                return dt
    except Exception:
        pass
    return None


def get_ticker_info(ticker: str) -> Optional[dict]:
    """
    Fetch fundamental and price data for a single ticker.

    Returns a dict or None on failure. Keys:
        ticker, name, price, hi52, lo52, pct_today,
        tgt_mean, tgt_high, tgt_low, n_analysts,
        buy_pct, sell_count, buy, hold, sell,
        rev_growth, gross_margin, op_margin, pe_forward, pe_trailing, peg,
        mcap, earnings_dt, sector
    """
    if not _YF_AVAILABLE:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="1y", auto_adjust=True)
        if hist.empty:
            return None

        curr = float(hist["Close"].iloc[-1])
        hi52 = float(hist["High"].max())
        lo52 = float(hist["Low"].min())

        pct_today = 0.0
        if len(hist) >= 2:
            prev = float(hist["Close"].iloc[-2])
            pct_today = (curr - prev) / prev * 100 if prev else 0.0

        tgt_mean = info.get("targetMeanPrice")
        tgt_high = info.get("targetHighPrice")
        tgt_low = info.get("targetLowPrice")
        n_analysts = int(info.get("numberOfAnalystOpinions") or 0)

        buy, hold, sell = _get_rec_counts(t)
        total = buy + hold + sell
        buy_pct = round(buy / total * 100) if total else None

        rev_growth = info.get("revenueGrowth")
        gross_margin = info.get("grossMargins")
        op_margin = info.get("operatingMargins")

        def clean_pe(v):
            return round(v, 1) if v and 0 < v < 2000 else None

        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", ""),
            "price": round(curr, 2),
            "hi52": round(hi52, 2),
            "lo52": round(lo52, 2),
            "pct_today": round(pct_today, 2),
            "tgt_mean": round(tgt_mean, 2) if tgt_mean else None,
            "tgt_high": round(tgt_high, 2) if tgt_high else None,
            "tgt_low": round(tgt_low, 2) if tgt_low else None,
            "n_analysts": n_analysts,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "sell_count": sell,
            "buy_pct": buy_pct,
            "rev_growth": round(rev_growth * 100, 1) if rev_growth else None,
            "gross_margin": round(gross_margin * 100, 1) if gross_margin else None,
            "op_margin": round(op_margin * 100, 1) if op_margin else None,
            "pe_trailing": clean_pe(info.get("trailingPE")),
            "pe_forward": clean_pe(info.get("forwardPE")),
            "peg": (
                round(info.get("pegRatio"), 2)
                if info.get("pegRatio") and 0 < info.get("pegRatio", 99) < 20
                else None
            ),
            "mcap": info.get("marketCap"),
            "earnings_dt": _get_earnings_date(t),
        }
    except Exception as e:
        print(f"  [{ticker}] fetcher error: {e}")
        return None


def bulk_history(
    tickers: List[str], period: str = "1y", interval: str = "1d"
) -> Dict[str, "pd.DataFrame"]:
    """Download OHLCV history for a list of tickers. Returns {ticker: DataFrame}."""
    if not _YF_AVAILABLE:
        return {}
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        result = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for tkr in tickers:
                try:
                    df = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                    if not df.empty:
                        result[tkr] = df
                except Exception:
                    pass
        else:
            # single ticker returned flat
            if len(tickers) == 1 and not raw.empty:
                result[tickers[0]] = raw
        return result
    except Exception:
        return {}


def get_index_data(symbols: List[str]) -> Dict[str, dict]:
    """Fetch latest price + 1-day pct change for a list of index / ETF symbols."""
    result = {}
    if not _YF_AVAILABLE:
        return result
    try:
        tickers = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                t = tickers.tickers[sym]
                info = t.fast_info
                price = info.last_price
                prev = info.previous_close
                pct = ((price - prev) / prev * 100) if prev else None
                result[sym] = {"price": price, "pct_change": pct, "prev_close": prev}
            except Exception:
                result[sym] = {"price": None, "pct_change": None, "prev_close": None}
    except Exception:
        pass
    return result
