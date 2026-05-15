"""
data/fetcher.py — stub / real implementation placeholder.

get_index_data(symbols) → {symbol: {price, pct_change, prev_close}}
bulk_history(tickers, period, interval) → {ticker: DataFrame(OHLCV)}
"""

import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    import pandas as pd
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


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


def bulk_history(tickers: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, "pd.DataFrame"]:
    """Download OHLCV history for a list of tickers. Returns {ticker: DataFrame}."""
    if not _YF_AVAILABLE:
        return {}
    try:
        import pandas as pd
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
