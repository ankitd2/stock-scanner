"""
tests/test_fetcher.py — unit tests for data/fetcher.py.

All tests mock yfinance so no real network calls are made.
Run with:  python -m pytest tests/test_fetcher.py -v
"""

import sys
import time
import types
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

# Ensure the repo root is on sys.path so `data` is importable
import importlib, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from data.fetcher import (
    _retry,
    bulk_history,
    get_index_data,
    get_news,
    get_ticker_info,
)
from data.cache import RunCache


# ─────────────────────────────────────────────────────────────────────────────
# _retry
# ─────────────────────────────────────────────────────────────────────────────

class TestRetry:
    def test_succeeds_on_first_attempt(self):
        fn = MagicMock(return_value=42)
        result = _retry(fn, attempts=3, backoff=0)
        assert result == 42
        assert fn.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        fn = MagicMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"), 99])
        with patch("time.sleep"):
            result = _retry(fn, attempts=3, backoff=1.5)
        assert result == 99
        assert fn.call_count == 3

    def test_returns_none_after_all_failures(self, capsys):
        fn = MagicMock(side_effect=RuntimeError("always fails"))
        with patch("time.sleep"):
            result = _retry(fn, attempts=3, backoff=1.5)
        assert result is None
        captured = capsys.readouterr()
        assert "[warn]" in captured.err

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value="ok")
        _retry(fn, "pos1", attempts=2, backoff=0, kwarg1="v1")
        fn.assert_called_with("pos1", kwarg1="v1")

    def test_backoff_sleep_called(self):
        fn = MagicMock(side_effect=[ValueError(), "ok"])
        with patch("time.sleep") as mock_sleep:
            _retry(fn, attempts=3, backoff=2.0)
        # First retry: sleep(2.0^1 = 2.0)
        mock_sleep.assert_called_once_with(2.0)

    def test_no_sleep_on_last_attempt(self):
        fn = MagicMock(side_effect=[ValueError(), ValueError(), ValueError()])
        with patch("time.sleep") as mock_sleep:
            _retry(fn, attempts=3, backoff=1.5)
        # sleeps only between attempts 1→2 and 2→3, not after 3
        assert mock_sleep.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# bulk_history
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv_df(n=10) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [105.0] * n,
            "Low": [98.0] * n,
            "Close": [102.0] * n,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


def _make_multi_df(tickers: list[str], n=10) -> pd.DataFrame:
    """Build a multi-level (field, ticker) DataFrame as yfinance returns."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    arrays = []
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for ticker in tickers:
            arrays.append((field, ticker))
    cols = pd.MultiIndex.from_tuples(arrays, names=["field", "ticker"])
    data = {}
    for field, ticker in arrays:
        if field == "Volume":
            data[(field, ticker)] = [1_000_000] * n
        else:
            data[(field, ticker)] = [100.0 + len(ticker)] * n
    return pd.DataFrame(data, index=idx)


class TestBulkHistory:
    def test_empty_tickers_returns_empty(self):
        result = bulk_history([])
        assert result == {}

    def test_single_ticker_flat_columns(self):
        flat_df = _make_ohlcv_df(5)
        with patch("yfinance.download", return_value=flat_df):
            result = bulk_history(["AAPL"])
        assert "AAPL" in result
        assert list(result["AAPL"].columns) == [
            "Open", "High", "Low", "Close", "Volume"
        ]

    def test_multi_ticker_multilevel_columns(self):
        tickers = ["AAPL", "MSFT"]
        multi_df = _make_multi_df(tickers, n=5)
        with patch("yfinance.download", return_value=multi_df):
            result = bulk_history(tickers)
        assert "AAPL" in result
        assert "MSFT" in result
        assert list(result["AAPL"].columns) == [
            "Open", "High", "Low", "Close", "Volume"
        ]

    def test_empty_download_returns_empty(self, capsys):
        empty = pd.DataFrame()
        with patch("yfinance.download", return_value=empty):
            result = bulk_history(["AAPL"])
        assert result == {}
        assert "[warn]" in capsys.readouterr().err

    def test_download_exception_returns_empty(self, capsys):
        with patch("yfinance.download", side_effect=ConnectionError("no network")):
            with patch("time.sleep"):
                result = bulk_history(["AAPL"])
        assert result == {}
        assert "[warn]" in capsys.readouterr().err

    def test_ticker_with_all_nan_rows_excluded(self):
        """A ticker whose slice is all-NaN should NOT appear in result."""
        tickers = ["AAPL", "BADTICKER"]
        idx = pd.date_range("2025-01-01", periods=3, freq="B")
        import numpy as np

        cols = pd.MultiIndex.from_tuples(
            [(f, t) for f in ["Open","High","Low","Close","Volume"] for t in tickers],
            names=["field", "ticker"],
        )
        data = {}
        for field in ["Open","High","Low","Close","Volume"]:
            data[(field, "AAPL")] = [100.0, 101.0, 102.0]
            data[(field, "BADTICKER")] = [float("nan")] * 3
        multi_df = pd.DataFrame(data, index=idx)

        with patch("yfinance.download", return_value=multi_df):
            result = bulk_history(tickers)
        assert "AAPL" in result
        assert "BADTICKER" not in result


# ─────────────────────────────────────────────────────────────────────────────
# get_ticker_info
# ─────────────────────────────────────────────────────────────────────────────

def _make_ticker_mock(info: dict, recs: pd.DataFrame = None, calendar=None):
    mock = MagicMock()
    mock.info = info
    mock.recommendations = recs if recs is not None else pd.DataFrame()
    mock.calendar = calendar
    return mock


class TestGetTickerInfo:
    def _base_info(self):
        return {
            "currentPrice": 150.0,
            "regularMarketPreviousClose": 145.0,
            "fiftyTwoWeekHigh": 200.0,
            "fiftyTwoWeekLow": 100.0,
            "52WeekChange": 0.25,
            "targetMeanPrice": 180.0,
            "targetHighPrice": 220.0,
            "targetLowPrice": 140.0,
            "numberOfAnalystOpinions": 30,
            "revenueGrowth": 0.35,
            "grossMargins": 0.75,
            "operatingMargins": 0.28,
            "trailingPE": 25.0,
            "forwardPE": 20.0,
            "pegRatio": 1.5,
            "marketCap": 5_000_000_000,
            "shortName": "Test Corp",
            "sector": "Technology",
            "industry": "Software",
        }

    def test_basic_fields_populated(self):
        info = self._base_info()
        mock_t = _make_ticker_mock(info)
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_ticker_info("TEST")
        assert result["price"] == 150.0
        assert result["hi52"] == 200.0
        assert result["lo52"] == 100.0
        assert abs(result["pct_today"] - 3.448) < 0.01
        assert result["pct_ytd"] == 25.0
        assert result["rev_growth"] == 35.0
        assert result["gross_margin"] == 75.0
        assert abs(result["op_margin"] - 28.0) < 1e-9
        assert result["mcap"] == 5_000_000_000
        assert result["name"] == "Test Corp"

    def test_returns_empty_dict_on_exception(self, capsys):
        with patch("yfinance.Ticker", side_effect=RuntimeError("network error")):
            result = get_ticker_info("BAD")
        assert result == {}
        assert "[warn]" in capsys.readouterr().err

    def test_returns_empty_on_empty_info(self, capsys):
        mock_t = _make_ticker_mock({})
        # Make _retry return None for info
        with patch("yfinance.Ticker", return_value=mock_t):
            with patch("data.fetcher._retry", return_value=None):
                result = get_ticker_info("EMPTY")
        assert result == {}

    def test_recommendations_buy_pct_modern_schema(self):
        """Modern schema: columns strongBuy/buy/hold/sell/strongSell."""
        recs = pd.DataFrame(
            {
                "strongBuy": [5, 3],
                "buy": [10, 8],
                "hold": [3, 4],
                "sell": [1, 0],
                "strongSell": [0, 0],
            }
        )
        mock_t = _make_ticker_mock(self._base_info(), recs=recs)
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_ticker_info("TEST")
        # buy_count = 5+3+10+8 = 26, hold = 3+4 = 7, sell = 1+0 = 1
        assert result["buy_count"] == 26
        assert result["sell_count"] == 1
        total = 26 + 7 + 1
        assert abs(result["buy_pct"] - 26 / total * 100) < 0.01

    def test_recommendations_legacy_schema(self):
        """Legacy schema: 'To Grade' column with string grades."""
        recs = pd.DataFrame(
            {
                "To Grade": [
                    "Buy", "Overweight", "Strong Buy",
                    "Hold", "Neutral",
                    "Sell", "Underweight",
                ]
            }
        )
        mock_t = _make_ticker_mock(self._base_info(), recs=recs)
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_ticker_info("TEST")
        assert result["buy_count"] == 3
        assert result["hold_count"] == 2
        assert result["sell_count"] == 2

    def test_earnings_dt_from_dict_calendar(self):
        future_date = date.today() + timedelta(days=10)
        cal = {"Earnings Date": [future_date]}
        mock_t = _make_ticker_mock(self._base_info(), calendar=cal)
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_ticker_info("TEST")
        assert result["earnings_dt"] == future_date

    def test_earnings_dt_past_date_excluded(self):
        past_date = date.today() - timedelta(days=5)
        cal = {"Earnings Date": [past_date]}
        mock_t = _make_ticker_mock(self._base_info(), calendar=cal)
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_ticker_info("TEST")
        assert result["earnings_dt"] is None

    def test_none_margins_handled(self):
        info = self._base_info()
        info["revenueGrowth"] = None
        info["grossMargins"] = None
        info["operatingMargins"] = None
        mock_t = _make_ticker_mock(info)
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_ticker_info("TEST")
        assert result["rev_growth"] is None
        assert result["gross_margin"] is None
        assert result["op_margin"] is None


# ─────────────────────────────────────────────────────────────────────────────
# get_news
# ─────────────────────────────────────────────────────────────────────────────

class TestGetNews:
    def test_old_schema_link_field(self):
        mock_t = MagicMock()
        mock_t.news = [
            {"title": "Old news 1", "link": "https://example.com/1"},
            {"title": "Old news 2", "link": "https://example.com/2"},
        ]
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_news("AAPL", n=2)
        assert len(result) == 2
        assert result[0] == {"title": "Old news 1", "url": "https://example.com/1"}

    def test_new_schema_canonicalurl(self):
        mock_t = MagicMock()
        mock_t.news = [
            {
                "title": "New news 1",
                "content": {
                    "canonicalUrl": {"url": "https://example.com/new/1"}
                },
            }
        ]
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_news("AAPL", n=3)
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/new/1"

    def test_respects_n_limit(self):
        mock_t = MagicMock()
        mock_t.news = [
            {"title": f"News {i}", "link": f"https://example.com/{i}"}
            for i in range(10)
        ]
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_news("AAPL", n=3)
        assert len(result) == 3

    def test_returns_empty_on_exception(self, capsys):
        with patch("yfinance.Ticker", side_effect=RuntimeError("fail")):
            result = get_news("AAPL")
        assert result == []
        assert "[warn]" in capsys.readouterr().err

    def test_empty_news_list(self):
        mock_t = MagicMock()
        mock_t.news = []
        with patch("yfinance.Ticker", return_value=mock_t):
            result = get_news("AAPL")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# get_index_data
# ─────────────────────────────────────────────────────────────────────────────

class TestGetIndexData:
    def _make_hist(self, symbols: list[str], n: int = 5) -> dict[str, pd.DataFrame]:
        result = {}
        for i, sym in enumerate(symbols):
            idx = pd.date_range("2025-01-01", periods=n, freq="B")
            result[sym] = pd.DataFrame(
                {
                    "Open": [100.0 + i] * n,
                    "High": [105.0 + i] * n,
                    "Low": [98.0 + i] * n,
                    "Close": [100.0 + i + j * 0.5 for j in range(n)],
                    "Volume": [1_000_000] * n,
                },
                index=idx,
            )
        return result

    def test_returns_price_and_pct_change(self):
        symbols = ["^GSPC", "^IXIC"]
        hist = self._make_hist(symbols)
        with patch("data.fetcher.bulk_history", return_value=hist):
            result = get_index_data(symbols)
        assert "^GSPC" in result
        assert "^IXIC" in result
        gspc = result["^GSPC"]
        assert "price" in gspc
        assert "pct_change" in gspc
        assert "prev_close" in gspc
        # last Close for ^GSPC: 100 + 0 + 4*0.5 = 102, prev: 100 + 3*0.5 = 101.5
        assert abs(gspc["price"] - 102.0) < 0.01
        assert abs(gspc["prev_close"] - 101.5) < 0.01

    def test_missing_symbol_not_in_result(self, capsys):
        symbols = ["^GSPC", "MISSING"]
        hist = self._make_hist(["^GSPC"])
        with patch("data.fetcher.bulk_history", return_value=hist):
            result = get_index_data(symbols)
        assert "^GSPC" in result
        assert "MISSING" not in result

    def test_returns_empty_on_exception(self, capsys):
        with patch("data.fetcher.bulk_history", side_effect=RuntimeError("err")):
            result = get_index_data(["^GSPC"])
        assert result == {}
        assert "[warn]" in capsys.readouterr().err

    def test_empty_symbols_returns_empty(self):
        result = get_index_data([])
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# RunCache
# ─────────────────────────────────────────────────────────────────────────────

class TestRunCache:
    def test_miss_calls_fn(self):
        cache = RunCache()
        fn = MagicMock(return_value=42)
        result = cache.get_or_fetch("key1", fn, "arg1")
        assert result == 42
        fn.assert_called_once_with("arg1")

    def test_hit_does_not_call_fn_again(self):
        cache = RunCache()
        fn = MagicMock(return_value=99)
        cache.get_or_fetch("key1", fn)
        cache.get_or_fetch("key1", fn)
        assert fn.call_count == 1

    def test_stats_tracking(self):
        cache = RunCache()
        fn = MagicMock(return_value=1)
        cache.get_or_fetch("a", fn)
        cache.get_or_fetch("a", fn)
        cache.get_or_fetch("b", fn)
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2

    def test_clear_resets_store_and_stats(self):
        cache = RunCache()
        fn = MagicMock(return_value=7)
        cache.get_or_fetch("x", fn)
        cache.clear()
        assert cache.stats() == {"hits": 0, "misses": 0}
        # After clear, same key should be a miss again
        cache.get_or_fetch("x", fn)
        assert cache.stats()["misses"] == 1

    def test_different_keys_independent(self):
        cache = RunCache()
        fn_a = MagicMock(return_value="a_val")
        fn_b = MagicMock(return_value="b_val")
        assert cache.get_or_fetch("key_a", fn_a) == "a_val"
        assert cache.get_or_fetch("key_b", fn_b) == "b_val"
        assert cache.get_or_fetch("key_a", fn_a) == "a_val"
        assert fn_a.call_count == 1
        assert fn_b.call_count == 1

    def test_caches_none_value(self):
        """None is a valid cached value and should not cause a re-fetch."""
        cache = RunCache()
        fn = MagicMock(return_value=None)
        cache.get_or_fetch("none_key", fn)
        cache.get_or_fetch("none_key", fn)
        assert fn.call_count == 1

    def test_kwargs_passed_to_fn(self):
        cache = RunCache()
        fn = MagicMock(return_value="kw")
        cache.get_or_fetch("kw_key", fn, kw1="v1", kw2=2)
        fn.assert_called_once_with(kw1="v1", kw2=2)
