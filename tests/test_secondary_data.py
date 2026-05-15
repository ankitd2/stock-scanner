"""
tests/test_secondary_data.py

Unit tests for data.fred and data.aaii.
All HTTP/network calls are mocked so the tests run offline.
"""

import io
import sys
import math
import types
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------

def _make_fred_df(values: list[float], freq: str = "B") -> pd.DataFrame:
    """Build a minimal DataFrame that mimics pandas-datareader FRED output."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.DataFrame({"BAMLH0A0HYM2": values}, index=idx)


def _make_aaii_bytes(bullish: list, bearish: list, neutral: list, dates: list) -> bytes:
    """Create a minimal in-memory Excel file that looks like the AAII export."""
    df = pd.DataFrame({
        "Reported Date": dates,
        "Bullish": bullish,
        "Bearish": bearish,
        "Neutral": neutral,
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# FRED tests
# ---------------------------------------------------------------------------

class TestFredZscore(unittest.TestCase):
    """fred_zscore should compute z-score correctly and handle edge cases."""

    def setUp(self):
        from data.fred import fred_zscore
        self.fred_zscore = fred_zscore

    def test_basic_zscore(self):
        """Last value exactly 1 std above mean should give z ≈ 1."""
        values = list(range(1, 253))  # 252 values
        series = pd.Series(values, dtype=float)
        z = self.fred_zscore(series, window=252)
        self.assertIsNotNone(z)
        # mean of 1..252 = 126.5, std ≈ 72.9; last=252 → z ≈ 1.72
        self.assertAlmostEqual(z, (252 - 126.5) / pd.Series(values, dtype=float).std(), places=4)

    def test_returns_none_when_too_few_obs(self):
        """Should return None if series length < window/2."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = self.fred_zscore(series, window=252)
        self.assertIsNone(result)

    def test_returns_none_for_zero_std(self):
        """Should return None when all values are identical (std=0)."""
        series = pd.Series([5.0] * 300)
        result = self.fred_zscore(series, window=252)
        self.assertIsNone(result)

    def test_none_series(self):
        """Should return None for None input."""
        result = self.fred_zscore(None, window=252)
        self.assertIsNone(result)

    def test_negative_zscore(self):
        """Last value well below mean should produce a negative z-score."""
        values = list(range(252, 0, -1))  # decreasing so last=1
        series = pd.Series(values, dtype=float)
        z = self.fred_zscore(series, window=252)
        self.assertIsNotNone(z)
        self.assertLess(z, 0)


class TestGetFredSeries(unittest.TestCase):
    """get_fred_series should return clean pd.Series per requested series."""

    def _mock_datareader(self, fred_id, source, start, end):
        """Fake DataReader returning 10 rows for any request."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({fred_id: list(range(10, 20))}, index=idx)
        return df

    @patch("data.fred.web.DataReader")
    def test_returns_dict_of_series(self, mock_dr):
        mock_dr.side_effect = self._mock_datareader
        from data.fred import get_fred_series, FRED_SERIES

        result = get_fred_series(FRED_SERIES, lookback_days=400)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(FRED_SERIES.keys()))
        for name, series in result.items():
            self.assertIsInstance(series, pd.Series)
            self.assertGreater(len(series), 0)

    @patch("data.fred.web.DataReader", side_effect=Exception("network down"))
    def test_returns_empty_dict_on_failure(self, mock_dr):
        from data.fred import get_fred_series, FRED_SERIES

        result = get_fred_series(FRED_SERIES, lookback_days=400)
        self.assertEqual(result, {})

    @patch("data.fred.web.DataReader")
    def test_drops_nan_rows(self, mock_dr):
        """NaN rows must be removed from the returned series."""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"DGS10": [1.0, float("nan"), 3.0, float("nan"), 5.0]}, index=idx)
        mock_dr.return_value = df

        from data.fred import get_fred_series

        result = get_fred_series({"dgs10": "DGS10"}, lookback_days=400)
        self.assertIn("dgs10", result)
        self.assertFalse(result["dgs10"].isna().any())
        self.assertEqual(len(result["dgs10"]), 3)

    @patch("data.fred.web.DataReader")
    def test_series_sorted_ascending(self, mock_dr):
        """Series must be sorted ascending (most-recent last)."""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        # Return intentionally unsorted by reversing index
        df = pd.DataFrame({"T10Y2Y": [5.0, 4.0, 3.0, 2.0, 1.0]}, index=idx[::-1])
        mock_dr.return_value = df

        from data.fred import get_fred_series

        result = get_fred_series({"t10y2y": "T10Y2Y"}, lookback_days=400)
        series = result["t10y2y"]
        self.assertTrue(series.index.is_monotonic_increasing)


class TestLatestFred(unittest.TestCase):
    """latest_fred should return float values for each series."""

    @patch("data.fred.web.DataReader")
    def test_returns_floats(self, mock_dr):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")

        def side_effect(fred_id, *args, **kwargs):
            return pd.DataFrame({fred_id: [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

        mock_dr.side_effect = side_effect

        from data.fred import latest_fred, FRED_SERIES

        result = latest_fred(FRED_SERIES)
        self.assertEqual(set(result.keys()), set(FRED_SERIES.keys()))
        for v in result.values():
            self.assertIsInstance(v, float)

    @patch("data.fred.web.DataReader", side_effect=Exception("fail"))
    def test_returns_empty_on_failure(self, mock_dr):
        from data.fred import latest_fred, FRED_SERIES

        result = latest_fred(FRED_SERIES)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# AAII tests
# ---------------------------------------------------------------------------

class TestGetAaiiSentiment(unittest.TestCase):
    """get_aaii_sentiment should parse the XLS/CSV and return clean data."""

    def _mock_response(self, bullish, bearish, neutral, dates):
        """Build a mock requests.Response with AAII-like Excel content."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.content = _make_aaii_bytes(bullish, bearish, neutral, dates)
        resp.text = ""
        return resp

    @patch("data.aaii.requests.get")
    def test_basic_parse(self, mock_get):
        dates = pd.date_range("2024-01-01", periods=5, freq="W-FRI")
        mock_get.return_value = self._mock_response(
            bullish=[40, 42, 38, 45, 50],
            bearish=[30, 28, 35, 25, 20],
            neutral=[30, 30, 27, 30, 30],
            dates=dates.strftime("%Y-%m-%d").tolist(),
        )

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment(n_weeks=260)
        self.assertFalse(df.empty)
        self.assertIn("bull_bear_spread", df.columns)
        self.assertEqual(list(df.columns), ["date", "bullish", "bearish", "neutral", "bull_bear_spread"])
        # bull_bear_spread for last row: 50-20=30
        self.assertAlmostEqual(float(df.iloc[-1]["bull_bear_spread"]), 30.0)

    @patch("data.aaii.requests.get")
    def test_sorted_ascending(self, mock_get):
        """Rows must be sorted ascending by date."""
        dates = pd.date_range("2024-06-01", periods=4, freq="W-FRI")
        mock_get.return_value = self._mock_response(
            bullish=[35, 40, 38, 42],
            bearish=[30, 28, 32, 25],
            neutral=[35, 32, 30, 33],
            dates=dates[::-1].strftime("%Y-%m-%d").tolist(),  # reversed
        )

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertTrue(df["date"].is_monotonic_increasing)

    @patch("data.aaii.requests.get")
    def test_n_weeks_limit(self, mock_get):
        """Should return at most n_weeks rows."""
        dates = pd.date_range("2020-01-01", periods=300, freq="W-FRI")
        mock_get.return_value = self._mock_response(
            bullish=[40.0] * 300,
            bearish=[30.0] * 300,
            neutral=[30.0] * 300,
            dates=dates.strftime("%Y-%m-%d").tolist(),
        )

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment(n_weeks=52)
        self.assertLessEqual(len(df), 52)

    @patch("data.aaii.requests.get", side_effect=Exception("timeout"))
    def test_returns_empty_on_http_failure(self, mock_get):
        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertTrue(df.empty)
        expected_cols = ["date", "bullish", "bearish", "neutral", "bull_bear_spread"]
        self.assertEqual(list(df.columns), expected_cols)

    @patch("data.aaii.requests.get")
    def test_returns_empty_on_bad_response_body(self, mock_get):
        """Non-parseable response should return empty DF, not raise."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.content = b"not valid excel or csv at all !!!!"
        resp.text = "not valid excel or csv at all !!!!"
        mock_get.return_value = resp

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertTrue(df.empty)

    @patch("data.aaii.requests.get")
    def test_column_detection_case_insensitive(self, mock_get):
        """Column matching must work regardless of case."""
        df_raw = pd.DataFrame({
            "Reported Date": pd.date_range("2024-01-01", periods=3, freq="W-FRI").strftime("%Y-%m-%d").tolist(),
            "BULLISH": [0.40, 0.42, 0.45],  # uppercase + fraction (0-1)
            "BEARISH": [0.30, 0.28, 0.25],
            "NEUTRAL": [0.30, 0.30, 0.30],
        })
        buf = io.BytesIO()
        df_raw.to_excel(buf, index=False)
        buf.seek(0)

        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.content = buf.read()
        resp.text = ""
        mock_get.return_value = resp

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertFalse(df.empty)
        # Values should be normalised to 0-100
        self.assertGreater(float(df["bullish"].max()), 1.0)

    @patch("data.aaii.requests.get")
    def test_sends_browser_user_agent(self, mock_get):
        """AAII's CDN 403s the default requests UA — a browser UA must be sent."""
        dates = pd.date_range("2024-01-01", periods=3, freq="W-FRI")
        mock_get.return_value = self._mock_response(
            bullish=[40, 42, 45],
            bearish=[30, 28, 25],
            neutral=[30, 30, 30],
            dates=dates.strftime("%Y-%m-%d").tolist(),
        )

        from data.aaii import get_aaii_sentiment

        get_aaii_sentiment()

        self.assertEqual(mock_get.call_count, 1)
        _args, kwargs = mock_get.call_args
        self.assertIn("headers", kwargs)
        headers = kwargs["headers"] or {}
        # Normalise header keys (HTTP headers are case-insensitive).
        lowered = {k.lower(): v for k, v in headers.items()}
        self.assertIn("user-agent", lowered)
        ua = lowered["user-agent"]
        self.assertNotIn("python-requests", ua.lower())
        # Sanity-check that it looks like a real browser UA.
        self.assertTrue(
            any(token in ua for token in ("Mozilla", "Chrome", "Safari")),
            f"User-Agent does not look like a browser: {ua!r}",
        )

    @patch("data.aaii.requests.get")
    def test_fraction_normalisation(self, mock_get):
        """Values stored as 0-1 fractions should be scaled to 0-100."""
        dates = pd.date_range("2024-01-01", periods=4, freq="W-FRI")
        df_raw = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d").tolist(),
            "Bullish": [0.40, 0.42, 0.38, 0.45],
            "Bearish": [0.30, 0.28, 0.35, 0.25],
            "Neutral": [0.30, 0.30, 0.27, 0.30],
        })
        buf = io.BytesIO()
        df_raw.to_excel(buf, index=False)
        buf.seek(0)

        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.content = buf.read()
        resp.text = ""
        mock_get.return_value = resp

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertFalse(df.empty)
        self.assertGreater(float(df["bullish"].max()), 1.0)
        self.assertAlmostEqual(float(df.iloc[-1]["bullish"]), 45.0, places=1)


class TestLatestAaii(unittest.TestCase):
    """latest_aaii should return the most-recent row with z-score."""

    def _patch_get_aaii(self, df: pd.DataFrame):
        return patch("data.aaii.get_aaii_sentiment", return_value=df)

    def test_returns_correct_keys(self):
        dates = pd.date_range("2023-01-01", periods=260, freq="W-FRI")
        df = pd.DataFrame({
            "date": dates,
            "bullish": np.random.uniform(30, 50, 260),
            "bearish": np.random.uniform(20, 40, 260),
            "neutral": np.random.uniform(20, 35, 260),
        })
        df["bull_bear_spread"] = df["bullish"] - df["bearish"]
        df = df.sort_values("date").reset_index(drop=True)

        with self._patch_get_aaii(df):
            from data.aaii import latest_aaii
            result = latest_aaii()

        self.assertIn("date", result)
        self.assertIn("bullish", result)
        self.assertIn("bearish", result)
        self.assertIn("neutral", result)
        self.assertIn("bull_bear_spread", result)
        self.assertIn("bull_bear_zscore_5y", result)

    def test_returns_empty_dict_on_empty_df(self):
        empty = pd.DataFrame(
            columns=["date", "bullish", "bearish", "neutral", "bull_bear_spread"]
        )
        with self._patch_get_aaii(empty):
            from data.aaii import latest_aaii
            result = latest_aaii()
        self.assertEqual(result, {})

    def test_date_is_iso_string(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="W-FRI")
        df = pd.DataFrame({
            "date": dates,
            "bullish": [40.0] * 10,
            "bearish": [30.0] * 10,
            "neutral": [30.0] * 10,
            "bull_bear_spread": [10.0] * 10,
        })
        with self._patch_get_aaii(df):
            from data.aaii import latest_aaii
            result = latest_aaii()

        self.assertIsInstance(result["date"], str)
        # Should be YYYY-MM-DD
        self.assertRegex(result["date"], r"^\d{4}-\d{2}-\d{2}$")

    def test_zscore_sign_makes_sense(self):
        """Extreme bullishness (large spread) should produce a high positive z-score."""
        dates = pd.date_range("2020-01-01", periods=260, freq="W-FRI")
        spreads = [10.0] * 259 + [50.0]  # last row is an extreme outlier
        df = pd.DataFrame({
            "date": dates,
            "bullish": [50.0] * 259 + [70.0],
            "bearish": [40.0] * 259 + [20.0],
            "neutral": [10.0] * 260,
            "bull_bear_spread": spreads,
        })
        with self._patch_get_aaii(df):
            from data.aaii import latest_aaii
            result = latest_aaii()

        self.assertGreater(result["bull_bear_zscore_5y"], 2.0)


# ---------------------------------------------------------------------------
# Column-detection helper tests
# ---------------------------------------------------------------------------

class TestFindColumn(unittest.TestCase):
    """_find_column should handle mixed cases, extra whitespace, partial matches."""

    def setUp(self):
        from data.aaii import _find_column
        self.find = _find_column

    def test_exact_match(self):
        self.assertEqual(self.find(["Bullish", "Bearish"], "bullish"), "Bullish")

    def test_case_insensitive(self):
        self.assertEqual(self.find(["BULLISH", "BEARISH"], "bullish"), "BULLISH")

    def test_partial_match(self):
        self.assertEqual(self.find(["% Bullish", "% Bearish"], "bullish"), "% Bullish")

    def test_returns_none_when_not_found(self):
        self.assertIsNone(self.find(["Alpha", "Beta"], "bullish"))

    def test_empty_columns(self):
        self.assertIsNone(self.find([], "bullish"))


if __name__ == "__main__":
    unittest.main()
