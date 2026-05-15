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

    @patch("data.fred.requests.get", side_effect=Exception("network down"))
    @patch("data.fred.web.DataReader", side_effect=Exception("network down"))
    def test_returns_empty_dict_on_failure(self, mock_dr, mock_req):
        """When both pandas-datareader AND the direct-CSV fallback fail,
        the function must return ``{}`` (and not raise)."""
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

    @patch("data.fred.requests.get", side_effect=Exception("fail"))
    @patch("data.fred.web.DataReader", side_effect=Exception("fail"))
    def test_returns_empty_on_failure(self, mock_dr, mock_req):
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
# AAII HTML fallback tests
# ---------------------------------------------------------------------------

# A realistic AAII results-page snippet. The live page renders the three
# headline percentages inline with their labels and a "Week Ending" date.
_AAII_HTML_FIXTURE = """
<html>
  <body>
    <h2>AAII Investor Sentiment Survey</h2>
    <p>Week Ending November 6, 2025</p>
    <div class="results">
      <span class="label">Bullish</span> <span class="pct">39.32%</span>
      <span class="label">Bearish</span> <span class="pct">36.61%</span>
      <span class="label">Neutral</span> <span class="pct">24.07%</span>
    </div>
  </body>
</html>
"""

# Imperva challenge page — what AAII serves when it blocks us.
_IMPERVA_HTML = """
<!DOCTYPE html><html><head>
<noscript><title>Pardon Our Interruption</title></noscript>
<script>window.onProtectionInitialized = function(){};</script>
</head><body></body></html>
"""


def _clear_aaii_warn_cache():
    """Reset the per-process warning cache so order-dependent tests are stable."""
    from data import aaii as _aaii
    _aaii._WARN_CACHE.clear()


def _clear_fred_warn_cache():
    from data import fred as _fred
    _fred._WARN_CACHE.clear()


class TestAaiiHtmlFallback(unittest.TestCase):
    """When the XLS endpoint returns an Imperva challenge page or
    otherwise fails to parse, get_aaii_sentiment must scrape the
    public HTML results page instead."""

    def setUp(self):
        _clear_aaii_warn_cache()

    def _mock_response(self, text: str, content: bytes | None = None):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = text
        resp.content = content if content is not None else text.encode("utf-8")
        return resp

    @patch("data.aaii.requests.get")
    def test_html_fallback_when_xls_is_imperva(self, mock_get):
        """Imperva on the XLS endpoint should trigger HTML scrape with the
        correct percentages parsed from the fallback page."""
        # First call (XLS) returns Imperva HTML; second call (HTML
        # fallback) returns the real results page.
        mock_get.side_effect = [
            self._mock_response(_IMPERVA_HTML),
            self._mock_response(_AAII_HTML_FIXTURE),
        ]

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)  # HTML fallback only returns one week
        self.assertAlmostEqual(float(df.iloc[-1]["bullish"]), 39.32, places=2)
        self.assertAlmostEqual(float(df.iloc[-1]["bearish"]), 36.61, places=2)
        self.assertAlmostEqual(float(df.iloc[-1]["neutral"]), 24.07, places=2)
        # spread = bullish - bearish
        self.assertAlmostEqual(
            float(df.iloc[-1]["bull_bear_spread"]), 39.32 - 36.61, places=2
        )

    @patch("data.aaii.requests.get")
    def test_html_fallback_when_xls_fails_to_parse(self, mock_get):
        """A non-Imperva, non-parseable XLS body should still trigger the
        HTML fallback (rather than just returning an empty DF)."""
        mock_get.side_effect = [
            self._mock_response("garbage body that is not xls or csv"),
            self._mock_response(_AAII_HTML_FIXTURE),
        ]

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df.iloc[-1]["bullish"]), 39.32, places=2)

    @patch("data.aaii.requests.get")
    def test_html_fallback_imperva_also_blocked(self, mock_get):
        """If the HTML results page is ALSO behind Imperva, return empty
        DataFrame (do not raise) and log a clean warning."""
        mock_get.side_effect = [
            self._mock_response(_IMPERVA_HTML),
            self._mock_response(_IMPERVA_HTML),
        ]

        from data.aaii import get_aaii_sentiment

        df = get_aaii_sentiment()
        self.assertTrue(df.empty)
        # Columns must still match the public contract.
        self.assertEqual(
            list(df.columns),
            ["date", "bullish", "bearish", "neutral", "bull_bear_spread"],
        )

    @patch("data.aaii.requests.get")
    def test_html_fallback_returns_correct_dict_shape_via_latest_aaii(self, mock_get):
        """latest_aaii() must return the full dict shape — including a
        ``note`` field and zero z-score — when only the HTML fallback
        succeeds."""
        mock_get.side_effect = [
            self._mock_response(_IMPERVA_HTML),
            self._mock_response(_AAII_HTML_FIXTURE),
        ]

        from data.aaii import latest_aaii

        result = latest_aaii()
        # Same keys as the normal path
        for key in ("date", "bullish", "bearish", "neutral",
                    "bull_bear_spread", "bull_bear_zscore_5y"):
            self.assertIn(key, result)
        # Z-score must be neutral (0.0) when only the single week is known
        self.assertEqual(result["bull_bear_zscore_5y"], 0.0)
        # And we must document the limitation
        self.assertIn("note", result)
        self.assertIn("HTML scrape", result["note"])

    @patch("data.aaii.requests.get")
    def test_warning_dedupe_on_multiple_calls(self, mock_get):
        """Calling get_aaii_sentiment twice in a row when AAII is fully
        blocked must NOT spam the same warning twice."""
        # Four mock responses — two pairs (XLS + HTML fallback), all Imperva.
        mock_get.side_effect = [
            self._mock_response(_IMPERVA_HTML),
            self._mock_response(_IMPERVA_HTML),
            self._mock_response(_IMPERVA_HTML),
            self._mock_response(_IMPERVA_HTML),
        ]

        from data.aaii import get_aaii_sentiment, _WARN_CACHE

        import io as _io
        import contextlib

        # First run primes the cache.
        get_aaii_sentiment()

        # Second run: capture stderr — there must be no new lines printed.
        buf = _io.StringIO()
        with contextlib.redirect_stderr(buf):
            get_aaii_sentiment()
        self.assertEqual(
            buf.getvalue(), "",
            f"Expected no duplicate warnings, got: {buf.getvalue()!r}",
        )
        # And the cache must contain the messages we recorded the first time.
        self.assertTrue(len(_WARN_CACHE) >= 1)


class TestAaiiHtmlFallbackHelpers(unittest.TestCase):
    """Direct unit tests for the regex helpers used by the HTML scrape."""

    def test_extract_pct_basic(self):
        from data.aaii import _extract_pct
        self.assertAlmostEqual(
            _extract_pct("Bullish 39.32%", "bullish"), 39.32, places=2
        )

    def test_extract_pct_with_markup(self):
        """Regex must work across HTML tags — the live AAII results page
        renders the label and the percent number in separate <span>s."""
        from data.aaii import _extract_pct
        html = '<span>Bearish</span> <strong>36.6%</strong>'
        result = _extract_pct(html, "bearish")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 36.6, places=1)

    def test_extract_pct_label_not_found(self):
        from data.aaii import _extract_pct
        self.assertIsNone(_extract_pct("nothing relevant here", "bullish"))

    def test_extract_date_week_of(self):
        from data.aaii import _extract_date
        self.assertEqual(
            _extract_date("Week Ending November 6, 2025"),
            "2025-11-06",
        )

    def test_extract_date_iso(self):
        from data.aaii import _extract_date
        self.assertEqual(_extract_date("data as of 2025-11-06"), "2025-11-06")

    def test_extract_date_missing(self):
        from data.aaii import _extract_date
        self.assertIsNone(_extract_date("no date in here at all"))

    def test_looks_like_imperva(self):
        from data.aaii import _looks_like_imperva
        self.assertTrue(_looks_like_imperva(_IMPERVA_HTML))
        self.assertFalse(_looks_like_imperva(_AAII_HTML_FIXTURE))
        self.assertFalse(_looks_like_imperva(""))


# ---------------------------------------------------------------------------
# FRED direct-CSV fallback tests
# ---------------------------------------------------------------------------

# A minimal FRED CSV — real responses have this exact shape.
_FRED_CSV_FIXTURE = """DATE,BAMLH0A0HYM2
2025-10-01,3.45
2025-10-02,3.50
2025-10-03,.
2025-10-04,3.48
"""


class TestFetchFredDirect(unittest.TestCase):
    """_fetch_fred_direct should parse the public fredgraph.csv endpoint
    without needing pandas-datareader."""

    def setUp(self):
        _clear_fred_warn_cache()

    def _mock_csv_response(self, text: str):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = text
        return resp

    @patch("data.fred.requests.get")
    def test_parses_csv_response(self, mock_get):
        mock_get.return_value = self._mock_csv_response(_FRED_CSV_FIXTURE)

        from data.fred import _fetch_fred_direct

        # Use very large lookback so we keep all rows from 2025-10.
        series = _fetch_fred_direct("BAMLH0A0HYM2", lookback_days=100000)
        self.assertIsNotNone(series)
        self.assertIsInstance(series, pd.Series)
        # FRED uses '.' for missing — must be dropped.
        self.assertEqual(len(series), 3)
        # Sorted ascending
        self.assertTrue(series.index.is_monotonic_increasing)
        self.assertAlmostEqual(float(series.iloc[-1]), 3.48, places=2)

    @patch("data.fred.requests.get", side_effect=Exception("network down"))
    def test_returns_none_on_http_failure(self, mock_get):
        from data.fred import _fetch_fred_direct
        self.assertIsNone(_fetch_fred_direct("BAMLH0A0HYM2"))

    @patch("data.fred.requests.get")
    def test_returns_none_on_empty_body(self, mock_get):
        mock_get.return_value = self._mock_csv_response("")
        from data.fred import _fetch_fred_direct
        self.assertIsNone(_fetch_fred_direct("BAMLH0A0HYM2"))

    @patch("data.fred.requests.get")
    def test_returns_none_on_unparseable_body(self, mock_get):
        mock_get.return_value = self._mock_csv_response("totally not csv {[")
        from data.fred import _fetch_fred_direct
        # Either None (single column = not enough data) or a parsed series
        # whose date column fails to coerce. Either way, must not crash.
        result = _fetch_fred_direct("BAMLH0A0HYM2")
        self.assertIsNone(result)


class TestFredFallbackChain(unittest.TestCase):
    """get_fred_series must transparently fall back from pandas-datareader
    to the direct CSV path when DataReader is unavailable or fails."""

    def setUp(self):
        _clear_fred_warn_cache()

    def _mock_csv_response(self, text: str):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = text
        return resp

    def _mock_csv_for(self, fred_id: str) -> str:
        return (
            f"DATE,{fred_id}\n"
            "2025-10-01,1.0\n"
            "2025-10-02,2.0\n"
            "2025-10-03,3.0\n"
        )

    @patch("data.fred.web.DataReader", side_effect=Exception("deprecate_kwarg fail"))
    @patch("data.fred.requests.get")
    def test_falls_back_to_direct_csv_when_datareader_raises(self, mock_req, mock_dr):
        """When pandas-datareader.DataReader raises, the direct CSV path
        must be tried for each series."""
        def _side_effect(url, params=None, headers=None, timeout=None, **kwargs):
            fred_id = (params or {}).get("id", "")
            return self._mock_csv_response(self._mock_csv_for(fred_id))

        mock_req.side_effect = _side_effect

        from data.fred import get_fred_series, FRED_SERIES

        result = get_fred_series(FRED_SERIES, lookback_days=100000)
        # All three series should be populated via the CSV fallback.
        self.assertEqual(set(result.keys()), set(FRED_SERIES.keys()))
        for name, series in result.items():
            self.assertIsInstance(series, pd.Series)
            self.assertGreater(len(series), 0)
            self.assertTrue(series.index.is_monotonic_increasing)

    @patch("data.fred.requests.get")
    def test_falls_back_when_web_is_none(self, mock_req):
        """When ``web is None`` (e.g. pandas-datareader import failed on
        CI), the direct CSV fallback must still produce results."""
        def _side_effect(url, params=None, headers=None, timeout=None, **kwargs):
            fred_id = (params or {}).get("id", "")
            return self._mock_csv_response(self._mock_csv_for(fred_id))

        mock_req.side_effect = _side_effect

        # Temporarily set data.fred.web to None to simulate the CI failure mode.
        from data import fred as _fred_mod
        original = _fred_mod.web
        _fred_mod.web = None
        try:
            from data.fred import get_fred_series, FRED_SERIES
            result = get_fred_series(FRED_SERIES, lookback_days=100000)
        finally:
            _fred_mod.web = original

        self.assertEqual(set(result.keys()), set(FRED_SERIES.keys()))
        for series in result.values():
            self.assertGreater(len(series), 0)

    @patch("data.fred.requests.get")
    def test_latest_fred_via_direct_csv(self, mock_req):
        """latest_fred() must return float values when only the direct
        CSV fallback is available."""
        def _side_effect(url, params=None, headers=None, timeout=None, **kwargs):
            fred_id = (params or {}).get("id", "")
            return self._mock_csv_response(self._mock_csv_for(fred_id))

        mock_req.side_effect = _side_effect

        from data import fred as _fred_mod
        original = _fred_mod.web
        _fred_mod.web = None
        try:
            from data.fred import latest_fred, FRED_SERIES
            result = latest_fred(FRED_SERIES)
        finally:
            _fred_mod.web = original

        self.assertEqual(set(result.keys()), set(FRED_SERIES.keys()))
        for v in result.values():
            self.assertIsInstance(v, float)
            self.assertEqual(v, 3.0)  # last row of every fixture


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
