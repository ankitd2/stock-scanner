"""
tests/test_indicators.py

Unit tests for analytics/indicators.py.
No network calls — all data is synthetic.
"""

import math
import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable regardless of working directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.indicators import (
    rsi,
    macd,
    bollinger,
    bollinger_width,
    atr,
    sma,
    ema,
    pct_from_high,
    pct_from_low,
    relative_strength,
    vol_adjusted_momentum,
    zscore,
    compute_all,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_close(n: int = 300, start: float = 100.0, drift: float = 0.001) -> pd.Series:
    """Synthetic close price series with slight upward drift and noise."""
    rng = np.random.default_rng(42)
    returns = rng.normal(drift, 0.01, n)
    prices = start * np.cumprod(1 + returns)
    return pd.Series(prices, dtype=float)


def make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(99)
    close = make_close(n)
    noise = rng.uniform(0.005, 0.015, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_length_preserved(self):
        c = make_close(100)
        result = rsi(c)
        assert len(result) == len(c)

    def test_first_period_nan(self):
        c = make_close(50)
        result = rsi(c, period=14)
        assert result.iloc[:14].isna().all(), "First 14 values should be NaN"

    def test_values_in_range(self):
        c = make_close(300)
        result = rsi(c, period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_empty_series(self):
        result = rsi(pd.Series(dtype=float))
        assert len(result) == 0

    def test_series_shorter_than_period(self):
        c = make_close(10)
        result = rsi(c, period=14)
        assert result.isna().all()

    def test_nan_input(self):
        c = pd.Series([np.nan] * 20)
        result = rsi(c, period=14)
        assert len(result) == 20

    def test_constant_series_sell_pressure(self):
        # All gains = 0, all losses = 0 — RSI is undefined; no crash
        c = pd.Series([100.0] * 30)
        result = rsi(c, period=14)
        assert len(result) == len(c)

    def test_monotone_up_high_rsi(self):
        # Steadily rising prices → RSI should be high (close to 100)
        c = pd.Series(np.linspace(1, 100, 60))
        result = rsi(c, period=14)
        valid = result.dropna()
        assert (valid > 70).all()

    def test_monotone_down_low_rsi(self):
        # Steadily falling prices → RSI should be low (close to 0)
        c = pd.Series(np.linspace(100, 1, 60))
        result = rsi(c, period=14)
        valid = result.dropna()
        assert (valid < 30).all()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_lengths_match(self):
        c = make_close(300)
        ml, sl, h = macd(c)
        assert len(ml) == len(c)
        assert len(sl) == len(c)
        assert len(h) == len(c)

    def test_histogram_is_diff(self):
        c = make_close(300)
        ml, sl, h = macd(c)
        pd.testing.assert_series_equal(h, ml - sl)

    def test_empty_series(self):
        ml, sl, h = macd(pd.Series(dtype=float))
        assert len(ml) == 0
        assert len(sl) == 0
        assert len(h) == 0

    def test_short_series(self):
        c = make_close(5)
        ml, sl, h = macd(c)
        assert len(ml) == 5

    def test_custom_params(self):
        c = make_close(300)
        ml, sl, h = macd(c, fast=8, slow=21, signal=5)
        assert len(ml) == len(c)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_lengths_match(self):
        c = make_close(200)
        u, m, l = bollinger(c)
        assert len(u) == len(m) == len(l) == len(c)

    def test_upper_above_lower(self):
        c = make_close(200)
        u, m, l = bollinger(c)
        valid = ~u.isna()
        assert (u[valid] >= l[valid]).all()

    def test_middle_is_sma(self):
        c = make_close(200)
        _, m, _ = bollinger(c, period=20)
        expected = c.rolling(20, min_periods=20).mean()
        pd.testing.assert_series_equal(m, expected, check_names=False)

    def test_first_n_nan(self):
        c = make_close(200)
        u, m, l = bollinger(c, period=20)
        assert u.iloc[:19].isna().all()

    def test_empty_series(self):
        u, m, l = bollinger(pd.Series(dtype=float))
        assert len(u) == 0

    def test_std_dev_scaling(self):
        c = make_close(200)
        u1, m1, l1 = bollinger(c, period=20, std_dev=1.0)
        u2, m2, l2 = bollinger(c, period=20, std_dev=2.0)
        # Width at 2 std should be exactly double that at 1 std
        valid = ~u1.isna()
        np.testing.assert_allclose(
            (u2[valid] - l2[valid]).values,
            2 * (u1[valid] - l1[valid]).values,
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Bollinger Width
# ---------------------------------------------------------------------------

class TestBollingerWidth:
    def test_positive_values(self):
        c = make_close(200)
        w = bollinger_width(c)
        valid = w.dropna()
        assert (valid >= 0).all()

    def test_length_preserved(self):
        c = make_close(100)
        w = bollinger_width(c)
        assert len(w) == len(c)

    def test_empty_series(self):
        w = bollinger_width(pd.Series(dtype=float))
        assert len(w) == 0


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_length_preserved(self):
        df = make_ohlcv(200)
        result = atr(df["High"], df["Low"], df["Close"])
        assert len(result) == 200

    def test_positive_values(self):
        df = make_ohlcv(200)
        result = atr(df["High"], df["Low"], df["Close"])
        valid = result.dropna()
        assert (valid > 0).all()

    def test_empty_series(self):
        empty = pd.Series(dtype=float)
        result = atr(empty, empty, empty)
        assert len(result) == 0

    def test_short_series(self):
        df = make_ohlcv(5)
        result = atr(df["High"], df["Low"], df["Close"])
        assert len(result) == 5


# ---------------------------------------------------------------------------
# SMA / EMA
# ---------------------------------------------------------------------------

class TestSMAEMA:
    def test_sma_length(self):
        c = make_close(100)
        assert len(sma(c, 20)) == 100

    def test_sma_first_n_nan(self):
        c = make_close(100)
        result = sma(c, 20)
        assert result.iloc[:19].isna().all()

    def test_sma_rolling_correctness(self):
        c = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(c, 3)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)

    def test_ema_length(self):
        c = make_close(100)
        assert len(ema(c, 12)) == 100

    def test_ema_no_nan_after_start(self):
        # EWM with adjust=False returns values from first element
        c = make_close(100)
        result = ema(c, 12)
        assert not result.isna().any()

    def test_sma_empty(self):
        assert len(sma(pd.Series(dtype=float), 20)) == 0

    def test_ema_empty(self):
        assert len(ema(pd.Series(dtype=float), 12)) == 0


# ---------------------------------------------------------------------------
# pct_from_high / pct_from_low
# ---------------------------------------------------------------------------

class TestPctFromHighLow:
    def test_at_high_is_zero(self):
        # Last value is the max → should be 0
        c = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pct_from_high(c, lookback=252)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_below_high_is_negative(self):
        c = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        result = pct_from_high(c, lookback=252)
        # After first element (the all-time high), all values should be negative
        assert (result.iloc[1:] < 0).all()

    def test_at_low_is_zero(self):
        c = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        result = pct_from_low(c, lookback=252)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_above_low_is_positive(self):
        c = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pct_from_low(c, lookback=252)
        assert (result.iloc[1:] > 0).all()

    def test_length_preserved(self):
        c = make_close(200)
        assert len(pct_from_high(c)) == 200
        assert len(pct_from_low(c)) == 200

    def test_empty_series(self):
        assert len(pct_from_high(pd.Series(dtype=float))) == 0
        assert len(pct_from_low(pd.Series(dtype=float))) == 0


# ---------------------------------------------------------------------------
# Relative Strength
# ---------------------------------------------------------------------------

class TestRelativeStrength:
    def test_returns_float(self):
        asset = make_close(300)
        bench = make_close(300, drift=0.0005)
        result = relative_strength(asset, bench, period=63)
        assert isinstance(result, float)

    def test_outperforming_asset(self):
        # Asset doubles, benchmark flat
        n = 100
        asset = pd.Series(np.linspace(100, 200, n))
        bench = pd.Series(np.full(n, 100.0))
        result = relative_strength(asset, bench, period=63)
        assert result > 0

    def test_underperforming_asset(self):
        n = 100
        asset = pd.Series(np.full(n, 100.0))
        bench = pd.Series(np.linspace(100, 200, n))
        result = relative_strength(asset, bench, period=63)
        assert result < 0

    def test_insufficient_data(self):
        asset = make_close(10)
        bench = make_close(10)
        result = relative_strength(asset, bench, period=63)
        assert math.isnan(result)

    def test_none_input(self):
        result = relative_strength(None, None)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Vol-Adjusted Momentum
# ---------------------------------------------------------------------------

class TestVolAdjustedMomentum:
    def test_returns_float_for_long_series(self):
        c = make_close(300)
        result = vol_adjusted_momentum(c)
        assert result is not None
        assert isinstance(result, float)

    def test_returns_none_for_short_series(self):
        c = make_close(100)
        result = vol_adjusted_momentum(c, lookback=252)
        assert result is None

    def test_returns_none_for_empty(self):
        result = vol_adjusted_momentum(pd.Series(dtype=float))
        assert result is None

    def test_positive_for_trending_up(self):
        # Strongly trending upward series → positive momentum
        c = pd.Series(np.linspace(1, 300, 300))
        result = vol_adjusted_momentum(c, skip_recent=21, lookback=252)
        assert result is not None and result > 0

    def test_negative_for_trending_down(self):
        c = pd.Series(np.linspace(300, 1, 300))
        result = vol_adjusted_momentum(c, skip_recent=21, lookback=252)
        assert result is not None and result < 0


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------

class TestZScore:
    def test_length_preserved(self):
        s = make_close(300)
        result = zscore(s)
        assert len(result) == 300

    def test_first_window_nan(self):
        s = make_close(300)
        result = zscore(s, window=50)
        assert result.iloc[:49].isna().all()

    def test_mean_zero(self):
        # Use a stationary (mean-reverting) series for z-score mean test.
        # A trending series has z-scores biased positive due to drift.
        rng = np.random.default_rng(123)
        s = pd.Series(rng.normal(0, 1, 500))  # pure white noise, mean=0
        result = zscore(s, window=100)
        valid = result.dropna()
        # Z-score of white noise should have mean close to 0
        assert abs(valid.mean()) < 0.5

    def test_empty_series(self):
        assert len(zscore(pd.Series(dtype=float))) == 0


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_returns_dict(self):
        df = make_ohlcv(300)
        result = compute_all(df)
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        expected_keys = {
            "rsi_14", "macd_line", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr_14", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
            "pct_from_52wh", "pct_from_52wl",
            "vol_adj_mom", "latest_rsi", "latest_close",
            "above_50dma", "above_200dma", "golden_cross", "volume_ratio",
        }
        df = make_ohlcv(300)
        result = compute_all(df)
        assert expected_keys == set(result.keys())

    def test_short_df_returns_empty(self):
        df = make_ohlcv(20)
        result = compute_all(df)
        assert result == {}

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = compute_all(df)
        assert result == {}

    def test_none_returns_empty(self):
        result = compute_all(None)
        assert result == {}

    def test_missing_column_returns_empty(self):
        df = make_ohlcv(300).drop(columns=["Volume"])
        result = compute_all(df)
        assert result == {}

    def test_scalar_types(self):
        df = make_ohlcv(300)
        result = compute_all(df)
        assert isinstance(result["latest_close"], float)
        assert isinstance(result["latest_rsi"], float)
        assert isinstance(result["above_50dma"], bool)
        assert isinstance(result["above_200dma"], bool)
        assert isinstance(result["golden_cross"], bool)
        assert isinstance(result["volume_ratio"], float)

    def test_series_types(self):
        df = make_ohlcv(300)
        result = compute_all(df)
        series_keys = [
            "rsi_14", "macd_line", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr_14", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
            "pct_from_52wh", "pct_from_52wl",
        ]
        for key in series_keys:
            assert isinstance(result[key], pd.Series), f"{key} should be a Series"
            assert len(result[key]) == 300, f"{key} should have 300 elements"

    def test_latest_close_matches_df(self):
        df = make_ohlcv(300)
        result = compute_all(df)
        assert result["latest_close"] == pytest.approx(df["Close"].iloc[-1])

    def test_exception_safety(self):
        # Corrupt DataFrame — should return {} not raise
        bad_df = pd.DataFrame({"Close": ["not", "a", "number"] * 100})
        result = compute_all(bad_df)
        assert result == {}

    def test_column_case_insensitivity(self):
        # compute_all should normalise lowercase column names
        df = make_ohlcv(300)
        df.columns = [c.lower() for c in df.columns]
        result = compute_all(df)
        assert len(result) > 0

    def test_volume_ratio_positive(self):
        df = make_ohlcv(300)
        result = compute_all(df)
        assert result["volume_ratio"] > 0

    def test_sma_ordering(self):
        # For upward trending series: sma_20 > sma_50 > sma_200 at the end
        n = 400
        c = pd.Series(np.linspace(1, 400, n))
        rng = np.random.default_rng(7)
        noise = rng.uniform(0.005, 0.01, n)
        df = pd.DataFrame({
            "Open": c * 0.999,
            "High": c * (1 + noise),
            "Low": c * (1 - noise),
            "Close": c,
            "Volume": np.ones(n) * 1_000_000,
        })
        result = compute_all(df)
        assert result["sma_20"].iloc[-1] > result["sma_50"].iloc[-1]
        assert result["sma_50"].iloc[-1] > result["sma_200"].iloc[-1]
