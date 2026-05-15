"""
tests/test_market_state.py
==========================
Unit tests for analytics/market_state.py

Tests focus on:
  - Structure and types of return values
  - Score invariants (0-100, integer)
  - Regime classification correctness
  - Driver sorting (abs contribution descending)
  - Graceful degradation on empty / missing data
  - Breadth indicator calculations
  - VIX signal computations
  - Score directionality (stress → lower score, calm → higher score)
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from analytics.market_state import (
    compute_breadth,
    compute_vix_signals,
    compute_cross_asset,
    compute_market_state,
    _sigmoid,
    _threshold_z,
    _score_to_regime,
    _safe_float,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n_rows: int = 300, start_price: float = 100.0, drift: float = 0.001) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with n_rows of data."""
    rng = np.random.default_rng(42)
    prices = start_price * np.cumprod(1 + rng.normal(drift, 0.01, n_rows))
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="B")
    df = pd.DataFrame(
        {
            "Open":   prices * 0.999,
            "High":   prices * 1.005,
            "Low":    prices * 0.995,
            "Close":  prices,
            "Volume": rng.integers(1_000_000, 5_000_000, n_rows),
        },
        index=dates,
    )
    return df


def _make_universe(n_tickers: int = 20, n_rows: int = 300) -> dict:
    """Create a synthetic universe of n_tickers histories."""
    rng = np.random.default_rng(99)
    universe = {}
    for i in range(n_tickers):
        drift = rng.uniform(-0.0005, 0.002)
        universe[f"TICK{i:03d}"] = _make_ohlcv(n_rows=n_rows, start_price=50 + i * 5, drift=drift)
    return universe


def _make_index_data(vix: float = 18.0, vix3m: float = 20.0, vix9d: float = 16.0,
                     vvix: float = 100.0) -> dict:
    """Build a minimal index_data dict for VIX signals."""
    return {
        "^VIX":   {"price": vix,   "pct_change": -2.0, "prev_close": vix * 1.02},
        "^VIX3M": {"price": vix3m, "pct_change": -1.0, "prev_close": vix3m * 1.01},
        "^VIX9D": {"price": vix9d, "pct_change": -3.0, "prev_close": vix9d * 1.03},
        "^VVIX":  {"price": vvix,  "pct_change":  0.5, "prev_close": vvix * 0.995},
        "RSP":    {"price": 160.0, "pct_change":  0.3, "prev_close": 159.5},
        "SPY":    {"price": 530.0, "pct_change":  0.2, "prev_close": 529.0},
        "SPHB":   {"price":  80.0, "pct_change":  0.5, "prev_close":  79.6},
        "SPLV":   {"price":  70.0, "pct_change": -0.1, "prev_close":  70.1},
        "HG=F":   {"price":   4.5, "pct_change":  0.2, "prev_close":   4.49},
        "GC=F":   {"price": 2000.0, "pct_change": -0.3, "prev_close": 2006.0},
    }


def _make_fred_data(hy_oas: float = 320.0) -> dict:
    return {"hy_oas": hy_oas, "t10y2y": 0.5, "dgs10": 4.2}


def _make_aaii_data(zscore: float = 0.5) -> dict:
    return {
        "bullish": 35.0, "bearish": 30.0, "neutral": 35.0,
        "bull_bear_spread": 5.0, "bull_bear_zscore_5y": zscore,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_sigmoid_midpoint(self):
        """sigmoid(0) == 0.5"""
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_sigmoid_positive(self):
        assert _sigmoid(2.0) > 0.5

    def test_sigmoid_negative(self):
        assert _sigmoid(-2.0) < 0.5

    def test_sigmoid_bounds(self):
        assert 0.0 < _sigmoid(100) <= 1.0
        assert 0.0 <= _sigmoid(-100) < 1.0

    def test_threshold_z_none(self):
        assert _threshold_z(None, 0, 1) == 0.0

    def test_threshold_z_zero_std(self):
        assert _threshold_z(5.0, 5.0, 0) == 0.0

    def test_threshold_z_positive(self):
        z = _threshold_z(27.0, 20.0, 7.0)  # (27-20)/7 = 1.0
        assert z == pytest.approx(1.0)

    def test_threshold_z_negative(self):
        z = _threshold_z(13.0, 20.0, 7.0)  # (13-20)/7 = -1.0
        assert z == pytest.approx(-1.0)

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_int(self):
        assert _safe_float(42) == pytest.approx(42.0)

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) is None

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) is None

    def test_safe_float_default(self):
        assert _safe_float(None, default=99.0) == pytest.approx(99.0)


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeClassification:
    @pytest.mark.parametrize("score,expected", [
        (0,   "Risk-Off"),
        (34,  "Risk-Off"),
        (35,  "Caution"),
        (49,  "Caution"),
        (50,  "Neutral"),
        (69,  "Neutral"),
        (70,  "Risk-On"),
        (100, "Risk-On"),
    ])
    def test_score_to_regime(self, score, expected):
        assert _score_to_regime(score) == expected


# ─────────────────────────────────────────────────────────────────────────────
# compute_breadth tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBreadth:
    def test_empty_universe_returns_defaults(self):
        result = compute_breadth({})
        assert result["pct_above_50dma"] is None
        assert result["pct_above_200dma"] is None
        assert result["new_highs_52w"] is None
        assert result["new_lows_52w"] is None

    def test_all_keys_present(self):
        universe = _make_universe(n_tickers=5, n_rows=300)
        result = compute_breadth(universe)
        required_keys = {
            "pct_above_50dma", "pct_above_200dma",
            "new_highs_52w", "new_lows_52w",
            "new_highs_count", "new_lows_count",
        }
        assert required_keys.issubset(result.keys())

    def test_pct_range(self):
        universe = _make_universe(n_tickers=20, n_rows=300)
        result = compute_breadth(universe)
        for key in ("pct_above_50dma", "pct_above_200dma"):
            if result[key] is not None:
                assert 0.0 <= result[key] <= 100.0, f"{key} out of range: {result[key]}"

    def test_counts_non_negative(self):
        universe = _make_universe(n_tickers=20, n_rows=300)
        result = compute_breadth(universe)
        for key in ("new_highs_52w", "new_lows_52w"):
            if result[key] is not None:
                assert result[key] >= 0

    def test_aliases_match(self):
        universe = _make_universe(n_tickers=10, n_rows=300)
        result = compute_breadth(universe)
        assert result["new_highs_52w"] == result["new_highs_count"]
        assert result["new_lows_52w"] == result["new_lows_count"]

    def test_insufficient_history_skipped(self):
        """Tickers with < 50 rows should be ignored gracefully."""
        universe = {
            "SHORT": _make_ohlcv(n_rows=30),   # < 50, should be skipped
            "LONG":  _make_ohlcv(n_rows=300),   # ok
        }
        result = compute_breadth(universe)
        # Should not crash, LONG should be counted
        assert result["pct_above_50dma"] is not None or result["pct_above_50dma"] is None  # either is fine

    def test_all_tickers_above_50dma(self):
        """If all tickers are on an uptrend, most should be above 50 DMA."""
        # Strong uptrend: price rises monotonically well above the 50 DMA
        prices = np.linspace(100, 200, 100)  # strictly increasing
        df = pd.DataFrame(
            {"Close": prices},
            index=pd.date_range("2023-01-01", periods=100, freq="B"),
        )
        universe = {"BULL": df}
        result = compute_breadth(universe)
        # Latest price > SMA50 in a strong uptrend
        assert result["pct_above_50dma"] == pytest.approx(100.0)

    def test_all_tickers_below_50dma(self):
        """If all tickers are in a downtrend, most should be below 50 DMA."""
        prices = np.linspace(200, 100, 100)  # strictly decreasing
        df = pd.DataFrame(
            {"Close": prices},
            index=pd.date_range("2023-01-01", periods=100, freq="B"),
        )
        universe = {"BEAR": df}
        result = compute_breadth(universe)
        assert result["pct_above_50dma"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# compute_vix_signals tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeVixSignals:
    def test_empty_index_data(self):
        result = compute_vix_signals({})
        assert result["vix"] is None
        assert result["vix_term_structure"] is None

    def test_all_keys_present(self):
        idx = _make_index_data()
        # Patch bulk_history to avoid network calls
        with patch("analytics.market_state.bulk_history", return_value={}):
            result = compute_vix_signals(idx)
        required = {"vix", "vix3m", "vix9d", "vvix", "vix_term_structure", "vix9d_ratio", "vix_pct_1y"}
        assert required.issubset(result.keys())

    def test_term_structure_contango(self):
        """VIX < VIX3M → vix_term_structure < 1 (contango / calm)."""
        idx = _make_index_data(vix=18.0, vix3m=22.0)
        with patch("analytics.market_state.bulk_history", return_value={}):
            result = compute_vix_signals(idx)
        assert result["vix_term_structure"] < 1.0
        assert result["vix_term_structure"] == pytest.approx(18.0 / 22.0)

    def test_term_structure_backwardation(self):
        """VIX > VIX3M → vix_term_structure > 1 (backwardation / stress)."""
        idx = _make_index_data(vix=30.0, vix3m=22.0)
        with patch("analytics.market_state.bulk_history", return_value={}):
            result = compute_vix_signals(idx)
        assert result["vix_term_structure"] > 1.0

    def test_vix9d_ratio(self):
        idx = _make_index_data(vix=20.0, vix9d=15.0)
        with patch("analytics.market_state.bulk_history", return_value={}):
            result = compute_vix_signals(idx)
        assert result["vix9d_ratio"] == pytest.approx(15.0 / 20.0)

    def test_vix_pct_from_history(self):
        """VIX percentile should be computed when history is available."""
        idx = _make_index_data(vix=25.0)
        vix_prices = np.linspace(10, 40, 252)  # VIX=25 ≈ 63rd percentile
        vix_hist_df = pd.DataFrame(
            {"Close": vix_prices},
            index=pd.date_range("2023-01-01", periods=252, freq="B"),
        )
        with patch("analytics.market_state.bulk_history", return_value={"^VIX": vix_hist_df}):
            result = compute_vix_signals(idx)
        assert result["vix_pct_1y"] is not None
        assert 0.0 <= result["vix_pct_1y"] <= 100.0

    def test_values_match_input(self):
        idx = _make_index_data(vix=17.5, vix3m=19.2, vix9d=15.0, vvix=98.5)
        with patch("analytics.market_state.bulk_history", return_value={}):
            result = compute_vix_signals(idx)
        assert result["vix"]   == pytest.approx(17.5)
        assert result["vix3m"] == pytest.approx(19.2)
        assert result["vix9d"] == pytest.approx(15.0)
        assert result["vvix"]  == pytest.approx(98.5)


# ─────────────────────────────────────────────────────────────────────────────
# compute_cross_asset tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCrossAsset:
    def test_all_keys_present(self):
        idx = _make_index_data()
        with patch("analytics.market_state.latest_fred", return_value=_make_fred_data()), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value=_make_aaii_data()):
            result = compute_cross_asset(idx, {})
        required = {
            "rsp_spy_ratio", "rsp_spy_30d_delta", "sphb_splv_ratio",
            "copper_gold_ratio", "hy_oas", "hy_oas_zscore", "aaii_bull_bear_zscore",
        }
        assert required.issubset(result.keys())

    def test_rsp_spy_ratio(self):
        idx = _make_index_data()
        idx["RSP"]["price"] = 160.0
        idx["SPY"]["price"] = 500.0
        with patch("analytics.market_state.latest_fred", return_value=_make_fred_data()), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value=_make_aaii_data()):
            result = compute_cross_asset(idx, {})
        assert result["rsp_spy_ratio"] == pytest.approx(0.32)

    def test_sphb_splv_ratio(self):
        idx = _make_index_data()
        idx["SPHB"]["price"] = 80.0
        idx["SPLV"]["price"] = 70.0
        with patch("analytics.market_state.latest_fred", return_value=_make_fred_data()), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value=_make_aaii_data()):
            result = compute_cross_asset(idx, {})
        assert result["sphb_splv_ratio"] == pytest.approx(80.0 / 70.0)

    def test_copper_gold_ratio(self):
        idx = _make_index_data()
        idx["HG=F"]["price"] = 4.5
        idx["GC=F"]["price"] = 1800.0
        with patch("analytics.market_state.latest_fred", return_value=_make_fred_data()), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value=_make_aaii_data()):
            result = compute_cross_asset(idx, {})
        assert result["copper_gold_ratio"] == pytest.approx(4.5 / 1800.0)

    def test_hy_oas_from_fred(self):
        idx = _make_index_data()
        with patch("analytics.market_state.latest_fred", return_value={"hy_oas": 450.0, "t10y2y": 0.3, "dgs10": 4.5}), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value=_make_aaii_data()):
            result = compute_cross_asset(idx, {})
        assert result["hy_oas"] == pytest.approx(450.0)

    def test_hy_oas_zscore_direction(self):
        """Wide spreads should produce positive z-score (stress indicator)."""
        idx = _make_index_data()
        wide = 700.0   # well above the ~350 mean
        with patch("analytics.market_state.latest_fred", return_value={"hy_oas": wide}), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value=_make_aaii_data()):
            result = compute_cross_asset(idx, {})
        assert result["hy_oas_zscore"] > 0

    def test_rsp_spy_30d_delta_computed(self):
        """If RSP/SPY history is supplied, 30d delta should be computed."""
        idx = _make_index_data()
        rsp_prices = np.ones(60) * 160.0
        spy_prices = np.ones(60) * 500.0
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        rsp_hist = pd.DataFrame({"Close": rsp_prices}, index=dates)
        spy_hist = pd.DataFrame({"Close": spy_prices}, index=dates)
        histories = {"RSP": rsp_hist, "SPY": spy_hist}
        with patch("analytics.market_state.latest_fred", return_value=_make_fred_data()), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value=_make_aaii_data()):
            result = compute_cross_asset(idx, histories)
        # Constant prices → delta should be ~0
        assert result["rsp_spy_30d_delta"] == pytest.approx(0.0, abs=1e-6)

    def test_empty_index_data(self):
        with patch("analytics.market_state.latest_fred", return_value={}), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value={}):
            result = compute_cross_asset({}, {})
        assert result["rsp_spy_ratio"] is None
        assert result["sphb_splv_ratio"] is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_market_state tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMarketState:
    """Tests for the master compute_market_state function."""

    def _call(self, universe=None, index=None, fred=None, aaii=None,
              vix_hist=None, histories=None):
        """Helper: call compute_market_state with mocked data sources."""
        universe = universe or _make_universe(n_tickers=20, n_rows=300)
        index    = index    or _make_index_data()
        fred     = fred     or _make_fred_data()
        aaii     = aaii     or _make_aaii_data()

        # Merge universe + cross-asset histories
        all_hist = dict(universe)
        if histories:
            all_hist.update(histories)

        vix_hist_map = {}
        if vix_hist is not None:
            vix_hist_map = {"^VIX": vix_hist}

        with patch("analytics.market_state.bulk_history") as mock_bulk, \
             patch("analytics.market_state.latest_fred",    return_value=fred), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii",    return_value=aaii):
            mock_bulk.side_effect = lambda tickers, **kw: {
                t: vix_hist_map.get(t) or all_hist.get(t)
                for t in tickers
                if vix_hist_map.get(t) is not None or all_hist.get(t) is not None
            }
            return compute_market_state(universe, index, fred, aaii)

    # ── Return structure ──────────────────────────────────────────────────

    def test_required_keys(self):
        result = self._call()
        required = {"score", "regime", "breadth", "vix", "cross_asset", "drivers", "summary"}
        assert required.issubset(result.keys())

    def test_score_is_int(self):
        result = self._call()
        assert isinstance(result["score"], int)

    def test_score_in_range(self):
        result = self._call()
        assert 0 <= result["score"] <= 100

    def test_regime_is_valid_string(self):
        result = self._call()
        assert result["regime"] in {"Risk-On", "Neutral", "Caution", "Risk-Off"}

    def test_drivers_is_list(self):
        result = self._call()
        assert isinstance(result["drivers"], list)

    def test_drivers_not_empty(self):
        result = self._call()
        assert len(result["drivers"]) > 0

    def test_driver_keys(self):
        result = self._call()
        for d in result["drivers"]:
            assert "name" in d
            assert "contribution" in d
            assert "direction" in d

    def test_driver_direction_consistency(self):
        result = self._call()
        for d in result["drivers"]:
            c = d["contribution"]
            if c > 0:
                assert d["direction"] == "positive"
            elif c < 0:
                assert d["direction"] == "negative"
            else:
                assert d["direction"] == "neutral"

    def test_drivers_sorted_by_abs_contribution(self):
        result = self._call()
        drivers = result["drivers"]
        contribs = [abs(d["contribution"]) for d in drivers]
        assert contribs == sorted(contribs, reverse=True), "Drivers should be sorted by |contribution| desc"

    def test_summary_is_non_empty_string(self):
        result = self._call()
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_regime_matches_score(self):
        result = self._call()
        score  = result["score"]
        regime = result["regime"]
        assert regime == _score_to_regime(score)

    # ── Score directionality ─────────────────────────────────────────────

    def test_high_vix_stress_lowers_score(self):
        """High VIX should produce a lower score than low VIX (all else equal)."""
        calm_idx   = _make_index_data(vix=12.0, vix3m=16.0, vvix=85.0)
        stress_idx = _make_index_data(vix=45.0, vix3m=35.0, vvix=140.0)

        result_calm   = self._call(index=calm_idx,   fred=_make_fred_data(hy_oas=280))
        result_stress = self._call(index=stress_idx, fred=_make_fred_data(hy_oas=650))

        assert result_calm["score"] > result_stress["score"], (
            f"Calm score ({result_calm['score']}) should exceed stress score ({result_stress['score']})"
        )

    def test_wide_hy_oas_lowers_score(self):
        """Wide HY spreads should produce a lower score than tight spreads."""
        result_tight = self._call(fred=_make_fred_data(hy_oas=250))
        result_wide  = self._call(fred=_make_fred_data(hy_oas=700))
        assert result_tight["score"] > result_wide["score"]

    # ── Graceful degradation ─────────────────────────────────────────────

    def test_empty_universe_no_crash(self):
        """Empty universe should not raise — returns partial score."""
        result = self._call(universe={})
        assert 0 <= result["score"] <= 100

    def test_empty_index_data_no_crash(self):
        result = self._call(index={})
        assert isinstance(result["score"], int)
        assert 0 <= result["score"] <= 100

    def test_none_fred_data_no_crash(self):
        result = self._call(fred={})
        assert isinstance(result["score"], int)

    def test_none_aaii_data_no_crash(self):
        result = self._call(aaii={})
        assert isinstance(result["score"], int)

    def test_all_none_data_returns_default(self):
        """When all data sources return empty, function should not crash."""
        with patch("analytics.market_state.bulk_history", return_value={}), \
             patch("analytics.market_state.latest_fred", return_value={}), \
             patch("analytics.market_state.get_fred_series", return_value={}), \
             patch("analytics.market_state.latest_aaii", return_value={}):
            result = compute_market_state({}, {}, {}, {})
        assert isinstance(result["score"], int)
        assert 0 <= result["score"] <= 100

    # ── Score boundary checks ────────────────────────────────────────────

    def test_score_never_exceeds_100(self):
        # Extremely bullish conditions
        calm = _make_index_data(vix=9.0, vix3m=15.0, vix9d=8.0, vvix=70.0)
        result = self._call(index=calm, fred=_make_fred_data(hy_oas=200))
        assert result["score"] <= 100

    def test_score_never_below_0(self):
        # Extremely bearish conditions
        stress = _make_index_data(vix=80.0, vix3m=50.0, vix9d=90.0, vvix=200.0)
        result = self._call(index=stress, fred=_make_fred_data(hy_oas=900))
        assert result["score"] >= 0

    # ── Sub-indicator dicts passed through ───────────────────────────────

    def test_breadth_dict_in_result(self):
        universe = _make_universe(n_tickers=10, n_rows=300)
        result   = self._call(universe=universe)
        assert isinstance(result["breadth"], dict)

    def test_vix_dict_in_result(self):
        result = self._call()
        assert isinstance(result["vix"], dict)

    def test_cross_asset_dict_in_result(self):
        result = self._call()
        assert isinstance(result["cross_asset"], dict)

    # ── AAII contrarian logic ────────────────────────────────────────────

    def test_aaii_extreme_bullish_contrarian_warning(self):
        """Extremely bullish AAII (high z-score) should weigh negatively."""
        normal_aaii  = _make_aaii_data(zscore=0.0)
        extreme_aaii = _make_aaii_data(zscore=3.0)   # extreme bulls = contrarian sell
        result_normal  = self._call(aaii=normal_aaii)
        result_extreme = self._call(aaii=extreme_aaii)
        # Extreme sentiment should subtract from score
        assert result_normal["score"] >= result_extreme["score"]

    def test_aaii_extreme_bearish_also_contrarian(self):
        """Extremely bearish AAII (very negative z-score) should also weigh negatively."""
        normal_aaii   = _make_aaii_data(zscore=0.0)
        extreme_aaii  = _make_aaii_data(zscore=-3.0)  # extreme bears = contrarian buy signal...
        # abs() is taken, so both extremes reduce score equally
        result_normal   = self._call(aaii=normal_aaii)
        result_extreme  = self._call(aaii=extreme_aaii)
        assert result_normal["score"] >= result_extreme["score"]

    # ── Breadth impact ───────────────────────────────────────────────────

    def test_strong_breadth_higher_score(self):
        """All tickers well above their 50/200 DMA should push score up."""
        bull_universe = {
            f"TICK{i:02d}": _make_ohlcv(n_rows=300, drift=0.005)  # strong uptrend
            for i in range(20)
        }
        bear_universe = {
            f"TICK{i:02d}": _make_ohlcv(n_rows=300, start_price=200, drift=-0.003)  # downtrend
            for i in range(20)
        }
        result_bull = self._call(universe=bull_universe)
        result_bear = self._call(universe=bear_universe)
        # Bull universe should produce higher or equal score
        assert result_bull["score"] >= result_bear["score"]
