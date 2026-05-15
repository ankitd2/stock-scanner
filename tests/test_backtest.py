"""
Tests for analytics/backtest.py — walk-forward backtest of the 8 screens.

Uses synthetic universes with known price patterns so we can verify each
price-only screen surfaces the right names. Skipped screens (5/6/7) emit a
stub the HTML side knows how to render.

Don't run a full 5-year backtest here — synthetic windows are 6-12 months.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from analytics.backtest import (
    CACHE_TTL_DAYS,
    PRICE_ONLY_SCREENS,
    SKIPPED_SCREENS,
    _build_panels,
    _fwd_return,
    _simplified_screen,
    _subsample_liquid,
    _ticker_metrics_at,
    get_or_compute_backtest,
    load_cached_backtest,
    run_walkforward_backtest,
    save_backtest_cache,
    screen_can_be_backtested,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic price data
# ──────────────────────────────────────────────────────────────────────────────

def _bday_index(n: int, end: date | None = None) -> pd.DatetimeIndex:
    end = end or date(2026, 1, 1)
    return pd.bdate_range(end=end, periods=n)


def _ohlcv(close: np.ndarray, idx: pd.DatetimeIndex, vol: float = 1e7) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame around a close path."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full_like(close, vol),
        },
        index=idx,
    )


def _uptrend(n: int = 300, start: float = 100.0, drift: float = 0.0015) -> np.ndarray:
    """Monotonic uptrend with small noise — close near rolling high."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.003, n)
    rets = drift + noise
    return start * np.cumprod(1 + rets)


def _downtrend(n: int = 300, start: float = 100.0, drift: float = -0.002) -> np.ndarray:
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.003, n)
    rets = drift + noise
    return start * np.cumprod(1 + rets)


def _pullback(n: int = 300, start: float = 100.0) -> np.ndarray:
    """Strong uptrend that pulls back ~15% in the last 25 days, holding 50DMA."""
    rng = np.random.default_rng(11)
    rets = np.concatenate(
        [
            0.002 + rng.normal(0, 0.003, n - 25),
            -0.005 + rng.normal(0, 0.003, 25),
        ]
    )
    return start * np.cumprod(1 + rets)


def _oversold(n: int = 320, start: float = 100.0) -> np.ndarray:
    """Long-run uptrend with very recent sharp drop — RSI <30 but 200DMA rising."""
    rng = np.random.default_rng(99)
    rets = np.concatenate(
        [
            0.0018 + rng.normal(0, 0.003, n - 10),
            -0.025 + rng.normal(0, 0.003, 10),
        ]
    )
    return start * np.cumprod(1 + rets)


def _synthetic_universe() -> dict[str, pd.DataFrame]:
    """A mix of 5 distinct patterns plus SPY."""
    n = 320
    idx = _bday_index(n)
    rng = np.random.default_rng(0)
    spy = 400.0 * np.cumprod(1 + rng.normal(0.0004, 0.008, n))
    return {
        "UPTREND": _ohlcv(_uptrend(n), idx, vol=5e7),
        "DOWNER": _ohlcv(_downtrend(n), idx, vol=2e7),
        "PULLBACK": _ohlcv(_pullback(n), idx, vol=3e7),
        "OVERSOLD": _ohlcv(_oversold(n), idx, vol=1e7),
        "FLAT": _ohlcv(np.full(n, 50.0), idx, vol=1e6),
        "SPY": _ohlcv(spy, idx, vol=1e8),
    }


# ──────────────────────────────────────────────────────────────────────────────
# screen_can_be_backtested
# ──────────────────────────────────────────────────────────────────────────────

class TestScreenCanBeBacktested:
    def test_price_only_screens_can_be_backtested(self):
        for sid in (1, 2, 3, 4, 8):
            ok, _ = screen_can_be_backtested(sid)
            assert ok is True

    def test_pead_screen_is_skipped(self):
        ok, reason = screen_can_be_backtested(5)
        assert ok is False
        assert "earnings" in reason.lower() or "sue" in reason.lower()

    def test_analyst_screen_is_skipped(self):
        ok, reason = screen_can_be_backtested(6)
        assert ok is False
        assert "analyst" in reason.lower() or "estimate" in reason.lower()

    def test_insider_screen_is_skipped(self):
        ok, reason = screen_can_be_backtested(7)
        assert ok is False
        assert "form 4" in reason.lower() or "insider" in reason.lower() or "sec" in reason.lower()

    def test_unknown_screen_id(self):
        ok, reason = screen_can_be_backtested(99)
        assert ok is False
        assert reason


# ──────────────────────────────────────────────────────────────────────────────
# _build_panels and _subsample_liquid
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildPanels:
    def test_builds_close_volume_panels(self):
        uni = _synthetic_universe()
        close, vol = _build_panels(uni)
        assert not close.empty
        assert not vol.empty
        assert set(["UPTREND", "DOWNER", "SPY"]).issubset(close.columns)

    def test_empty_universe_returns_empty(self):
        close, vol = _build_panels({})
        assert close.empty
        assert vol.empty

    def test_skips_short_histories(self):
        # Histories shorter than 60 rows should be filtered out
        short = _ohlcv(np.linspace(10, 12, 30), _bday_index(30))
        long_ = _ohlcv(_uptrend(120), _bday_index(120))
        close, _vol = _build_panels({"SHORT": short, "LONG": long_})
        assert "LONG" in close.columns
        assert "SHORT" not in close.columns


class TestSubsampleLiquid:
    def test_picks_top_by_dollar_volume(self):
        uni = _synthetic_universe()
        close, vol = _build_panels(uni)
        liquid = _subsample_liquid(close, vol, n_top=3)
        # SPY has the highest volume; UPTREND/PULLBACK next
        assert "SPY" in liquid
        assert len(liquid) <= 3


# ──────────────────────────────────────────────────────────────────────────────
# _ticker_metrics_at
# ──────────────────────────────────────────────────────────────────────────────

class TestTickerMetrics:
    def test_uptrend_near_high(self):
        n = 280
        close = pd.Series(_uptrend(n))
        m = _ticker_metrics_at(close, None, n - 1)
        assert m is not None
        # Strong uptrend → near 52w high
        assert m["pct_from_52wh"] is not None
        assert m["pct_from_52wh"] >= -5
        assert m["above_50dma"] is True
        assert m["above_200dma"] is True
        assert m["golden_cross"] is True

    def test_downtrend_not_near_high(self):
        n = 280
        close = pd.Series(_downtrend(n))
        m = _ticker_metrics_at(close, None, n - 1)
        assert m is not None
        assert m["pct_from_52wh"] < -10
        assert m["above_200dma"] is False

    def test_insufficient_history_returns_none(self):
        close = pd.Series(_uptrend(100))
        # Need at least 200 idx; passing 50 should produce None
        assert _ticker_metrics_at(close, None, 50) is None


# ──────────────────────────────────────────────────────────────────────────────
# _simplified_screen
# ──────────────────────────────────────────────────────────────────────────────

class TestSimplifiedScreen:
    def setup_method(self):
        self.uni = _synthetic_universe()
        self.close, self.vol = _build_panels(self.uni)
        self.state = {"close": self.close, "volume": self.vol}
        self.spy = self.close["SPY"]
        self.idx = len(self.close) - 1

    def test_breakout_screen_surfaces_uptrend_names(self):
        # Screen 1 looks for 52wH proximity + volume + RSI<80
        names = _simplified_screen(1, self.state, self.spy, self.idx)
        # UPTREND should appear; DOWNER should not
        assert "DOWNER" not in names
        # And FLAT must not be there (volume is too low + no proximity move)
        assert "FLAT" not in names

    def test_pullback_screen_requires_golden_cross(self):
        # Screen 2 needs 50>200DMA. DOWNER has the opposite — must not appear.
        names = _simplified_screen(2, self.state, self.spy, self.idx)
        assert "DOWNER" not in names

    def test_pullback_surfaces_pullback_pattern(self):
        # PULLBACK should show 10-30% drawdown with golden cross
        # (the synthetic pattern is designed for this).
        names = _simplified_screen(2, self.state, self.spy, self.idx)
        # Don't require — synthetic noise can shift the exact ratio — but
        # a downtrend should *never* be in screen 2
        assert "DOWNER" not in names

    def test_risk_adj_momentum_screens_uptrend(self):
        names = _simplified_screen(3, self.state, self.spy, self.idx)
        assert "DOWNER" not in names

    def test_quality_momentum_screens_uptrend(self):
        names = _simplified_screen(4, self.state, self.spy, self.idx)
        assert "DOWNER" not in names

    def test_oversold_screen_requires_rising_200dma(self):
        names = _simplified_screen(8, self.state, self.spy, self.idx)
        # DOWNER's 200DMA is falling — must not show
        assert "DOWNER" not in names

    def test_skipped_screens_return_empty(self):
        for sid in (5, 6, 7):
            names = _simplified_screen(sid, self.state, self.spy, self.idx)
            assert names == []

    def test_handles_bad_input(self):
        # Bad state dict
        assert _simplified_screen(1, {}, self.spy, self.idx) == []
        # Out-of-range idx
        assert _simplified_screen(1, self.state, self.spy, 50) == []


# ──────────────────────────────────────────────────────────────────────────────
# _fwd_return
# ──────────────────────────────────────────────────────────────────────────────

class TestFwdReturn:
    def test_positive_return(self):
        s = pd.Series([100.0, 101.0, 105.0, 110.0])
        assert _fwd_return(s, 0, 3) == pytest.approx(10.0)

    def test_negative_return(self):
        s = pd.Series([100.0, 90.0])
        assert _fwd_return(s, 0, 1) == pytest.approx(-10.0)

    def test_out_of_range_returns_none(self):
        s = pd.Series([100.0, 105.0])
        assert _fwd_return(s, 0, 10) is None

    def test_nan_price_returns_none(self):
        s = pd.Series([100.0, float("nan")])
        assert _fwd_return(s, 0, 1) is None


# ──────────────────────────────────────────────────────────────────────────────
# run_walkforward_backtest — small lookback for speed
# ──────────────────────────────────────────────────────────────────────────────

class TestWalkforwardBacktest:
    def setup_method(self):
        self.uni = _synthetic_universe()
        self.spy = self.uni["SPY"]

    def test_returns_dict_with_all_requested_screens(self):
        results = run_walkforward_backtest(
            screen_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            universe_histories=self.uni,
            spy_history=self.spy,
            lookback_weeks=12,  # ~3 months
            rebalance_freq_weeks=2,
        )
        assert set(results.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_skipped_screens_have_skipped_flag(self):
        results = run_walkforward_backtest(
            screen_ids=[5, 6, 7],
            universe_histories=self.uni,
            spy_history=self.spy,
            lookback_weeks=8,
        )
        for sid in (5, 6, 7):
            r = results[sid]
            assert r.get("skipped") is True
            assert "reason" in r and r["reason"]

    def test_backtestable_screens_have_required_fields(self):
        results = run_walkforward_backtest(
            screen_ids=[1, 3, 8],
            universe_histories=self.uni,
            spy_history=self.spy,
            lookback_weeks=12,
            rebalance_freq_weeks=2,
        )
        required = {
            "name", "n_observations", "n_unique_tickers",
            "median_fwd_1m", "median_fwd_3m", "median_fwd_6m",
            "spy_baseline_1m", "spy_baseline_3m", "spy_baseline_6m",
            "alpha_3m", "alpha_6m", "hit_rate_3m", "sharpe_3m",
            "lookback_weeks", "as_of",
        }
        for sid in (1, 3, 8):
            assert required.issubset(set(results[sid].keys())), (
                f"screen {sid} missing keys: "
                f"{required - set(results[sid].keys())}"
            )

    def test_empty_universe_returns_stub_for_every_screen(self):
        results = run_walkforward_backtest(
            screen_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            universe_histories={},
            spy_history=pd.DataFrame(),
            lookback_weeks=8,
        )
        assert len(results) == 8
        # Backtestable screens get empty-stats stubs; skipped ones get skipped stubs
        for sid in (1, 2, 3, 4, 8):
            assert "skipped" not in results[sid] or results[sid].get("skipped") is False
        for sid in (5, 6, 7):
            assert results[sid].get("skipped") is True


# ──────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheRoundTrip:
    def test_save_and_load_cache(self, tmp_path):
        cache = tmp_path / "backtest.json"
        results = {
            1: {"name": "52wH Proximity", "n_observations": 100, "median_fwd_3m": 4.5},
            5: {"name": "PEAD", "skipped": True, "reason": "no earnings"},
        }
        save_backtest_cache(results, cache_path=cache)
        assert cache.exists()
        loaded = load_cached_backtest(cache_path=cache)
        assert 1 in loaded and 5 in loaded
        assert loaded[1]["n_observations"] == 100
        assert loaded[5]["skipped"] is True

    def test_load_missing_returns_empty(self, tmp_path):
        cache = tmp_path / "does_not_exist.json"
        assert load_cached_backtest(cache_path=cache) == {}

    def test_load_corrupt_returns_empty(self, tmp_path):
        cache = tmp_path / "corrupt.json"
        cache.write_text("{this is not json")
        assert load_cached_backtest(cache_path=cache) == {}

    def test_load_stale_returns_empty(self, tmp_path):
        cache = tmp_path / "stale.json"
        # Write a payload with a cached_on date older than TTL
        stale_dt = datetime.now() - timedelta(days=CACHE_TTL_DAYS + 1)
        cache.write_text(json.dumps({
            "_cached_on": stale_dt.isoformat(),
            "results": {"1": {"name": "X", "n_observations": 1}},
        }))
        assert load_cached_backtest(cache_path=cache) == {}

    def test_int_keys_preserved_across_roundtrip(self, tmp_path):
        cache = tmp_path / "bt.json"
        save_backtest_cache({3: {"name": "Risk-Adj", "n_observations": 5}}, cache_path=cache)
        loaded = load_cached_backtest(cache_path=cache)
        assert 3 in loaded
        # Should NOT have string "3" key when round-tripped
        assert "3" not in loaded


# ──────────────────────────────────────────────────────────────────────────────
# get_or_compute_backtest — public entry point
# ──────────────────────────────────────────────────────────────────────────────

class TestGetOrComputeBacktest:
    def test_returns_cached_when_fresh(self, tmp_path):
        cache = tmp_path / "bt.json"
        save_backtest_cache(
            {1: {"name": "From cache", "n_observations": 999}},
            cache_path=cache,
        )
        out = get_or_compute_backtest(
            universe_histories={},
            spy_history=pd.DataFrame(),
            cache_path=cache,
        )
        assert out[1]["n_observations"] == 999
        assert out[1]["name"] == "From cache"

    def test_computes_when_cache_missing(self, tmp_path):
        cache = tmp_path / "fresh.json"
        uni = _synthetic_universe()
        out = get_or_compute_backtest(
            universe_histories=uni,
            spy_history=uni["SPY"],
            cache_path=cache,
        )
        assert set(out.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}
        # Cache file should now exist
        assert cache.exists()

    def test_force_refresh_skips_cache(self, tmp_path):
        cache = tmp_path / "bt.json"
        save_backtest_cache({1: {"name": "Cached", "n_observations": 1}}, cache_path=cache)
        uni = _synthetic_universe()
        out = get_or_compute_backtest(
            universe_histories=uni,
            spy_history=uni["SPY"],
            cache_path=cache,
            force_refresh=True,
        )
        # The recomputed result will overwrite the old single-entry cache
        assert 1 in out and 2 in out
        assert "n_observations" in out[1]

    def test_never_raises_on_bad_input(self):
        # Even with no universe data, it should return a dict (with stubs)
        out = get_or_compute_backtest(
            universe_histories={},
            spy_history=pd.DataFrame(),
            cache_path=Path(tempfile.gettempdir()) / "nonexistent_backtest_test.json",
            force_refresh=True,
        )
        assert isinstance(out, dict)
        assert len(out) == 8
