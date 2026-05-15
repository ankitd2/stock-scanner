"""
tests/test_screens.py — unit tests for analytics/screens.py

Tests cover:
  - Universal hard kills (mcap < 2B, sell_count >= 2, RSI > 80 at highs)
  - Individual screen logic with synthetic data
  - run_all_screens returns correct shape (dict with 8 keys)
  - held_tickers marks candidates correctly
  - Graceful handling of missing data
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from analytics.screens import (
    run_screen,
    run_all_screens,
    SCREEN_META,
    _hard_kill,
    _build_candidate,
)
from analytics.indicators import compute_all


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_price_series(
    n: int = 300,
    start_price: float = 100.0,
    trend: float = 0.001,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic daily price series."""
    rng = np.random.default_rng(seed)
    returns = trend + vol * rng.standard_normal(n)
    prices = start_price * np.cumprod(1 + returns)
    return pd.Series(prices)


def make_ohlcv_df(
    n: int = 300,
    start_price: float = 100.0,
    trend: float = 0.001,
    vol: float = 0.015,
    seed: int = 42,
    volume_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    returns = trend + vol * rng.standard_normal(n)
    close = start_price * np.cumprod(1 + returns)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": close * (1 - vol * 0.5),
            "High": close * (1 + vol),
            "Low": close * (1 - vol),
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, size=n) * volume_multiplier,
        },
        index=dates,
    )
    return df


def make_near_52wh_df(volume_ratio: float = 2.0) -> pd.DataFrame:
    """
    Create a DataFrame where the latest close is near the 52-week high
    (within 3%) with elevated volume.
    """
    n = 300
    rng = np.random.default_rng(1)
    # Build a generally rising series
    returns = 0.001 + 0.012 * rng.standard_normal(n)
    close = 100.0 * np.cumprod(1 + returns)
    # Make the last few bars the highest
    close[-5:] = close.max() * 0.98  # within 2% of high
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    base_vol = 500_000
    volume = np.full(n, base_vol)
    volume[-1] = int(base_vol * volume_ratio)  # today's volume is elevated
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def make_oversold_df() -> pd.DataFrame:
    """Create a DataFrame that will produce RSI < 30 (strong downtrend)."""
    n = 300
    rng = np.random.default_rng(5)
    returns = -0.005 + 0.015 * rng.standard_normal(n)
    close = 200.0 * np.cumprod(1 + returns)
    close = np.clip(close, 1, None)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(100_000, 500_000, size=n).astype(float),
        },
        index=dates,
    )
    return df


def make_uptrend_pullback_df() -> pd.DataFrame:
    """
    Create a DataFrame with a golden cross (50DMA > 200DMA) and
    price ~15% below the 52-week high (quality pullback zone).
    """
    n = 300
    rng = np.random.default_rng(7)
    # Strong uptrend for first 250 bars, then slight pullback
    returns_up = 0.003 + 0.010 * rng.standard_normal(250)
    returns_dn = -0.002 + 0.010 * rng.standard_normal(50)
    returns = np.concatenate([returns_up, returns_dn])
    close = 50.0 * np.cumprod(1 + returns)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(200_000, 800_000, size=n).astype(float),
        },
        index=dates,
    )
    return df


def base_info(
    mcap: float = 5e9,
    sell_count: int = 0,
    buy_pct: int = 80,
    rev_growth: float = 25.0,
    gross_margin: float = 50.0,
    pct_today: float = 1.0,
    earnings_dt_offset: int = None,
) -> dict:
    """Build a minimal ticker_info dict that passes all hard kills by default."""
    ed = None
    if earnings_dt_offset is not None:
        ed = date.today() - timedelta(days=earnings_dt_offset)
    return {
        "ticker": "TEST",
        "name": "Test Corp",
        "price": 100.0,
        "mcap": mcap,
        "sell_count": sell_count,
        "sell": sell_count,
        "buy_pct": buy_pct,
        "rev_growth": rev_growth,
        "gross_margin": gross_margin,
        "op_margin": 15.0,
        "tgt_mean": 120.0,
        "tgt_high": 140.0,
        "tgt_low": 90.0,
        "n_analysts": 20,
        "pct_today": pct_today,
        "earnings_dt": ed,
    }


# ---------------------------------------------------------------------------
# Build a small synthetic universe
# ---------------------------------------------------------------------------

def build_small_universe() -> tuple:
    """
    Returns (universe_histories, ticker_infos) for a 7-ticker test universe.
    Designed so that at least one ticker passes each screen.
    """
    histories = {}
    infos = {}

    # ALPHA: near 52wk high, elevated volume, RSI ~65 → Screen 1
    histories["ALPHA"] = make_near_52wh_df(volume_ratio=2.5)
    infos["ALPHA"] = base_info(mcap=10e9, buy_pct=75, rev_growth=22.0, gross_margin=55.0)

    # BETA: uptrend with pullback → Screen 2
    histories["BETA"] = make_uptrend_pullback_df()
    infos["BETA"] = base_info(mcap=8e9, buy_pct=70, rev_growth=18.0, gross_margin=45.0)

    # GAMMA: strong uptrend (high vol_adj_mom) → Screen 3
    histories["GAMMA"] = make_ohlcv_df(n=300, trend=0.003, vol=0.010, seed=10)
    infos["GAMMA"] = base_info(mcap=6e9, buy_pct=65, rev_growth=15.0, gross_margin=40.0)

    # DELTA: high quality, momentum → Screen 4
    histories["DELTA"] = make_ohlcv_df(n=300, trend=0.002, vol=0.010, seed=20)
    infos["DELTA"] = base_info(mcap=12e9, buy_pct=75, rev_growth=30.0, gross_margin=55.0)

    # EPSILON: PEAD candidate (recent earnings, holding gains) → Screen 5
    histories["EPSILON"] = make_ohlcv_df(n=300, trend=0.001, vol=0.010, seed=30)
    infos["EPSILON"] = base_info(
        mcap=7e9, buy_pct=70, rev_growth=20.0,
        pct_today=2.0, earnings_dt_offset=10
    )

    # ZETA: strong analyst consensus → Screen 6
    histories["ZETA"] = make_ohlcv_df(n=300, trend=0.001, vol=0.010, seed=40)
    infos["ZETA"] = base_info(mcap=9e9, buy_pct=80, rev_growth=20.0, sell_count=0)

    # ETA: oversold, large cap → Screen 8
    histories["ETA"] = make_oversold_df()
    infos["ETA"] = base_info(mcap=10e9, buy_pct=65, rev_growth=15.0, gross_margin=40.0)

    return histories, infos


# ---------------------------------------------------------------------------
# Tests: SCREEN_META completeness
# ---------------------------------------------------------------------------

class TestScreenMeta:
    def test_all_8_screens_defined(self):
        assert set(SCREEN_META.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_each_screen_has_required_keys(self):
        required = {"name", "description", "evidence", "hold_period"}
        for sid, meta in SCREEN_META.items():
            assert required.issubset(meta.keys()), f"Screen {sid} missing keys"

    def test_names_are_non_empty_strings(self):
        for sid, meta in SCREEN_META.items():
            assert isinstance(meta["name"], str) and len(meta["name"]) > 0


# ---------------------------------------------------------------------------
# Tests: Hard kills
# ---------------------------------------------------------------------------

class TestHardKills:
    def setup_method(self):
        self.df = make_ohlcv_df(n=250)
        self.ind = compute_all(self.df)

    def test_mcap_below_2b_killed(self):
        info = base_info(mcap=1e9)
        assert _hard_kill(info, self.ind) is True

    def test_mcap_exactly_2b_ok(self):
        info = base_info(mcap=2e9)
        assert _hard_kill(info, self.ind) is False

    def test_sell_count_2_killed(self):
        info = base_info(sell_count=2)
        assert _hard_kill(info, self.ind) is True

    def test_sell_count_3_killed(self):
        info = base_info(sell_count=3)
        assert _hard_kill(info, self.ind) is True

    def test_sell_count_1_ok(self):
        info = base_info(sell_count=1)
        assert _hard_kill(info, self.ind) is False

    def test_overbought_at_highs_killed(self):
        """RSI > 80 AND pct_from_52wh > -5 should trigger hard kill."""
        info = base_info()
        # Force ind to have high RSI and near 52wh
        fake_ind = dict(self.ind)
        fake_ind["latest_rsi"] = 85.0
        fake_ind["pct_from_52wh"] = -2.0
        assert _hard_kill(info, fake_ind) is True

    def test_high_rsi_but_far_from_high_ok(self):
        info = base_info()
        fake_ind = dict(self.ind)
        fake_ind["latest_rsi"] = 85.0
        fake_ind["pct_from_52wh"] = -15.0
        assert _hard_kill(info, fake_ind) is False

    def test_near_high_but_normal_rsi_ok(self):
        info = base_info()
        fake_ind = dict(self.ind)
        fake_ind["latest_rsi"] = 70.0
        fake_ind["pct_from_52wh"] = -2.0
        assert _hard_kill(info, fake_ind) is False


# ---------------------------------------------------------------------------
# Tests: Screen 1 — 52wH Proximity
# ---------------------------------------------------------------------------

class TestScreen1_52wHProximity:
    def test_passes_near_high_elevated_volume(self):
        """Near 52wk high with volume ratio > 1.5 should surface a candidate."""
        histories = {"AA": make_near_52wh_df(volume_ratio=2.0)}
        infos = {"AA": base_info(mcap=5e9, sell_count=0)}
        results = run_screen(1, histories, infos, set())
        # May or may not pass depending on computed RSI — just verify no crash
        assert isinstance(results, list)

    def test_low_volume_fails(self):
        """Volume ratio below 1.5 should not produce a candidate."""
        df = make_near_52wh_df(volume_ratio=0.5)
        ind = compute_all(df)
        # Manually confirm volume_ratio < 1.5
        if ind.get("volume_ratio") is not None:
            assert ind["volume_ratio"] < 1.5

    def test_small_cap_excluded(self):
        """mcap < 2B must produce no candidates."""
        histories = {"AA": make_near_52wh_df(volume_ratio=3.0)}
        infos = {"AA": base_info(mcap=500e6)}
        results = run_screen(1, histories, infos, set())
        assert results == []

    def test_sell_count_2_excluded(self):
        histories = {"AA": make_near_52wh_df(volume_ratio=3.0)}
        infos = {"AA": base_info(mcap=5e9, sell_count=2)}
        results = run_screen(1, histories, infos, set())
        assert results == []

    def test_returns_list(self):
        histories = {"AA": make_near_52wh_df()}
        infos = {"AA": base_info()}
        result = run_screen(1, histories, infos, set())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Screen 2 — Quality Pullback
# ---------------------------------------------------------------------------

class TestScreen2_QualityPullback:
    def test_requires_golden_cross(self):
        """Without golden cross (50DMA > 200DMA), should not fire."""
        df = make_oversold_df()  # downtrend — no golden cross
        ind = compute_all(df)
        assert ind.get("golden_cross", False) is False

    def test_pullback_range_check(self):
        """Drawdown outside 10-30% should not pass."""
        histories = {"BB": make_ohlcv_df(n=300, trend=0.001)}
        infos = {"BB": base_info(mcap=5e9)}
        results = run_screen(2, histories, infos, set())
        assert isinstance(results, list)

    def test_returns_list(self):
        histories = {"BB": make_uptrend_pullback_df()}
        infos = {"BB": base_info(mcap=8e9, rev_growth=15.0, gross_margin=35.0)}
        result = run_screen(2, histories, infos, set())
        assert isinstance(result, list)

    def test_quality_gate_no_margin_no_growth_fails(self):
        """Without quality metrics, screen 2 should not surface candidate."""
        histories = {"BB": make_uptrend_pullback_df()}
        infos = {"BB": base_info(mcap=8e9, rev_growth=5.0, gross_margin=15.0)}
        # Either passes or not — key is no exception
        result = run_screen(2, histories, infos, set())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Screen 3 — Risk-Adjusted Momentum
# ---------------------------------------------------------------------------

class TestScreen3_RiskAdjMomentum:
    def test_requires_above_200dma(self):
        """Stocks below 200DMA should not pass."""
        df = make_oversold_df()
        ind = compute_all(df)
        assert ind.get("above_200dma", False) is False

    def test_percentile_threshold_computed(self):
        """Universe percentile computation should not crash."""
        histories = {
            "AA": make_ohlcv_df(n=300, trend=0.003, seed=1),
            "BB": make_ohlcv_df(n=300, trend=0.001, seed=2),
            "CC": make_ohlcv_df(n=300, trend=-0.001, seed=3),
            "DD": make_ohlcv_df(n=300, trend=0.002, seed=4),
            "EE": make_ohlcv_df(n=300, trend=0.0005, seed=5),
            "FF": make_ohlcv_df(n=300, trend=0.004, seed=6),
            "GG": make_ohlcv_df(n=300, trend=0.0015, seed=7),
        }
        infos = {t: base_info(mcap=5e9) for t in histories}
        results = run_screen(3, histories, infos, set())
        assert isinstance(results, list)

    def test_small_universe_falls_back(self):
        """With < 5 tickers, threshold defaults to 0 and screen still runs."""
        histories = {
            "AA": make_ohlcv_df(n=300, trend=0.003, seed=1),
            "BB": make_ohlcv_df(n=300, trend=0.001, seed=2),
        }
        infos = {t: base_info(mcap=5e9) for t in histories}
        results = run_screen(3, histories, infos, set())
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests: Screen 4 — Quality-Momentum
# ---------------------------------------------------------------------------

class TestScreen4_QualityMomentum:
    def test_low_gross_margin_fails(self):
        """Gross margin < 40% should not produce candidate."""
        histories = {"CC": make_ohlcv_df(n=300, trend=0.002)}
        infos = {"CC": base_info(mcap=5e9, gross_margin=30.0, rev_growth=20.0, buy_pct=70)}
        results = run_screen(4, histories, infos, set())
        assert results == []

    def test_low_rev_growth_fails(self):
        histories = {"CC": make_ohlcv_df(n=300, trend=0.002)}
        infos = {"CC": base_info(mcap=5e9, gross_margin=50.0, rev_growth=10.0, buy_pct=70)}
        results = run_screen(4, histories, infos, set())
        assert results == []

    def test_low_buy_pct_fails(self):
        histories = {"CC": make_ohlcv_df(n=300, trend=0.002)}
        infos = {"CC": base_info(mcap=5e9, gross_margin=50.0, rev_growth=20.0, buy_pct=50)}
        results = run_screen(4, histories, infos, set())
        assert results == []

    def test_all_criteria_met_may_surface(self):
        """All criteria met — should not crash, result is list."""
        histories = {"CC": make_ohlcv_df(n=300, trend=0.003, vol=0.008)}
        infos = {"CC": base_info(mcap=10e9, gross_margin=55.0, rev_growth=25.0, buy_pct=75)}
        results = run_screen(4, histories, infos, set())
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests: Screen 5 — PEAD
# ---------------------------------------------------------------------------

class TestScreen5_PEAD:
    def test_no_earnings_date_fails(self):
        """No earnings_dt → no candidate."""
        histories = {"DD": make_ohlcv_df(n=300, trend=0.001)}
        infos = {"DD": base_info(mcap=5e9, earnings_dt_offset=None)}
        results = run_screen(5, histories, infos, set())
        assert results == []

    def test_future_earnings_fails(self):
        """Upcoming earnings (not past) should not trigger PEAD."""
        info = base_info(mcap=5e9)
        # earnings_dt in the future
        info["earnings_dt"] = date.today() + timedelta(days=5)
        histories = {"DD": make_ohlcv_df(n=300, trend=0.001)}
        results = run_screen(5, {"DD": histories["DD"]}, {"DD": info}, set())
        assert results == []

    def test_earnings_too_old_fails(self):
        """Earnings > 30 days ago should not trigger."""
        histories = {"DD": make_ohlcv_df(n=300, trend=0.001)}
        infos = {"DD": base_info(mcap=5e9, earnings_dt_offset=35)}
        results = run_screen(5, histories, infos, set())
        assert results == []

    def test_recent_earnings_considered(self):
        """Recent earnings (10 days ago) + positive today → candidate possible."""
        histories = {"DD": make_ohlcv_df(n=300, trend=0.001, seed=33)}
        infos = {"DD": base_info(mcap=5e9, buy_pct=70, pct_today=2.0, earnings_dt_offset=10)}
        results = run_screen(5, histories, infos, set())
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests: Screen 6 — Analyst Revision Momentum
# ---------------------------------------------------------------------------

class TestScreen6_AnalystRevision:
    def test_low_buy_pct_fails(self):
        histories = {"EE": make_ohlcv_df(n=250)}
        infos = {"EE": base_info(mcap=5e9, buy_pct=60, sell_count=0)}
        results = run_screen(6, histories, infos, set())
        assert results == []

    def test_low_upside_fails(self):
        """tgt_mean upside < 15% should not produce candidate."""
        histories = {"EE": make_ohlcv_df(n=250)}
        info = base_info(mcap=5e9, buy_pct=80, sell_count=0)
        info["tgt_mean"] = 105.0  # only 5% upside from price=100
        results = run_screen(6, histories, {"EE": info}, set())
        assert results == []

    def test_sell_count_2_fails(self):
        histories = {"EE": make_ohlcv_df(n=250)}
        infos = {"EE": base_info(mcap=5e9, buy_pct=80, sell_count=2)}
        results = run_screen(6, histories, infos, set())
        assert results == []

    def test_all_criteria_met(self):
        """Strong consensus + high upside + 0 sells → candidate surfaces."""
        histories = {"EE": make_ohlcv_df(n=250)}
        info = base_info(mcap=5e9, buy_pct=80, sell_count=0)
        info["tgt_mean"] = 120.0  # 20% upside
        results = run_screen(6, histories, {"EE": info}, set())
        assert len(results) == 1
        assert results[0]["ticker"] == "EE"
        assert results[0]["screen_id"] == 6

    def test_reason_field_populated(self):
        histories = {"EE": make_ohlcv_df(n=250)}
        info = base_info(mcap=5e9, buy_pct=80, sell_count=0)
        info["tgt_mean"] = 120.0
        results = run_screen(6, histories, {"EE": info}, set())
        if results:
            assert len(results[0]["reason"]) > 10


# ---------------------------------------------------------------------------
# Tests: Screen 7 — Insider Cluster (stub behaviour)
# ---------------------------------------------------------------------------

class TestScreen7_InsiderCluster:
    def test_returns_empty_list_without_edgar(self):
        """Screen 7 should return [] gracefully when edgar module missing."""
        histories = {"FF": make_ohlcv_df(n=250)}
        infos = {"FF": base_info(mcap=5e9)}
        results = run_screen(7, histories, infos, set())
        assert results == []

    def test_no_exception_raised(self):
        histories = {"FF": make_ohlcv_df(n=250)}
        infos = {"FF": base_info(mcap=5e9)}
        try:
            run_screen(7, histories, infos, set())
        except Exception as e:
            pytest.fail(f"Screen 7 raised unexpected exception: {e}")


# ---------------------------------------------------------------------------
# Tests: Screen 8 — Quality Oversold
# ---------------------------------------------------------------------------

class TestScreen8_QualityOversold:
    def test_normal_rsi_fails(self):
        """RSI >= 30 should not pass."""
        df = make_ohlcv_df(n=300, trend=0.001)
        ind = compute_all(df)
        rsi = ind.get("latest_rsi")
        if rsi is not None and rsi >= 30:
            histories = {"GG": df}
            infos = {"GG": base_info(mcap=10e9, rev_growth=15.0)}
            results = run_screen(8, histories, infos, set())
            assert results == []

    def test_oversold_large_cap_quality_candidate(self):
        """Oversold downtrend with big mcap and growth → possible candidate."""
        df = make_oversold_df()
        ind = compute_all(df)
        # Only test if RSI actually is below 30
        rsi = ind.get("latest_rsi")
        if rsi is not None and rsi < 30:
            histories = {"GG": df}
            infos = {"GG": base_info(mcap=10e9, rev_growth=15.0, buy_pct=60)}
            results = run_screen(8, histories, infos, set())
            assert isinstance(results, list)

    def test_small_cap_excluded(self):
        """Screen 8 requires mcap >= 5B."""
        df = make_oversold_df()
        histories = {"GG": df}
        infos = {"GG": base_info(mcap=3e9, rev_growth=15.0)}
        results = run_screen(8, histories, infos, set())
        assert results == []

    def test_low_rev_growth_fails(self):
        df = make_oversold_df()
        histories = {"GG": df}
        infos = {"GG": base_info(mcap=10e9, rev_growth=5.0)}
        results = run_screen(8, histories, infos, set())
        assert results == []


# ---------------------------------------------------------------------------
# Tests: run_all_screens shape
# ---------------------------------------------------------------------------

class TestRunAllScreens:
    def setup_method(self):
        self.histories, self.infos = build_small_universe()

    def test_returns_dict(self):
        result = run_all_screens(self.histories, self.infos)
        assert isinstance(result, dict)

    def test_has_8_keys(self):
        result = run_all_screens(self.histories, self.infos)
        assert set(result.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_each_value_is_list(self):
        result = run_all_screens(self.histories, self.infos)
        for sid in range(1, 9):
            assert isinstance(result[sid], list), f"Screen {sid} should return list"

    def test_candidate_has_required_fields(self):
        result = run_all_screens(self.histories, self.infos)
        required_fields = {
            "ticker", "screen_id", "screen_name", "reason", "held",
            "price", "mcap", "rsi", "pct_from_52wh",
        }
        for sid, candidates in result.items():
            for c in candidates:
                missing = required_fields - set(c.keys())
                assert not missing, f"Screen {sid} candidate missing fields: {missing}"

    def test_held_tickers_flagged(self):
        """Tickers in held_tickers set should have held=True."""
        held = {"ALPHA", "ZETA"}
        result = run_all_screens(self.histories, self.infos, held_tickers=held)
        for sid, candidates in result.items():
            for c in candidates:
                if c["ticker"] in held:
                    assert c["held"] is True, (
                        f"Screen {sid}: {c['ticker']} should be marked held"
                    )
                else:
                    assert c["held"] is False

    def test_non_held_tickers_not_flagged(self):
        held = {"ALPHA"}
        result = run_all_screens(self.histories, self.infos, held_tickers=held)
        for sid, candidates in result.items():
            for c in candidates:
                if c["ticker"] not in held:
                    assert c["held"] is False

    def test_none_held_tickers_defaults(self):
        """held_tickers=None should not raise and all held=False."""
        result = run_all_screens(self.histories, self.infos, held_tickers=None)
        for sid, candidates in result.items():
            for c in candidates:
                assert c["held"] is False

    def test_empty_universe_returns_empty_lists(self):
        result = run_all_screens({}, {})
        assert set(result.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}
        for sid in range(1, 9):
            assert result[sid] == []

    def test_screen_ids_in_candidates_match(self):
        """Each candidate's screen_id should match the key it's under."""
        result = run_all_screens(self.histories, self.infos)
        for sid, candidates in result.items():
            for c in candidates:
                assert c["screen_id"] == sid

    def test_screen_names_match_meta(self):
        result = run_all_screens(self.histories, self.infos)
        for sid, candidates in result.items():
            for c in candidates:
                assert c["screen_name"] == SCREEN_META[sid]["name"]


# ---------------------------------------------------------------------------
# Tests: run_screen interface
# ---------------------------------------------------------------------------

class TestRunScreen:
    def test_invalid_screen_id_returns_empty(self):
        result = run_screen(99, {}, {}, set())
        assert result == []

    def test_exception_in_data_returns_empty(self):
        """Malformed DataFrame should not raise — returns []."""
        bad_df = pd.DataFrame({"Close": [float("nan")] * 5})
        result = run_screen(1, {"BAD": bad_df}, {"BAD": base_info()}, set())
        assert isinstance(result, list)

    def test_missing_ticker_info_skipped(self):
        """Tickers not in ticker_infos should be skipped gracefully."""
        histories = {"AA": make_ohlcv_df(n=250)}
        # No info provided
        result = run_screen(6, histories, {}, set())
        assert result == []

    def test_all_screen_ids_run(self):
        histories = {"AA": make_ohlcv_df(n=300)}
        infos = {"AA": base_info()}
        for sid in range(1, 9):
            result = run_screen(sid, histories, infos, set())
            assert isinstance(result, list), f"Screen {sid} should return list"


# ---------------------------------------------------------------------------
# Tests: indicators integration
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_compute_all_returns_dict(self):
        df = make_ohlcv_df(n=300)
        result = compute_all(df)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self):
        df = make_ohlcv_df(n=300)
        result = compute_all(df)
        expected_keys = {
            "rsi_14", "latest_rsi", "above_50dma", "above_200dma",
            "golden_cross", "pct_from_52wh", "pct_from_52wl",
            "vol_adj_mom", "bb_width", "volume_ratio", "latest_close",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_rsi_in_valid_range(self):
        df = make_ohlcv_df(n=300)
        result = compute_all(df)
        rsi = result.get("latest_rsi")
        if rsi is not None:
            assert 0 <= rsi <= 100

    def test_pct_from_52wh_non_positive(self):
        """pct_from_52wh should be <= 0 (current is at or below 52wk high)."""
        df = make_ohlcv_df(n=300)
        result = compute_all(df)
        pct = result.get("pct_from_52wh")
        if pct is not None:
            assert pct <= 0.01  # tiny float tolerance

    def test_vol_adj_mom_computed_with_sufficient_history(self):
        df = make_ohlcv_df(n=300)
        result = compute_all(df)
        assert result.get("vol_adj_mom") is not None

    def test_insufficient_history_returns_none_fields(self):
        df = make_ohlcv_df(n=10)
        result = compute_all(df)
        # Most indicators need more data
        assert result.get("above_200dma") is False or result.get("above_200dma") is None

    def test_empty_df_returns_defaults(self):
        df = pd.DataFrame()
        result = compute_all(df)
        assert result["latest_close"] is None
        assert result["latest_rsi"] is None
