"""
tests/test_themes.py

Tests for analytics/themes.py and analytics/clustering.py.
No network calls — all data is synthetic.
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from analytics.clustering import (
    cluster_universe,
    compute_correlation_matrix,
    detect_emerging_clusters,
    mantegna_distance,
)
from analytics.themes import (
    _zscore_column,
    analyze_themes,
    compute_sector_rotation,
    compute_theme_strength,
    rank_themes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_series(
    n: int = 300,
    start: float = 100.0,
    drift: float = 0.0003,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.Series:
    """Create a synthetic daily price series."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, vol, n)
    prices = start * np.exp(np.cumsum(log_rets))
    idx = pd.date_range(end="2026-01-15", periods=n, freq="B")
    return pd.Series(prices, index=idx)


def _make_df(series: pd.Series) -> pd.DataFrame:
    """Wrap a price Series into a DataFrame with a 'Close' column."""
    return pd.DataFrame({"Close": series})


def _correlated_pair(
    base: pd.Series,
    correlation: float = 0.9,
    noise_vol: float = 0.008,
    seed: int = 99,
) -> pd.Series:
    """
    Create a series that is approximately `correlation`-correlated with `base`.
    """
    rng = np.random.default_rng(seed)
    base_log = np.log(base / base.shift(1)).dropna()
    noise = rng.normal(0, noise_vol, len(base_log))
    mixed_log = correlation * base_log.values + np.sqrt(1 - correlation ** 2) * noise
    prices = base.iloc[0] * np.exp(np.concatenate([[0], np.cumsum(mixed_log)]))
    return pd.Series(prices, index=base.index)


# ---------------------------------------------------------------------------
# Tests — clustering.py
# ---------------------------------------------------------------------------


class TestMantegnaDistance:
    def test_diagonal_is_zero(self):
        corr = pd.DataFrame(
            [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        dist = mantegna_distance(corr)
        np.testing.assert_array_almost_equal(np.diag(dist.values), 0.0)

    def test_formula(self):
        """d(i,j) = sqrt(2 * (1 - rho))"""
        rho = 0.5
        expected = np.sqrt(2.0 * (1.0 - rho))
        corr = pd.DataFrame(
            [[1.0, rho], [rho, 1.0]], index=["X", "Y"], columns=["X", "Y"]
        )
        dist = mantegna_distance(corr)
        assert abs(dist.loc["X", "Y"] - expected) < 1e-9

    def test_high_correlation_short_distance(self):
        """rho=0.9 => shorter distance than rho=0.1"""
        corr_high = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], index=["A", "B"], columns=["A", "B"])
        corr_low = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"])
        d_high = mantegna_distance(corr_high).loc["A", "B"]
        d_low = mantegna_distance(corr_low).loc["A", "B"]
        assert d_high < d_low

    def test_empty_input(self):
        dist = mantegna_distance(pd.DataFrame())
        assert dist.empty

    def test_symmetry(self):
        corr = pd.DataFrame(
            [[1.0, 0.6, -0.2], [0.6, 1.0, 0.4], [-0.2, 0.4, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        dist = mantegna_distance(corr)
        np.testing.assert_array_almost_equal(dist.values, dist.values.T)


class TestComputeCorrelationMatrix:
    def test_basic_returns(self):
        """Two perfectly correlated series should have corr ~1.0"""
        prices = _make_price_series(n=150, seed=1)
        h = {
            "A": _make_df(prices),
            "B": _make_df(prices * 1.05),  # same moves, different scale
        }
        corr = compute_correlation_matrix(h, lookback_days=60)
        assert not corr.empty
        assert abs(corr.loc["A", "B"] - 1.0) < 0.01

    def test_insufficient_data_excluded(self):
        """Ticker with < 80% of lookback_days should be excluded."""
        prices_long = _make_price_series(n=150, seed=2)
        prices_short = _make_price_series(n=30, seed=3)  # only 30 rows, min_obs=48 for 60d
        h = {
            "LONG": _make_df(prices_long),
            "SHORT": _make_df(prices_short),
        }
        corr = compute_correlation_matrix(h, lookback_days=60)
        if not corr.empty:
            assert "SHORT" not in corr.columns

    def test_empty_histories(self):
        corr = compute_correlation_matrix({}, lookback_days=60)
        assert corr.empty

    def test_single_ticker(self):
        prices = _make_price_series(n=100, seed=4)
        h = {"ONLY": _make_df(prices)}
        corr = compute_correlation_matrix(h, lookback_days=60)
        assert corr.empty  # need at least 2 tickers


class TestClusterUniverse:
    def _make_correlated_histories(self, n_tickers: int = 6, n_days: int = 150) -> dict[str, pd.DataFrame]:
        """Create a group of highly correlated tickers."""
        base = _make_price_series(n=n_days, seed=10)
        histories = {}
        for i in range(n_tickers):
            series = _correlated_pair(base, correlation=0.85, noise_vol=0.005, seed=100 + i)
            histories[f"T{i}"] = _make_df(series)
        return histories

    def test_finds_tight_cluster(self):
        """Highly correlated tickers should form a cluster."""
        histories = self._make_correlated_histories(n_tickers=6)
        corr = compute_correlation_matrix(histories, lookback_days=60)
        dist = mantegna_distance(corr)
        clusters = cluster_universe(dist, min_size=4, min_internal_corr=0.55)
        # Should find at least one cluster
        assert len(clusters) >= 1
        # The cluster should contain several of our tickers
        assert max(len(c) for c in clusters) >= 4

    def test_min_size_filter(self):
        """Clusters below min_size must be excluded."""
        histories = self._make_correlated_histories(n_tickers=3)  # only 3 correlated tickers
        corr = compute_correlation_matrix(histories, lookback_days=60)
        dist = mantegna_distance(corr)
        clusters = cluster_universe(dist, min_size=4)
        # With only 3 tickers, no cluster of size >= 4 can form
        assert all(len(c) >= 4 for c in clusters)

    def test_empty_distance_matrix(self):
        clusters = cluster_universe(pd.DataFrame())
        assert clusters == []

    def test_small_distance_matrix(self):
        """Only 2 tickers — below min_size=4, should return []."""
        dist = pd.DataFrame([[0.0, 0.3], [0.3, 0.0]], index=["A", "B"], columns=["A", "B"])
        clusters = cluster_universe(dist, min_size=4)
        assert clusters == []


# ---------------------------------------------------------------------------
# Tests — emerging cluster detection
# ---------------------------------------------------------------------------


class TestDetectEmergingClusters:
    def _build_histories_with_shift(
        self, n_tickers: int = 5, n_total: int = 180
    ) -> dict[str, pd.DataFrame]:
        """
        Build histories where tickers are UN-correlated in the first 60d
        but HIGHLY correlated in the last 60d.
        The middle 60d is a buffer / transition.
        """
        rng = np.random.default_rng(77)
        tickers = [f"E{i}" for i in range(n_tickers)]
        idx = pd.date_range(end="2026-01-15", periods=n_total, freq="B")
        histories: dict[str, pd.DataFrame] = {}

        for t in tickers:
            # Independent noise for first 60 days
            prior_noise = rng.normal(0, 0.015, 60)
            # Common signal for last 60 days (high correlation)
            common_signal = rng.normal(0, 0.015, 60)
            ticker_noise_late = rng.normal(0, 0.002, 60)
            middle_noise = rng.normal(0, 0.015, 60)

            all_returns = np.concatenate([prior_noise, middle_noise, common_signal + ticker_noise_late])
            prices = 100.0 * np.exp(np.cumsum(all_returns))
            histories[t] = pd.DataFrame({"Close": prices}, index=idx)

        return histories

    def test_basic_no_crash(self):
        """detect_emerging_clusters should not raise on valid input."""
        histories = self._build_histories_with_shift(n_tickers=6)
        result = detect_emerging_clusters(histories, lookback_days=60)
        assert isinstance(result, list)

    def test_insufficient_data_returns_empty(self):
        """Too few tickers should return []."""
        prices = _make_price_series(n=200, seed=5)
        h = {"A": _make_df(prices), "B": _make_df(prices * 1.01)}
        result = detect_emerging_clusters(h, lookback_days=60)
        assert result == []

    def test_empty_histories_returns_empty(self):
        result = detect_emerging_clusters({})
        assert result == []

    def test_result_structure(self):
        """Each returned cluster should have required keys."""
        histories = self._build_histories_with_shift(n_tickers=8)
        result = detect_emerging_clusters(histories, lookback_days=60)
        for cluster in result:
            assert "members" in cluster
            assert "corr_now" in cluster
            assert "corr_prior" in cluster
            assert "delta" in cluster
            assert "label" in cluster
            assert "size" in cluster
            assert cluster["size"] == len(cluster["members"])
            assert cluster["delta"] == pytest.approx(
                cluster["corr_now"] - cluster["corr_prior"], abs=0.001
            )

    def test_max_3_results(self):
        """Should return at most 3 clusters."""
        histories = self._build_histories_with_shift(n_tickers=8)
        result = detect_emerging_clusters(histories, lookback_days=60)
        assert len(result) <= 3

    def test_sorted_by_delta(self):
        """Results should be sorted descending by delta."""
        histories = self._build_histories_with_shift(n_tickers=8)
        result = detect_emerging_clusters(histories, lookback_days=60)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["delta"] >= result[i + 1]["delta"]


# ---------------------------------------------------------------------------
# Tests — theme scoring
# ---------------------------------------------------------------------------


class TestZscoreColumn:
    def test_basic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        z = _zscore_column(vals)
        assert len(z) == 5
        assert abs(sum(z)) < 1e-9  # mean == 0
        assert abs(np.std(z) - 1.0) < 0.01

    def test_constant_returns_zeros(self):
        vals = [5.0, 5.0, 5.0]
        z = _zscore_column(vals)
        assert z == [0.0, 0.0, 0.0]

    def test_single_element(self):
        z = _zscore_column([42.0])
        assert z == [0.0]


class TestComputeThemeStrength:
    def _make_histories(self, tickers: list[str], n: int = 300) -> dict[str, pd.DataFrame]:
        histories = {}
        for i, t in enumerate(tickers):
            histories[t] = _make_df(_make_price_series(n=n, seed=i * 7 + 1))
        return histories

    def test_returns_six_components(self):
        tickers = ["A", "B", "C", "D"]
        h = self._make_histories(tickers)
        spy = _make_df(_make_price_series(n=300, seed=999))
        result = compute_theme_strength("test", tickers, h, spy)
        assert set(result.keys()) == {
            "rel_strength_4w",
            "rel_strength_12w",
            "pct_above_50dma",
            "internal_cohesion",
            "pct_near_52wh",
            "median_rsi",
            "available_members",
        }

    def test_returns_empty_on_insufficient_data(self):
        """Fewer than 3 members with data should return {}."""
        tickers = ["A", "B"]
        h = self._make_histories(tickers)
        spy = _make_df(_make_price_series(n=300, seed=999))
        result = compute_theme_strength("test", tickers, h, spy)
        assert result == {}

    def test_missing_members_handled_gracefully(self):
        """Members not in histories should be silently skipped."""
        h = {
            "REAL": _make_df(_make_price_series(n=300, seed=1)),
            "ALSO_REAL": _make_df(_make_price_series(n=300, seed=2)),
            "THIRD": _make_df(_make_price_series(n=300, seed=3)),
        }
        spy = _make_df(_make_price_series(n=300, seed=999))
        members = ["REAL", "ALSO_REAL", "THIRD", "MISSING1", "MISSING2"]
        result = compute_theme_strength("test", members, h, spy)
        assert result  # should succeed with 3 valid members
        assert "MISSING1" not in result["available_members"]
        assert "MISSING2" not in result["available_members"]

    def test_pct_above_50dma_range(self):
        """pct_above_50dma should be between 0 and 100."""
        tickers = ["A", "B", "C", "D", "E"]
        h = self._make_histories(tickers)
        spy = _make_df(_make_price_series(n=300, seed=999))
        result = compute_theme_strength("test", tickers, h, spy)
        assert 0.0 <= result["pct_above_50dma"] <= 100.0

    def test_median_rsi_range(self):
        """median_rsi should be between 0 and 100."""
        tickers = ["A", "B", "C", "D"]
        h = self._make_histories(tickers)
        spy = _make_df(_make_price_series(n=300, seed=999))
        result = compute_theme_strength("test", tickers, h, spy)
        assert 0.0 <= result["median_rsi"] <= 100.0

    def test_internal_cohesion_range(self):
        """internal_cohesion should be between -1 and 1."""
        tickers = ["A", "B", "C", "D"]
        h = self._make_histories(tickers)
        spy = _make_df(_make_price_series(n=300, seed=999))
        result = compute_theme_strength("test", tickers, h, spy)
        assert -1.0 <= result["internal_cohesion"] <= 1.0

    def test_empty_histories(self):
        result = compute_theme_strength("test", ["A", "B"], {}, pd.DataFrame())
        assert result == {}


class TestRankThemes:
    def _build_universe(self, n: int = 300) -> dict[str, pd.DataFrame]:
        """Build enough tickers to satisfy themes.yaml members."""
        tickers = [
            "NVDA", "AMD", "AVGO", "ANET", "CRDO", "VRT", "GEV", "APLD", "CRWV",
            "ARM", "TSM", "MU", "AMAT", "MRVL",
            "MSFT", "GOOGL", "META", "PLTR", "APP", "CRM", "NOW", "SNOW", "DDOG",
            "HUBS", "ADBE", "INTU",
            "CRWD", "ZS", "PANW", "NET", "AXON",
            "AMZN", "SHOP", "MELI", "BKNG", "ABNB", "UBER", "TTD",
            "COIN", "SOFI",
            "NFLX", "RDDT",
            "RKLB", "KTOS",
            "CEG", "VST",
            "ISRG", "DXCM", "HIMS", "VEEV",
            "TSLA", "IBM",
            "SPY",
        ]
        h: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            h[t] = _make_df(_make_price_series(n=n, seed=i + 100))
        return h

    def test_returns_list(self):
        h = self._build_universe()
        spy = h.get("SPY", pd.DataFrame())
        result = rank_themes(h, spy)
        assert isinstance(result, list)

    def test_sorted_by_score_desc(self):
        h = self._build_universe()
        spy = h.get("SPY", pd.DataFrame())
        result = rank_themes(h, spy)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_sequential(self):
        h = self._build_universe()
        spy = h.get("SPY", pd.DataFrame())
        result = rank_themes(h, spy)
        for i, r in enumerate(result):
            assert r["rank"] == i + 1

    def test_each_result_has_required_keys(self):
        h = self._build_universe()
        spy = h.get("SPY", pd.DataFrame())
        result = rank_themes(h, spy)
        required = {"key", "name", "description", "score", "components", "rank", "members", "available_members"}
        for r in result:
            assert required.issubset(r.keys()), f"Missing keys in {r.get('key')}: {required - r.keys()}"

    def test_empty_histories_returns_empty(self):
        result = rank_themes({}, pd.DataFrame())
        assert result == []


# ---------------------------------------------------------------------------
# Tests — sector rotation
# ---------------------------------------------------------------------------


class TestComputeSectorRotation:
    SECTOR_ETFS = ["XLK", "XLY", "XLC", "XLE", "XLB", "XLI", "XLU", "XLP", "XLV", "XLF", "XLRE"]

    def _make_sector_histories(self, n: int = 300) -> dict[str, pd.DataFrame]:
        h: dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(self.SECTOR_ETFS):
            h[sym] = _make_df(_make_price_series(n=n, seed=i * 13 + 50))
        h["SPY"] = _make_df(_make_price_series(n=n, seed=999))
        return h

    def test_returns_sorted_list(self):
        h = self._make_sector_histories()
        result = compute_sector_rotation(h, h["SPY"])
        assert isinstance(result, list)
        # Sorted descending by rank_score
        scores = [r["rank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_delta_with_prev_ranks(self):
        h = self._make_sector_histories()
        # Fake previous ranks: invert current ranks
        result_no_prev = compute_sector_rotation(h, h["SPY"])
        prev_ranks = {r["symbol"]: (len(result_no_prev) + 1 - r["rank"]) for r in result_no_prev}
        result_with_prev = compute_sector_rotation(h, h["SPY"], previous_ranks=prev_ranks)
        for r in result_with_prev:
            assert r["rank_delta"] is not None

    def test_each_result_has_required_keys(self):
        h = self._make_sector_histories()
        result = compute_sector_rotation(h, h["SPY"])
        required = {"symbol", "name", "rank_score", "rs_3m", "rs_6m", "rank", "rank_delta", "ytd_return"}
        for r in result:
            assert required.issubset(r.keys())

    def test_missing_etf_handled(self):
        """Missing ETF data should be skipped, not raise."""
        h = self._make_sector_histories()
        del h["XLK"]  # remove one sector
        result = compute_sector_rotation(h, h["SPY"])
        symbols = [r["symbol"] for r in result]
        assert "XLK" not in symbols

    def test_empty_histories(self):
        result = compute_sector_rotation({}, pd.DataFrame())
        assert result == []


# ---------------------------------------------------------------------------
# Tests — analyze_themes (integration)
# ---------------------------------------------------------------------------


class TestAnalyzeThemes:
    def _build_full_universe(self, n: int = 300) -> dict[str, pd.DataFrame]:
        tickers = [
            "NVDA", "AMD", "AVGO", "ANET", "CRDO", "VRT", "GEV", "APLD", "CRWV",
            "ARM", "TSM", "MU", "AMAT", "MRVL",
            "MSFT", "GOOGL", "META", "PLTR", "APP", "CRM", "NOW", "SNOW", "DDOG",
            "HUBS", "ADBE", "INTU",
            "CRWD", "ZS", "PANW", "NET", "AXON",
            "AMZN", "SHOP", "MELI", "BKNG", "ABNB", "UBER", "TTD",
            "COIN", "SOFI",
            "NFLX", "RDDT",
            "RKLB", "KTOS",
            "CEG", "VST",
            "ISRG", "DXCM", "HIMS", "VEEV",
            "TSLA", "IBM",
            # Sector ETFs
            "XLK", "XLY", "XLC", "XLE", "XLB", "XLI", "XLU", "XLP", "XLV", "XLF", "XLRE",
            "SPY",
        ]
        h: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            h[t] = _make_df(_make_price_series(n=n, seed=i + 200))
        return h

    def test_returns_expected_keys(self):
        h = self._build_full_universe()
        result = analyze_themes(h)
        expected_keys = {
            "ranked_themes",
            "top_themes",
            "sector_rotation",
            "rotation_call",
            "emerging_clusters",
            "stovall_phase",
        }
        assert expected_keys.issubset(result.keys())

    def test_top_themes_is_top_5(self):
        h = self._build_full_universe()
        result = analyze_themes(h)
        assert len(result["top_themes"]) <= 5
        if len(result["ranked_themes"]) >= 5:
            assert len(result["top_themes"]) == 5

    def test_top_themes_matches_ranked_head(self):
        h = self._build_full_universe()
        result = analyze_themes(h)
        for i, theme in enumerate(result["top_themes"]):
            assert theme["key"] == result["ranked_themes"][i]["key"]

    def test_stovall_phase_valid(self):
        h = self._build_full_universe()
        result = analyze_themes(h)
        valid_phases = {"Early Cycle", "Mid Cycle", "Late Cycle", "Recession"}
        assert result["stovall_phase"] in valid_phases

    def test_sector_rotation_all_11(self):
        h = self._build_full_universe()
        result = analyze_themes(h)
        assert len(result["sector_rotation"]) == 11

    def test_empty_histories(self):
        """Should not crash on empty input."""
        result = analyze_themes({})
        assert result["ranked_themes"] == []
        assert result["top_themes"] == []
        assert result["sector_rotation"] == []
        assert result["stovall_phase"] == "Early Cycle"

    def test_emerging_clusters_is_list(self):
        h = self._build_full_universe()
        result = analyze_themes(h)
        assert isinstance(result["emerging_clusters"], list)
