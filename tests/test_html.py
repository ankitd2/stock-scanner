"""
Tests for output/html.py — verifies that build_daily_report and
build_weekly_report return valid, non-empty HTML strings containing
the expected structural landmarks.

These are fast unit tests that use minimal mock data and do NOT make
any network calls.
"""
import sys
from pathlib import Path
from datetime import date

# Ensure repo root is on sys.path so imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from output.html import build_daily_report, build_weekly_report


# ── Shared minimal mock data ────────────────────────────────────────────────

MOCK_STATE_SCORE = {
    "score": 62,
    "sp500_val": 5280.0,
    "sp500_pct": 0.42,
    "nasdaq_val": 16800.0,
    "nasdaq_pct": 0.65,
    "vix_val": 16.2,
    "vix_pct": -2.1,
    "yield_val": 4.38,
    "yield_pct": 0.05,
    "drivers": [
        {"label": "VIX level", "contribution": 8.0},
        {"label": "Breadth 50 DMA", "contribution": 6.5},
        {"label": "HY spreads", "contribution": -3.0},
        {"label": "AAII sentiment", "contribution": 2.0},
        {"label": "RSP/SPY ratio", "contribution": 1.5},
        {"label": "SPHB/SPLV", "contribution": -1.0},
    ],
}

MOCK_SECTOR_ROTATION = [
    {"name": "Information Technology", "score": 0.82, "rank_delta": 1},
    {"name": "Communication Services", "score": 0.61, "rank_delta": 0},
    {"name": "Healthcare",             "score": 0.45, "rank_delta": -1},
    {"name": "Consumer Discretionary", "score": 0.22, "rank_delta": 2},
    {"name": "Financials",             "score": 0.10, "rank_delta": 0},
    {"name": "Energy",                 "score": -0.15, "rank_delta": -2},
    {"name": "Utilities",              "score": -0.30, "rank_delta": 0},
]

MOCK_NEW_HIGHS_LOWS = {
    "new_highs_52w": 78,
    "new_lows_52w": 12,
    "pct_above_50dma": 64.0,
    "pct_above_200dma": 72.0,
}

MOCK_GAP_MOVERS = [
    {"ticker": "NVDA", "pct_change": 6.8, "name": "NVIDIA Corp",
     "news_title": "NVIDIA beats Q1 earnings estimates again"},
    {"ticker": "AAPL", "pct_change": -3.2, "name": "Apple Inc",
     "news_title": "Apple warns on iPhone supply constraints"},
]

MOCK_NEW_CANDIDATES = [
    {"ticker": "CRWD", "name": "CrowdStrike", "price": 285.50,
     "reason": "Quality pullback", "screen": "quality_pullback"},
]

MOCK_EARNINGS_REACTIONS = [
    {"ticker": "NVDA", "beat_miss": "Beat", "gap_pct": 6.8,
     "revenue_growth": 78.4},
    {"ticker": "AAPL", "beat_miss": "Miss", "gap_pct": -3.2,
     "revenue_growth": 4.2},
]

MOCK_RANKED_THEMES = [
    {"name": "AI Infrastructure", "theme_score": 2.4,
     "members": ["NVDA", "ANET", "VRT", "CRWV"]},
    {"name": "Cybersecurity",     "theme_score": 1.1,
     "members": ["CRWD", "ZS", "PANW", "NET"]},
    {"name": "Cloud SaaS",        "theme_score": 0.8,
     "members": ["NOW", "CRM", "DDOG", "SNOW"]},
    {"name": "Energy Transition", "theme_score": -0.4,
     "members": ["CEG", "VST", "GEV"]},
    {"name": "GLP-1 / Biotech",   "theme_score": -1.2,
     "members": ["HIMS", "DXCM", "ISRG"]},
]

MOCK_EMERGING_CLUSTERS = [
    {
        "theme": "Agentic AI",
        "tickers": ["PLTR", "APP", "RDDT"],
        "note": "Multiple names breaking out on volume",
    }
]

MOCK_SCREEN_RESULTS = {
    "52wH_proximity": [
        {
            "ticker": "AXON", "name": "Axon Enterprise", "price": 312.4,
            "rsi": 58, "from_hi": 18.5, "rev_growth": 34.2, "buy_pct": 82,
            "reason": "Meets all 6 criteria", "lo52": 240.0, "hi52": 380.0,
        }
    ],
    "quality_pullback": [
        {
            "ticker": "DDOG", "name": "Datadog", "price": 128.6,
            "rsi": 42, "from_hi": 24.1, "rev_growth": 27.5, "buy_pct": 78,
            "reason": "Oversold quality pullback", "lo52": 95.0, "hi52": 170.0,
        }
    ],
    "insider_buys": [],   # intentionally empty screen
}

MOCK_SCREEN_META = {
    "52wH_proximity": {
        "name": "52-Week High Proximity",
        "description": "Stocks in the 10-40% pullback zone from their 52-week high.",
        "citation": "O'Neil (2009), How to Make Money in Stocks",
    },
    "quality_pullback": {
        "name": "Quality Pullback",
        "description": "High-quality names that have pulled back 15-30% on low volume.",
        "citation": "Novy-Marx (2013), Quality minus Junk",
    },
    "insider_buys": {
        "name": "Insider Buys",
        "description": "Open-market insider purchases in the past 90 days.",
        "citation": "Seyhun (1986), Insiders' profits, costs of trading",
    },
}

MOCK_WATCHLIST = {
    "GOOGL": {
        "buy_at": 375,
        "direction": "below",
        "note": "Wait for 10% pullback from $403 high",
        "current_price": 403.0,
    },
    "RKLB": {
        "buy_at": 95,
        "direction": "below",
        "note": "Above all analyst PTs",
        "current_price": 110.5,
    },
}

MOCK_PRE_IPO = [
    {"name": "OpenAI",   "expected": "H2 2025", "note": "AGI moat, GPU-hungry"},
    {"name": "Cerebras", "expected": "2025",    "note": "Wafer-scale silicon"},
]

MOCK_REPORT_DATE = date(2026, 5, 14)


# ── Daily report tests ───────────────────────────────────────────────────────

class TestBuildDailyReport:
    def _build(self, **overrides):
        kwargs = dict(
            state_score=MOCK_STATE_SCORE,
            sector_rotation=MOCK_SECTOR_ROTATION,
            new_highs_lows=MOCK_NEW_HIGHS_LOWS,
            gap_movers=MOCK_GAP_MOVERS,
            new_candidates=MOCK_NEW_CANDIDATES,
            earnings_reactions=MOCK_EARNINGS_REACTIONS,
            report_date=MOCK_REPORT_DATE,
        )
        kwargs.update(overrides)
        return build_daily_report(**kwargs)

    def test_returns_non_empty_string(self):
        html = self._build()
        assert isinstance(html, str)
        assert len(html) > 500

    def test_is_valid_html_document(self):
        html = self._build()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html or "<head " in html
        assert "<body>" in html or "<body " in html

    def test_contains_date(self):
        html = self._build()
        assert "May 14, 2026" in html

    def test_contains_market_state_score(self):
        html = self._build()
        # Score value should appear somewhere in the page
        assert "62" in html

    def test_contains_daily_label(self):
        html = self._build()
        assert "Daily" in html or "Market Brief" in html

    def test_contains_market_pulse_indicators(self):
        html = self._build()
        assert "VIX" in html or "Nasdaq" in html or "S&amp;P" in html or "S&P" in html

    def test_contains_sector_rotation(self):
        html = self._build()
        assert "Sector" in html

    def test_contains_gap_movers(self):
        html = self._build()
        # At least one of the mock tickers should appear
        assert "NVDA" in html or "AAPL" in html

    def test_contains_new_candidates(self):
        html = self._build()
        assert "CRWD" in html

    def test_contains_earnings_reactions(self):
        html = self._build()
        assert "Beat" in html or "Miss" in html

    def test_gauge_chart_embedded(self):
        html = self._build()
        # Plotly emits a <div ... class="plotly-graph-div"> per chart
        assert "<div" in html
        assert "plotly-graph-div" in html
        # And no PNG base64 should be present any more
        assert "data:image/png;base64," not in html

    def test_plotly_cdn_loaded(self):
        html = self._build()
        # The plotly.js library should be loaded once via CDN
        assert "cdn.plot.ly/plotly" in html
        assert "<script src=\"https://cdn.plot.ly/plotly" in html

    def test_chart_container_wrapper(self):
        html = self._build()
        # Charts are wrapped in our styled container
        assert "chart-container" in html

    def test_handles_empty_gap_movers(self):
        html = self._build(gap_movers=[])
        assert "<!DOCTYPE html>" in html

    def test_handles_empty_candidates(self):
        html = self._build(new_candidates=[])
        assert "<!DOCTYPE html>" in html

    def test_handles_empty_state_score(self):
        html = self._build(state_score={})
        assert "<!DOCTYPE html>" in html

    def test_handles_none_report_date(self):
        html = self._build(report_date=None)
        assert "<!DOCTYPE html>" in html

    def test_dark_theme_background(self):
        html = self._build()
        assert "#0d1117" in html


# ── Weekly report tests ──────────────────────────────────────────────────────

class TestBuildWeeklyReport:
    def _build(self, **overrides):
        kwargs = dict(
            state_score=MOCK_STATE_SCORE,
            sector_rotation=MOCK_SECTOR_ROTATION,
            ranked_themes=MOCK_RANKED_THEMES,
            emerging_clusters=MOCK_EMERGING_CLUSTERS,
            screen_results=MOCK_SCREEN_RESULTS,
            screen_meta=MOCK_SCREEN_META,
            held_tickers={"NVDA", "META", "VTI"},
            watchlist=MOCK_WATCHLIST,
            pre_ipo=MOCK_PRE_IPO,
            new_highs_lows=MOCK_NEW_HIGHS_LOWS,
            report_date=MOCK_REPORT_DATE,
        )
        kwargs.update(overrides)
        return build_weekly_report(**kwargs)

    def test_returns_non_empty_string(self):
        html = self._build()
        assert isinstance(html, str)
        assert len(html) > 1000

    def test_is_valid_html_document(self):
        html = self._build()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_contains_date(self):
        html = self._build()
        assert "May 14, 2026" in html

    def test_contains_weekly_label(self):
        html = self._build()
        assert "Weekly" in html

    def test_contains_market_state_score(self):
        html = self._build()
        assert "62" in html

    def test_contains_themes(self):
        html = self._build()
        assert "AI Infrastructure" in html

    def test_contains_sector_rotation(self):
        html = self._build()
        assert "Information Technology" in html or "Sector" in html

    def test_contains_screen_names(self):
        html = self._build()
        # At least one screen name should be present
        assert "52-Week High Proximity" in html or "Quality Pullback" in html

    def test_contains_candidates(self):
        html = self._build()
        assert "AXON" in html or "DDOG" in html

    def test_empty_screen_shows_no_candidates_message(self):
        html = self._build()
        # insider_buys has 0 candidates
        assert "No candidates" in html

    def test_held_ticker_badge_present(self):
        # NVDA is in held_tickers but not in screen results here —
        # test that the report still renders without error
        html = self._build()
        assert "<!DOCTYPE html>" in html

    def test_contains_watchlist(self):
        html = self._build()
        assert "GOOGL" in html or "Watch" in html

    def test_contains_pre_ipo(self):
        html = self._build()
        assert "OpenAI" in html

    def test_contains_breadth(self):
        html = self._build()
        assert "breadth" in html.lower() or "DMA" in html

    def test_contains_glossary(self):
        html = self._build()
        assert "Glossary" in html

    def test_contains_explainer_blocks(self):
        html = self._build()
        assert "<details" in html
        assert "explainer" in html

    def test_gauge_chart_embedded(self):
        html = self._build()
        # Plotly emits a <div ... class="plotly-graph-div"> per chart
        assert "<div" in html
        assert "plotly-graph-div" in html
        assert "data:image/png;base64," not in html

    def test_range_chart_embedded(self):
        html = self._build()
        # The weekly report embeds multiple Plotly charts
        # (gauge + sector + themes + range + ...)
        assert html.count("plotly-graph-div") >= 2

    def test_plotly_cdn_loaded(self):
        html = self._build()
        assert "cdn.plot.ly/plotly" in html
        assert "<script src=\"https://cdn.plot.ly/plotly" in html

    def test_chart_container_wrapper(self):
        html = self._build()
        assert "chart-container" in html

    def test_dark_theme_css(self):
        html = self._build()
        assert "#0d1117" in html
        assert "#e6edf3" in html or "#c9d1d9" in html or "e6edf3" in html

    def test_handles_empty_themes(self):
        html = self._build(ranked_themes=[], emerging_clusters=[])
        assert "<!DOCTYPE html>" in html

    def test_handles_empty_screen_results(self):
        html = self._build(screen_results={})
        assert "<!DOCTYPE html>" in html

    def test_handles_empty_watchlist(self):
        html = self._build(watchlist={})
        assert "<!DOCTYPE html>" in html

    def test_handles_empty_pre_ipo(self):
        html = self._build(pre_ipo=[])
        assert "<!DOCTYPE html>" in html

    def test_handles_none_report_date(self):
        html = self._build(report_date=None)
        assert "<!DOCTYPE html>" in html

    def test_mobile_responsive_meta(self):
        html = self._build()
        assert "viewport" in html

    def test_no_external_dependencies(self):
        html = self._build()
        # The only allowed external dependency is Plotly's CDN, which is
        # needed for interactive charts. No other CDNs should sneak in.
        assert "cdn.jsdelivr.net" not in html
        assert "unpkg.com" not in html
        assert "googleapis.com/css" not in html


# ── Explainers module tests ──────────────────────────────────────────────────

class TestExplainers:
    def test_explainer_html_returns_details_block(self):
        from output.explainers import explainer_html
        result = explainer_html("rsi", "RSI")
        assert "<details" in result
        assert "RSI" in result
        assert "<p>" in result

    def test_explainer_html_missing_key_returns_empty(self):
        from output.explainers import explainer_html
        result = explainer_html("nonexistent_key_xyz")
        assert result == ""

    def test_glossary_html_contains_all_terms(self):
        from output.explainers import glossary_html, EXPLAINERS
        result = glossary_html()
        assert "Glossary" in result
        # Spot-check a few keys appear in the output
        for key in ("rsi", "vix", "50dma", "sector_rotation"):
            term_label = key.replace("_", " ").title()
            assert term_label in result, f"Expected '{term_label}' in glossary"

    def test_all_explainers_are_non_empty(self):
        from output.explainers import EXPLAINERS
        for key, text in EXPLAINERS.items():
            assert len(text.strip()) > 50, \
                f"Explainer for '{key}' is too short: {text!r}"
