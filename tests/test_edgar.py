"""
tests/test_edgar.py — unit tests for data/edgar.py

All HTTP/network calls are mocked so the suite runs offline and fast.
Covers:
  - openinsider HTML parsing (purchase rows, malformed input, P-only filter)
  - get_insider_purchases caching / failure modes
  - cluster_buy_signal: min insiders, min value, no-selling, sort order
  - Empty input → empty output
  - Form 4 XML parsing (transaction code filter)
  - Network failure → graceful empty
"""

import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import data.edgar as edgar


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_openinsider_row(
    *,
    filing_date: str = "",
    trade_date: str,
    ticker: str,
    insider: str,
    title: str = "CEO",
    trade_type: str = "P - Purchase",
    price: str = "$10.00",
    qty: str = "1,000",
    owned: str = "5,000",
    dchg: str = "+25%",
    value: str = "$10,000",
) -> str:
    """Build a single openinsider-style <tr> row string."""
    cells = [
        "",            # X marker
        filing_date,
        trade_date,
        ticker,
        "Company X",
        insider,
        title,
        trade_type,
        price,
        qty,
        owned,
        dchg,
        value,
    ]
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _wrap_html(rows: list[str]) -> str:
    return "<html><body><table>" + "".join(rows) + "</table></body></html>"


@pytest.fixture(autouse=True)
def _reset_edgar_caches():
    """Wipe module-level caches before each test."""
    edgar._reset_caches()
    yield
    edgar._reset_caches()


# ---------------------------------------------------------------------------
# openinsider HTML parsing
# ---------------------------------------------------------------------------

class TestOpenInsiderParsing:
    def test_parses_single_purchase_row(self):
        today = date.today().strftime("%Y-%m-%d")
        html = _wrap_html([
            _make_openinsider_row(
                trade_date=today,
                ticker="AAPL",
                insider="Tim Cook",
                title="CEO",
                price="$150.00",
                qty="2,000",
                value="$300,000",
            ),
        ])
        rows = edgar._parse_openinsider_html(html)
        assert len(rows) == 1
        r = rows[0]
        assert r["ticker"] == "AAPL"
        assert r["insider"] == "Tim Cook"
        assert r["shares"] == 2000
        assert r["price"] == 150.0
        assert r["value"] == 300_000.0
        assert r["code"] == "P"
        assert r["date"] == today

    def test_filters_non_purchase_rows(self):
        today = date.today().strftime("%Y-%m-%d")
        html = _wrap_html([
            _make_openinsider_row(
                trade_date=today,
                ticker="AAPL",
                insider="Tim Cook",
                trade_type="S - Sale",   # not a purchase
            ),
            _make_openinsider_row(
                trade_date=today,
                ticker="MSFT",
                insider="Satya Nadella",
                trade_type="P - Purchase",
            ),
        ])
        rows = edgar._parse_openinsider_html(html)
        tickers = {r["ticker"] for r in rows}
        assert tickers == {"MSFT"}

    def test_ignores_malformed_rows(self):
        # A row with only 3 cells should be skipped, not crash.
        bad_html = "<tr><td>1</td><td>2</td><td>3</td></tr>"
        today = date.today().strftime("%Y-%m-%d")
        good = _make_openinsider_row(
            trade_date=today,
            ticker="NVDA",
            insider="Jensen Huang",
        )
        html = _wrap_html([bad_html, good])
        rows = edgar._parse_openinsider_html(html)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "NVDA"

    def test_empty_html_returns_empty(self):
        assert edgar._parse_openinsider_html("") == []
        assert edgar._parse_openinsider_html("<html></html>") == []


# ---------------------------------------------------------------------------
# get_insider_purchases — high-level aggregator
# ---------------------------------------------------------------------------

class TestGetInsiderPurchases:
    def test_groups_by_ticker(self):
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        html = _wrap_html([
            _make_openinsider_row(trade_date=today, ticker="AAPL",
                                   insider="A", value="$10,000", qty="100"),
            _make_openinsider_row(trade_date=yesterday, ticker="AAPL",
                                   insider="B", value="$20,000", qty="200"),
            _make_openinsider_row(trade_date=today, ticker="MSFT",
                                   insider="C", value="$50,000", qty="500"),
        ])
        with patch.object(edgar, "_fetch_openinsider_html", return_value=html):
            grouped = edgar.get_insider_purchases(lookback_days=30, use_cache=False)
        assert set(grouped.keys()) == {"AAPL", "MSFT"}
        assert len(grouped["AAPL"]) == 2
        assert len(grouped["MSFT"]) == 1

    def test_drops_old_rows(self):
        # 60 days ago should be outside a 30-day lookback window.
        old = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        today = date.today().strftime("%Y-%m-%d")
        html = _wrap_html([
            _make_openinsider_row(trade_date=old, ticker="AAPL", insider="A"),
            _make_openinsider_row(trade_date=today, ticker="MSFT", insider="B"),
        ])
        with patch.object(edgar, "_fetch_openinsider_html", return_value=html):
            grouped = edgar.get_insider_purchases(lookback_days=30, use_cache=False)
        assert "AAPL" not in grouped
        assert "MSFT" in grouped

    def test_returns_empty_on_network_failure(self):
        with patch.object(edgar, "_fetch_openinsider_html", return_value=None):
            grouped = edgar.get_insider_purchases(lookback_days=30, use_cache=False)
        assert grouped == {}

    def test_returns_empty_on_garbled_response(self):
        with patch.object(edgar, "_fetch_openinsider_html",
                           return_value="<<malformed!! not html>>"):
            grouped = edgar.get_insider_purchases(lookback_days=30, use_cache=False)
        assert grouped == {}

    def test_cache_avoids_second_fetch(self):
        today = date.today().strftime("%Y-%m-%d")
        html = _wrap_html([
            _make_openinsider_row(trade_date=today, ticker="AAPL", insider="A"),
        ])
        with patch.object(edgar, "_fetch_openinsider_html",
                           return_value=html) as mock_fetch:
            edgar.get_insider_purchases(lookback_days=30, use_cache=True)
            edgar.get_insider_purchases(lookback_days=30, use_cache=True)
            assert mock_fetch.call_count == 1


# ---------------------------------------------------------------------------
# cluster_buy_signal — pure logic
# ---------------------------------------------------------------------------

class TestClusterBuySignal:
    def _purchase(self, insider, value=200_000, days_ago=5, shares=1000, price=200.0):
        d = (date.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        return {
            "insider": insider,
            "title": "Director",
            "date": d,
            "shares": shares,
            "price": price,
            "value": value,
        }

    def test_empty_input_returns_empty(self):
        assert edgar.cluster_buy_signal({}) == []

    def test_single_insider_rejected(self):
        purchases = {"AAA": [self._purchase("Alice", value=2_000_000)]}
        out = edgar.cluster_buy_signal(purchases, min_insiders=3)
        assert out == []

    def test_three_insiders_above_value_passes(self):
        purchases = {
            "BBB": [
                self._purchase("Alice", value=200_000),
                self._purchase("Bob",   value=200_000),
                self._purchase("Carol", value=200_000),
            ]
        }
        out = edgar.cluster_buy_signal(
            purchases, min_insiders=3, min_value=500_000
        )
        assert len(out) == 1
        sig = out[0]
        assert sig["ticker"] == "BBB"
        assert sig["n_insiders"] == 3
        assert sig["total_value"] == 600_000
        assert sig["no_selling"] is True
        assert set(sig["insider_names"]) == {"Alice", "Bob", "Carol"}

    def test_below_value_threshold_rejected(self):
        purchases = {
            "CCC": [
                self._purchase("Alice", value=100_000),
                self._purchase("Bob",   value=100_000),
                self._purchase("Carol", value=100_000),
            ]
        }
        out = edgar.cluster_buy_signal(
            purchases, min_insiders=3, min_value=500_000
        )
        assert out == []

    def test_no_selling_filter_with_sales(self):
        purchases = {
            "DDD": [
                self._purchase("Alice", value=200_000),
                self._purchase("Bob",   value=200_000),
                self._purchase("Carol", value=200_000),
            ]
        }
        sales = {
            "DDD": [self._purchase("Dave", value=50_000, days_ago=3)],
        }
        out = edgar.cluster_buy_signal(
            purchases, min_insiders=3, min_value=500_000,
            insider_sales=sales,
        )
        assert len(out) == 1
        assert out[0]["no_selling"] is False

    def test_sorted_by_total_value_desc(self):
        purchases = {
            "LOW": [
                self._purchase("A", value=200_000),
                self._purchase("B", value=200_000),
                self._purchase("C", value=200_000),
            ],
            "HI": [
                self._purchase("X", value=1_000_000),
                self._purchase("Y", value=1_000_000),
                self._purchase("Z", value=1_000_000),
            ],
        }
        out = edgar.cluster_buy_signal(purchases, min_insiders=3, min_value=500_000)
        assert [s["ticker"] for s in out] == ["HI", "LOW"]
        assert out[0]["total_value"] > out[1]["total_value"]

    def test_out_of_window_dates_excluded(self):
        old = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
        purchases = {
            "OLD": [
                {"insider": "A", "date": old, "shares": 100, "price": 1, "value": 200_000},
                {"insider": "B", "date": old, "shares": 100, "price": 1, "value": 200_000},
                {"insider": "C", "date": old, "shares": 100, "price": 1, "value": 200_000},
            ]
        }
        out = edgar.cluster_buy_signal(purchases, min_insiders=3, window_days=30)
        assert out == []

    def test_duplicate_insider_name_counted_once(self):
        # Same insider trading twice should only count as 1 insider.
        purchases = {
            "DUP": [
                self._purchase("Alice", value=300_000),
                self._purchase("Alice", value=300_000),  # same person
                self._purchase("Bob",   value=300_000),
            ]
        }
        out = edgar.cluster_buy_signal(purchases, min_insiders=3, min_value=500_000)
        assert out == []  # only 2 distinct insiders → rejected


# ---------------------------------------------------------------------------
# Form 4 XML parsing
# ---------------------------------------------------------------------------

class TestForm4XmlParsing:
    _SAMPLE_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerName>Apple Inc.</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Tim Cook</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-04-15</value></transactionDate>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.00</value></transactionPricePerShare>
      </transactionAmounts>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-04-15</value></transactionDate>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>150.00</value></transactionPricePerShare>
      </transactionAmounts>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

    def test_extracts_issuer_and_owner(self):
        parsed = edgar._parse_form4_xml_text(self._SAMPLE_XML)
        assert parsed is not None
        assert parsed["ticker"] == "AAPL"
        assert parsed["company"] == "Apple Inc."
        assert parsed["insider_name"] == "Tim Cook"
        assert parsed["insider_title"] == "CEO"

    def test_only_purchase_transactions_returned(self):
        parsed = edgar._parse_form4_xml_text(self._SAMPLE_XML)
        assert parsed is not None
        # Only the "P" transaction should survive; the "S" is dropped.
        assert len(parsed["transactions"]) == 1
        txn = parsed["transactions"][0]
        assert txn["code"] == "P"
        assert txn["shares"] == 1000
        assert txn["price"] == 150.0
        assert txn["value"] == 150_000.0

    def test_malformed_xml_returns_none(self):
        assert edgar._parse_form4_xml_text("not xml at all") is None
        assert edgar._parse_form4_xml_text("<unclosed>") is None


# ---------------------------------------------------------------------------
# get_insider_cluster (per-ticker convenience)
# ---------------------------------------------------------------------------

class TestGetInsiderCluster:
    def test_returns_none_for_unknown_ticker(self):
        with patch.object(edgar, "_fetch_openinsider_html", return_value=None):
            assert edgar.get_insider_cluster("ZZZZ") is None

    def test_returns_signal_dict_for_qualifying_ticker(self):
        today = date.today().strftime("%Y-%m-%d")
        html = _wrap_html([
            _make_openinsider_row(trade_date=today, ticker="QQQQ", insider="A",
                                   value="$200,000", qty="100", price="$2000"),
            _make_openinsider_row(trade_date=today, ticker="QQQQ", insider="B",
                                   value="$200,000", qty="100", price="$2000"),
            _make_openinsider_row(trade_date=today, ticker="QQQQ", insider="C",
                                   value="$200,000", qty="100", price="$2000"),
        ])
        with patch.object(edgar, "_fetch_openinsider_html", return_value=html):
            out = edgar.get_insider_cluster("QQQQ", days=30)
        assert out is not None
        assert out["ticker"] == "QQQQ"
        assert out["n_buyers"] == 3
        assert out["total_value"] >= 500_000


# ---------------------------------------------------------------------------
# Integration: Screen 7 dispatch via analytics.screens.run_screen
# ---------------------------------------------------------------------------

class TestScreen7Integration:
    def test_run_screen_7_returns_candidate_with_expected_fields(self):
        """Screen 7 should produce candidates with the standard field contract."""
        # Set up a fake universe + mock openinsider response.
        import pandas as pd
        import numpy as np
        from analytics.screens import run_screen, _reset_cluster_cache

        # Synthetic price history (300 days)
        rng = np.random.default_rng(0)
        rets = 0.001 + 0.012 * rng.standard_normal(300)
        close = 100.0 * np.cumprod(1 + rets)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=300, freq="B")
        df = pd.DataFrame({
            "Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98,
            "Close": close, "Volume": rng.integers(100_000, 500_000, 300),
        }, index=dates)

        today = date.today().strftime("%Y-%m-%d")
        html = _wrap_html([
            _make_openinsider_row(trade_date=today, ticker="ZZZZ",
                                   insider="Alice", value="$200,000",
                                   qty="100", price="$2000"),
            _make_openinsider_row(trade_date=today, ticker="ZZZZ",
                                   insider="Bob", value="$200,000",
                                   qty="100", price="$2000"),
            _make_openinsider_row(trade_date=today, ticker="ZZZZ",
                                   insider="Carol", value="$200,000",
                                   qty="100", price="$2000"),
        ])
        infos = {"ZZZZ": {
            "ticker": "ZZZZ", "name": "ZZZZ Corp",
            "mcap": 5e9, "sell_count": 0, "buy_pct": 70,
            "rev_growth": 20.0, "gross_margin": 50.0,
            "tgt_mean": 150.0, "price": 100.0,
        }}
        _reset_cluster_cache()
        with patch.object(edgar, "_fetch_openinsider_html", return_value=html):
            results = run_screen(7, {"ZZZZ": df}, infos, set())

        assert len(results) == 1
        c = results[0]
        # Verify the standard candidate contract.
        for key in (
            "ticker", "screen_id", "screen_name", "reason", "held", "price",
            "mcap", "rsi", "pct_from_52wh", "rev_growth", "buy_pct",
            "upside_to_pt", "vol_adj_mom", "hi52", "lo52", "name", "sector",
        ):
            assert key in c, f"missing key {key}"
        assert c["ticker"] == "ZZZZ"
        assert c["screen_id"] == 7
        assert "insider_buys" in c
        assert c["insider_buys"]["n_insiders"] == 3

    def test_run_screen_7_empty_on_network_failure(self):
        """When openinsider returns None, Screen 7 should be empty (no crash)."""
        import pandas as pd
        import numpy as np
        from analytics.screens import run_screen, _reset_cluster_cache

        rng = np.random.default_rng(0)
        rets = 0.001 + 0.012 * rng.standard_normal(200)
        close = 100.0 * np.cumprod(1 + rets)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=200, freq="B")
        df = pd.DataFrame({
            "Open": close, "High": close, "Low": close, "Close": close,
            "Volume": rng.integers(100_000, 500_000, 200),
        }, index=dates)
        infos = {"FOO": {"mcap": 5e9, "sell_count": 0}}

        _reset_cluster_cache()
        with patch.object(edgar, "_fetch_openinsider_html", return_value=None):
            results = run_screen(7, {"FOO": df}, infos, set())
        assert results == []
