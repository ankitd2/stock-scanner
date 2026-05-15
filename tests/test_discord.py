"""Tests for output/discord.py — dedup logic, TTL expiry, atomic write, truncation."""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the repo root is on the path so we can import output.discord
sys.path.insert(0, str(Path(__file__).parent.parent))

from output.discord import (
    ALERT_TTLS,
    STATE_FILE,
    _alert_key,
    _is_suppressed,
    _mark_fired,
    alert_big_move,
    alert_emerging_cluster,
    alert_regime_shift,
    alert_screener_candidate,
    alert_watch_trigger,
    load_state,
    save_state,
    send_webhook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_state() -> dict:
    return {"alerts": {}, "last_universe_refresh": None, "last_state_score": None}


# ---------------------------------------------------------------------------
# _alert_key
# ---------------------------------------------------------------------------

class TestAlertKey:
    def test_basic_key(self):
        assert _alert_key("watch", "GOOGL", "below", "375") == "watch:GOOGL:below:375"

    def test_single_part(self):
        assert _alert_key("move", "NVDA") == "move:NVDA"

    def test_numeric_parts_coerced_to_str(self):
        assert _alert_key("screen", "MSFT", 42) == "screen:MSFT:42"


# ---------------------------------------------------------------------------
# _is_suppressed / _mark_fired
# ---------------------------------------------------------------------------

class TestSuppression:
    def test_not_suppressed_when_missing(self):
        state = fresh_state()
        assert not _is_suppressed("watch:GOOGL:below:375", state)

    def test_suppressed_when_fired_today(self):
        state = fresh_state()
        key = "watch:GOOGL:below:375"
        state["alerts"][key] = date.today().isoformat()
        assert _is_suppressed(key, state)

    def test_suppressed_when_fired_yesterday_ttl_1(self):
        state = fresh_state()
        key = "watch:GOOGL:below:375"
        # TTL=1 means the window is "today only" (cutoff = today - 0 = today)
        state["alerts"][key] = (date.today() - timedelta(days=0)).isoformat()
        assert _is_suppressed(key, state)

    def test_not_suppressed_when_fired_before_ttl(self):
        state = fresh_state()
        key = "watch:GOOGL:below:375"
        # Filed 2 days ago, TTL is 1 day → cutoff is today, last_fired is 2 days ago → not suppressed
        state["alerts"][key] = (date.today() - timedelta(days=2)).isoformat()
        assert not _is_suppressed(key, state)

    def test_screen_ttl_7_days(self):
        state = fresh_state()
        key = "screen:NVDA:growth"
        # Fired 6 days ago → within 7-day TTL → still suppressed
        state["alerts"][key] = (date.today() - timedelta(days=6)).isoformat()
        assert _is_suppressed(key, state)

    def test_screen_ttl_7_days_expired(self):
        state = fresh_state()
        key = "screen:NVDA:growth"
        # Fired 8 days ago → outside 7-day TTL → not suppressed
        state["alerts"][key] = (date.today() - timedelta(days=8)).isoformat()
        assert not _is_suppressed(key, state)

    def test_mark_fired_sets_today(self):
        state = fresh_state()
        key = "move:TSLA"
        _mark_fired(key, state)
        assert state["alerts"][key] == date.today().isoformat()

    def test_mark_fired_creates_alerts_dict_if_missing(self):
        state = {}  # no "alerts" key
        _mark_fired("move:TSLA", state)
        assert "alerts" in state
        assert state["alerts"]["move:TSLA"] == date.today().isoformat()

    def test_invalid_date_string_not_suppressed(self):
        state = fresh_state()
        key = "watch:X:below:10"
        state["alerts"][key] = "not-a-date"
        assert not _is_suppressed(key, state)


# ---------------------------------------------------------------------------
# load_state / save_state
# ---------------------------------------------------------------------------

class TestStateIO:
    def test_load_missing_file_returns_defaults(self, tmp_path):
        missing = tmp_path / "no_such.json"
        with patch("output.discord.STATE_FILE", missing):
            state = load_state()
        assert state == {"alerts": {}, "last_universe_refresh": None, "last_state_score": None}

    def test_load_malformed_json_returns_defaults(self, tmp_path):
        f = tmp_path / "state.json"
        f.write_text("{{invalid json")
        with patch("output.discord.STATE_FILE", f):
            state = load_state()
        assert state["alerts"] == {}

    def test_load_non_dict_returns_defaults(self, tmp_path):
        f = tmp_path / "state.json"
        f.write_text("[1, 2, 3]")
        with patch("output.discord.STATE_FILE", f):
            state = load_state()
        assert state["alerts"] == {}

    def test_load_missing_alerts_key_repaired(self, tmp_path):
        f = tmp_path / "state.json"
        f.write_text('{"last_universe_refresh": null}')
        with patch("output.discord.STATE_FILE", f):
            state = load_state()
        assert state["alerts"] == {}

    def test_roundtrip_save_load(self, tmp_path):
        f = tmp_path / "state.json"
        state = {"alerts": {"watch:GOOGL:below:375": "2026-05-14"}, "last_universe_refresh": None, "last_state_score": 42}
        with patch("output.discord.STATE_FILE", f):
            save_state(state)
            loaded = load_state()
        assert loaded == state

    def test_save_atomic_uses_tmp_then_replace(self, tmp_path):
        f = tmp_path / "state.json"
        tmp_f = tmp_path / "state.json.tmp"
        state = fresh_state()
        with patch("output.discord.STATE_FILE", f):
            # After save, the .tmp file should be gone (replaced)
            save_state(state)
        assert f.exists()
        assert not tmp_f.exists()

    def test_save_creates_valid_json(self, tmp_path):
        f = tmp_path / "state.json"
        state = {"alerts": {"move:TSLA": "2026-05-14"}, "last_universe_refresh": None, "last_state_score": None}
        with patch("output.discord.STATE_FILE", f):
            save_state(state)
        with open(f) as fh:
            data = json.load(fh)
        assert data["alerts"]["move:TSLA"] == "2026-05-14"


# ---------------------------------------------------------------------------
# send_webhook
# ---------------------------------------------------------------------------

class TestSendWebhook:
    def test_empty_url_returns_false_no_request(self):
        with patch("output.discord.requests.post") as mock_post:
            result = send_webhook("hello", "")
        assert result is False
        mock_post.assert_not_called()

    def test_none_url_returns_false(self):
        with patch("output.discord.requests.post") as mock_post:
            result = send_webhook("hello", None)
        assert result is False
        mock_post.assert_not_called()

    def test_success_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 204  # Discord returns 204 No Content on success
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            result = send_webhook("hello", "https://discord.com/api/webhooks/test")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["content"] == "hello"
        assert call_kwargs[1]["timeout"] == 10

    def test_truncation_at_2000_chars(self):
        long_msg = "x" * 3000
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            send_webhook(long_msg, "https://discord.com/api/webhooks/test")
        sent_content = mock_post.call_args[1]["json"]["content"]
        assert len(sent_content) == 2000
        assert sent_content == "x" * 2000

    def test_http_error_returns_false(self, capsys):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        with patch("output.discord.requests.post", return_value=mock_resp):
            result = send_webhook("hello", "https://discord.com/api/webhooks/test")
        assert result is False
        captured = capsys.readouterr()
        assert "400" in captured.err

    def test_network_exception_returns_false(self, capsys):
        with patch("output.discord.requests.post", side_effect=ConnectionError("timeout")):
            result = send_webhook("hello", "https://discord.com/api/webhooks/test")
        assert result is False
        captured = capsys.readouterr()
        assert "timeout" in captured.err


# ---------------------------------------------------------------------------
# alert_watch_trigger
# ---------------------------------------------------------------------------

class TestAlertWatchTrigger:
    def test_fires_and_marks_state(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            result = alert_watch_trigger("GOOGL", "below", 375, 372.50, "some note", "https://wh", state)
        assert result is True
        assert "watch:GOOGL:below:375" in state["alerts"]
        content = mock_post.call_args[1]["json"]["content"]
        assert "GOOGL" in content
        assert "$372.50" in content
        assert "$375" in content
        assert "below" in content
        assert "some note" in content

    def test_suppressed_when_already_fired(self):
        state = fresh_state()
        key = "watch:GOOGL:below:375"
        state["alerts"][key] = date.today().isoformat()
        with patch("output.discord.requests.post") as mock_post:
            result = alert_watch_trigger("GOOGL", "below", 375, 372.50, "note", "https://wh", state)
        assert result is False
        mock_post.assert_not_called()

    def test_not_fired_on_webhook_failure(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=500, text="error")
        with patch("output.discord.requests.post", return_value=mock_resp):
            result = alert_watch_trigger("GOOGL", "below", 375, 372.50, "note", "https://wh", state)
        assert result is False
        # State should NOT be marked since send failed
        assert "watch:GOOGL:below:375" not in state["alerts"]


# ---------------------------------------------------------------------------
# alert_big_move
# ---------------------------------------------------------------------------

class TestAlertBigMove:
    def test_positive_move_format(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            result = alert_big_move("NVDA", 7.3, "https://wh", state)
        assert result is True
        content = mock_post.call_args[1]["json"]["content"]
        assert "NVDA" in content
        assert "+7.3%" in content

    def test_negative_move_format(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            result = alert_big_move("MU", -6.1, "https://wh", state)
        assert result is True
        content = mock_post.call_args[1]["json"]["content"]
        assert "-6.1%" in content

    def test_suppressed_on_second_call(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp):
            alert_big_move("NVDA", 7.3, "https://wh", state)
            result = alert_big_move("NVDA", 8.0, "https://wh", state)
        assert result is False


# ---------------------------------------------------------------------------
# alert_screener_candidate
# ---------------------------------------------------------------------------

class TestAlertScreenerCandidate:
    def test_fires_with_correct_format(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            result = alert_screener_candidate("PLTR", "growth_6c", "Rev +45%, BUY 85%", "https://wh", state)
        assert result is True
        content = mock_post.call_args[1]["json"]["content"]
        assert "PLTR" in content
        assert "growth_6c" in content
        assert "Rev +45%" in content

    def test_7_day_dedup(self):
        state = fresh_state()
        key = "screen:PLTR:growth_6c"
        state["alerts"][key] = (date.today() - timedelta(days=3)).isoformat()
        with patch("output.discord.requests.post") as mock_post:
            result = alert_screener_candidate("PLTR", "growth_6c", "reason", "https://wh", state)
        assert result is False
        mock_post.assert_not_called()

    def test_7_day_dedup_expired(self):
        state = fresh_state()
        key = "screen:PLTR:growth_6c"
        state["alerts"][key] = (date.today() - timedelta(days=8)).isoformat()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp):
            result = alert_screener_candidate("PLTR", "growth_6c", "reason", "https://wh", state)
        assert result is True


# ---------------------------------------------------------------------------
# alert_regime_shift
# ---------------------------------------------------------------------------

class TestAlertRegimeShift:
    def test_fires_when_delta_exceeds_15(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            result = alert_regime_shift("Entering bear", 70, 40, "https://wh", state)
        assert result is True
        content = mock_post.call_args[1]["json"]["content"]
        assert "70→40" in content
        assert "bear" in content

    def test_does_not_fire_when_delta_at_15(self):
        state = fresh_state()
        with patch("output.discord.requests.post") as mock_post:
            result = alert_regime_shift("small shift", 70, 55, "https://wh", state)
        assert result is False
        mock_post.assert_not_called()

    def test_does_not_fire_when_delta_below_15(self):
        state = fresh_state()
        with patch("output.discord.requests.post") as mock_post:
            result = alert_regime_shift("tiny shift", 70, 60, "https://wh", state)
        assert result is False
        mock_post.assert_not_called()

    def test_suppressed_on_repeat(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp):
            alert_regime_shift("Entering bear", 70, 40, "https://wh", state)
            result = alert_regime_shift("Entering bear", 70, 40, "https://wh", state)
        assert result is False

    def test_negative_delta_fires(self):
        state = fresh_state()
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp):
            result = alert_regime_shift("Rally", 40, 70, "https://wh", state)
        assert result is True


# ---------------------------------------------------------------------------
# alert_emerging_cluster
# ---------------------------------------------------------------------------

class TestAlertEmergingCluster:
    def test_fires_with_up_to_5_members(self):
        state = fresh_state()
        members = ["NVDA", "AMD", "ARM", "MRVL", "AVGO", "TSM"]
        mock_resp = MagicMock(status_code=204)
        with patch("output.discord.requests.post", return_value=mock_resp) as mock_post:
            result = alert_emerging_cluster(members, "AI chip cluster", "https://wh", state)
        assert result is True
        content = mock_post.call_args[1]["json"]["content"]
        # Only first 5 members should appear
        assert "NVDA" in content
        assert "AVGO" in content
        assert "TSM" not in content
        assert "AI chip cluster" in content

    def test_7_day_dedup(self):
        state = fresh_state()
        members = ["NVDA", "AMD"]
        cluster_id = ",".join(sorted(members))
        key = f"cluster:{cluster_id}"
        state["alerts"][key] = (date.today() - timedelta(days=2)).isoformat()
        with patch("output.discord.requests.post") as mock_post:
            result = alert_emerging_cluster(members, "desc", "https://wh", state)
        assert result is False
        mock_post.assert_not_called()

    def test_cluster_key_is_order_independent(self):
        """Same cluster in different order should produce the same key."""
        key1 = _alert_key("cluster", ",".join(sorted(["NVDA", "AMD"])))
        key2 = _alert_key("cluster", ",".join(sorted(["AMD", "NVDA"])))
        assert key1 == key2
