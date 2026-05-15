import json, os, sys, requests
from datetime import date, timedelta
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent / "state.json"

# Alert TTLs in days
ALERT_TTLS = {
    "watch":   1,   # watch price trigger — fire once per 24h
    "move":    1,   # big portfolio move — once per 24h
    "screen":  7,   # new screener candidate — once per 7 days
    "regime":  1,   # market state regime shift — once per 24h
    "cluster": 7,   # emerging theme cluster — once per 7 days
}


def load_state() -> dict:
    """Read state.json. Returns {"alerts": {}, ...} if missing or malformed."""
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("state.json root is not a dict")
        if "alerts" not in data or not isinstance(data["alerts"], dict):
            data["alerts"] = {}
        return data
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {"alerts": {}, "last_universe_refresh": None, "last_state_score": None}


def save_state(state: dict) -> None:
    """Atomically write state.json (write to .tmp then rename)."""
    tmp = Path(str(STATE_FILE) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_FILE)


def _alert_key(alert_type: str, *parts: str) -> str:
    """Build a dedup key: e.g. 'watch:GOOGL:below:375'"""
    return ":".join([alert_type] + [str(p) for p in parts])


def _is_suppressed(key: str, state: dict) -> bool:
    """Return True if this alert key was fired within its TTL window."""
    alerts = state.get("alerts", {})
    if key not in alerts:
        return False
    # Derive alert_type from the key prefix
    alert_type = key.split(":")[0]
    ttl_days = ALERT_TTLS.get(alert_type, 1)
    last_fired_str = alerts[key]
    try:
        last_fired = date.fromisoformat(last_fired_str)
    except (ValueError, TypeError):
        return False
    cutoff = date.today() - timedelta(days=ttl_days - 1)
    return last_fired >= cutoff


def _mark_fired(key: str, state: dict) -> None:
    """Record that `key` fired today in state['alerts']."""
    if "alerts" not in state:
        state["alerts"] = {}
    state["alerts"][key] = date.today().isoformat()


def send_webhook(message: str, webhook_url: str) -> bool:
    """
    POST message to Discord webhook. Truncates to 2000 chars.
    Returns True on success (HTTP 2xx), False otherwise.
    Logs failures to stderr. Never raises.
    """
    if not webhook_url:
        return False
    try:
        resp = requests.post(
            webhook_url,
            json={"content": message[:2000]},
            timeout=10,
        )
        if resp.status_code >= 200 and resp.status_code < 300:
            return True
        print(
            f"Discord webhook error: HTTP {resp.status_code} — {resp.text[:200]}",
            file=sys.stderr,
        )
        return False
    except Exception as exc:
        print(f"Discord webhook exception: {exc}", file=sys.stderr)
        return False


def alert_watch_trigger(
    ticker: str,
    direction: str,
    target: float,
    current_price: float,
    note: str,
    webhook_url: str,
    state: dict,
) -> bool:
    """
    Fire a watch-price-hit alert if not suppressed.
    Message format: "🎯 {ticker} hit ${current:.2f} (target: ${target} {direction})\n{note}"
    Returns True if alert was sent (not suppressed).
    """
    key = _alert_key("watch", ticker, direction, str(target))
    if _is_suppressed(key, state):
        return False
    message = f"🎯 {ticker} hit ${current_price:.2f} (target: ${target} {direction})\n{note}"
    sent = send_webhook(message, webhook_url)
    if sent:
        _mark_fired(key, state)
    return sent


def alert_big_move(
    ticker: str,
    pct_change: float,
    webhook_url: str,
    state: dict,
) -> bool:
    """
    Fire a big-move alert (±N% in a day) if not suppressed.
    Message format: "📈 {ticker} {+/-}{pct:.1f}% today (alert threshold triggered)"
    """
    key = _alert_key("move", ticker)
    if _is_suppressed(key, state):
        return False
    sign = "+" if pct_change >= 0 else ""
    message = f"📈 {ticker} {sign}{pct_change:.1f}% today (alert threshold triggered)"
    sent = send_webhook(message, webhook_url)
    if sent:
        _mark_fired(key, state)
    return sent


def alert_screener_candidate(
    ticker: str,
    screen_name: str,
    reason: str,
    webhook_url: str,
    state: dict,
) -> bool:
    """
    Fire a new screener candidate alert (7-day TTL) if not suppressed.
    Message format: "🟢 New candidate: {ticker}\nScreen: {screen_name}\n{reason}"
    """
    key = _alert_key("screen", ticker, screen_name)
    if _is_suppressed(key, state):
        return False
    message = f"🟢 New candidate: {ticker}\nScreen: {screen_name}\n{reason}"
    sent = send_webhook(message, webhook_url)
    if sent:
        _mark_fired(key, state)
    return sent


def alert_regime_shift(
    description: str,
    score_before: int,
    score_after: int,
    webhook_url: str,
    state: dict,
) -> bool:
    """
    Fire a market regime shift alert if score delta > 15 points and not suppressed.
    Message format: "⚠️ Market State shifted: {score_before}→{score_after}\n{description}"
    """
    if abs(score_after - score_before) <= 15:
        return False
    key = _alert_key("regime", str(score_before), str(score_after))
    if _is_suppressed(key, state):
        return False
    message = f"⚠️ Market State shifted: {score_before}→{score_after}\n{description}"
    sent = send_webhook(message, webhook_url)
    if sent:
        _mark_fired(key, state)
    return sent


def alert_emerging_cluster(
    members: list,
    description: str,
    webhook_url: str,
    state: dict,
) -> bool:
    """
    Fire an emerging cluster discovery alert (7-day TTL) if not suppressed.
    Message format: "🌐 Emerging theme detected: {', '.join(members[:5])}\n{description}"
    """
    cluster_id = ",".join(sorted(members))
    key = _alert_key("cluster", cluster_id)
    if _is_suppressed(key, state):
        return False
    names = ", ".join(members[:5])
    message = f"🌐 Emerging theme detected: {names}\n{description}"
    sent = send_webhook(message, webhook_url)
    if sent:
        _mark_fired(key, state)
    return sent
