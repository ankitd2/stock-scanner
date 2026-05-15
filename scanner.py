#!/usr/bin/env python3
"""
Market Intelligence Scanner — orchestrator
Runs daily (pre/post market) or weekly (full deep dive).
"""
import json, os, sys, argparse
import pandas as pd
from datetime import date
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from data.fetcher import bulk_history, get_ticker_info, get_index_data, get_news
    from data.fred import latest_fred, get_fred_series, fred_zscore
    from data.aaii import latest_aaii
    from data.cache import RunCache
    from analytics.market_state import compute_market_state
    from analytics.themes import analyze_themes
    from analytics.screens import run_all_screens, SCREEN_META
    from output.html import build_daily_report, build_weekly_report
    from output.discord import (load_state, save_state,
                                 alert_watch_trigger, alert_big_move,
                                 alert_screener_candidate, alert_regime_shift,
                                 alert_emerging_cluster)
    from output.pages import stage_report, update_index
    _MODULES_AVAILABLE = True
except ImportError as _import_err:
    _MODULES_AVAILABLE = False
    _IMPORT_ERR = _import_err


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        cfg = json.load(f)
    # Support both "portfolio" (legacy key) and "held"
    held = cfg.get("held", cfg.get("portfolio", []))
    return {
        "held": held,
        "watchlist": cfg.get("watchlist", {}),
        "pre_ipo": cfg.get("pre_ipo", []),
        "alert_move_pct": cfg.get("alert_move_pct", 5.0),
        "discord_webhook": os.environ.get("DISCORD_WEBHOOK", cfg.get("discord_webhook", "")),
    }


def get_universe_tickers() -> list[str]:
    """Load Russell 1000 + growth extension, deduplicated."""
    r1k_path = Path("universe/russell1000.csv")
    ext_path = Path("universe/growth_extension.csv")
    r1k = r1k_path.read_text().splitlines() if r1k_path.exists() else []
    ext = ext_path.read_text().splitlines() if ext_path.exists() else []
    seen, result = set(), []
    for t in r1k + ext:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    # Fallback: built-in universe if files are missing
    if not result:
        result = [
            "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AAPL",
            "CRM", "NOW", "SNOW", "DDOG", "NET", "ZS", "CRWD", "PANW",
            "PLTR", "APP", "TTD", "HUBS", "SHOP", "RDDT", "NFLX",
            "AMD", "AVGO", "MU", "AMAT", "MRVL", "ARM", "TSM",
            "ANET", "CRDO", "VRT", "GEV", "CEG", "VST", "NBIS", "APLD", "CRWV",
            "UBER", "ABNB", "BKNG", "ISRG", "DXCM", "HIMS", "VEEV",
            "INTU", "ADBE", "RKLB", "KTOS", "AXON", "COIN", "SOFI", "MELI",
            "SPY", "RSP", "SPHB", "SPLV",
        ]
    return result


def check_watchlist_triggers(
    watchlist: dict,
    prices: dict,   # {ticker: current_price}
    webhook: str,
    state: dict,
) -> list[str]:
    """
    Check each watchlist entry against current price.
    Returns list of triggered ticker strings.
    Fires discord alerts for hits.
    """
    triggered = []
    for ticker, cfg in watchlist.items():
        price = prices.get(ticker)
        if price is None:
            continue
        buy_at = cfg.get("buy_at")
        direction = cfg.get("direction", "below")
        if buy_at is None:
            continue
        hit = False
        if direction == "below" and price <= buy_at:
            hit = True
        elif direction == "above" and price >= buy_at:
            hit = True
        if hit:
            triggered.append(ticker)
            if webhook and _MODULES_AVAILABLE:
                try:
                    alert_watch_trigger(
                        ticker, price, buy_at,
                        cfg.get("note", ""),
                        webhook, state
                    )
                except Exception:
                    pass
    return triggered


def check_big_moves(
    held: list[str],
    ticker_data: dict,    # {ticker: {pct_today: float}}
    threshold: float,
    webhook: str,
    state: dict,
) -> list[str]:
    """Check held tickers for ±threshold% moves. Fire discord alerts. Return triggered list."""
    triggered = []
    for ticker in held:
        data = ticker_data.get(ticker, {})
        pct = data.get("pct_today", 0.0)
        if abs(pct) >= threshold:
            triggered.append(ticker)
            if webhook and _MODULES_AVAILABLE:
                try:
                    alert_big_move(ticker, pct, webhook, state)
                except Exception:
                    pass
    return triggered


def write_gha_summary(
    state_score: dict,
    screen_results: dict,
    watchlist_triggers: list,
    big_moves: list,
    mode: str,
) -> None:
    """Write markdown to $GITHUB_STEP_SUMMARY env var path if set."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not summary_path:
        return

    lines = []
    lines.append(f"# Market Scanner — {mode} ({date.today()})\n")

    # Market state summary
    regime = state_score.get("regime", "N/A")
    score = state_score.get("score", "N/A")
    summary_text = state_score.get("summary", "")
    lines.append(f"## Market State\n")
    lines.append(f"**Regime:** {regime} | **Score:** {score}\n")
    if summary_text:
        lines.append(f"{summary_text}\n")

    # Watchlist triggers
    if watchlist_triggers:
        lines.append(f"\n## Watchlist Triggers\n")
        for t in watchlist_triggers:
            lines.append(f"- **{t}** — price target hit\n")

    # Big moves
    if big_moves:
        lines.append(f"\n## Big Moves (Held)\n")
        for t in big_moves:
            lines.append(f"- **{t}**\n")

    # Screener results
    if screen_results:
        lines.append(f"\n## Screener Results\n")
        for screen_id, candidates in screen_results.items():
            if candidates:
                lines.append(f"\n### Screen {screen_id}\n")
                lines.append("| Ticker | Reason |\n|--------|--------|\n")
                for cand in candidates[:5]:
                    ticker = cand.get("ticker", "")
                    reason = cand.get("reason", "")
                    lines.append(f"| {ticker} | {reason} |\n")

    with open(summary_path, "a") as f:
        f.writelines(lines)


def _select_deep_dive_candidates(
    histories: dict,
    held: list[str],
    n: int = 150,
) -> list[str]:
    """
    Pick top N candidates for deep-dive fundamental fetch.
    Priority: held tickers first, then tickers with recent momentum
    (price within 20% of 52w high and mcap heuristic from history).
    """
    result = list(held)
    # Add names within 20% of 52-week high (momentum candidates)
    for ticker, df in histories.items():
        if ticker in result or len(df) < 50:
            continue
        hi52 = df["Close"].rolling(252, min_periods=50).max().iloc[-1]
        curr = df["Close"].iloc[-1]
        if hi52 > 0 and (hi52 - curr) / hi52 <= 0.20:
            result.append(ticker)
        if len(result) >= n:
            break
    return result[:n]


def _compute_gap_movers(
    histories: dict,
    threshold: float = 5.0,
) -> list[dict]:
    """Find tickers that gapped > threshold% from prior close."""
    movers = []
    for ticker, df in histories.items():
        if len(df) < 2:
            continue
        prev = float(df["Close"].iloc[-2])
        curr = float(df["Close"].iloc[-1])
        if prev <= 0:
            continue
        pct = (curr - prev) / prev * 100
        if abs(pct) >= threshold:
            movers.append({"ticker": ticker, "pct_change": pct, "price": curr})
    return sorted(movers, key=lambda x: abs(x["pct_change"]), reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["daily-pre", "daily-post", "weekly"],
                        default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Discord, skip Pages deploy, print report path only")
    args = parser.parse_args()

    # Determine mode from arg or env (SCANNER_MODE env var) or day of week
    mode = args.mode or os.environ.get("SCANNER_MODE", "")
    if not mode:
        dow = date.today().weekday()  # 0=Monday
        mode = "weekly" if dow == 0 else "daily-post"

    cfg = load_config()
    webhook = cfg["discord_webhook"] if not args.dry_run else ""

    print(f"[scanner] mode={mode} date={date.today()}", flush=True)

    if not _MODULES_AVAILABLE:
        print(f"[scanner] WARNING: sub-modules not available ({_IMPORT_ERR}). "
              f"Running in legacy fallback mode.", flush=True)
        _run_legacy(cfg, mode, args.dry_run, webhook)
        return

    state = load_state()
    cache = RunCache()

    # ── Universe data ──────────────────────────────────────────────────────────
    tickers = get_universe_tickers()
    # Add sector ETFs, SPY/RSP/SPHB/SPLV for theme + sector rotation analysis,
    # plus watchlist tickers so we can fetch current prices for the report.
    SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY",
                   "XLU", "XLB", "XLRE", "XLC"]
    MARKET_ETFS = ["SPY", "RSP", "SPHB", "SPLV", "QQQ", "IWM"]
    watch_tickers = list(cfg["watchlist"].keys())
    extra_tickers = SECTOR_ETFS + MARKET_ETFS + watch_tickers
    universe_set = set(tickers)
    for t in extra_tickers:
        if t not in universe_set:
            tickers.append(t)
            universe_set.add(t)
    print(f"[scanner] universe: {len(tickers)} tickers "
          f"(+{len(SECTOR_ETFS)} sector ETFs, +{len(MARKET_ETFS)} market ETFs, "
          f"+{len(watch_tickers)} watchlist)", flush=True)

    # Bulk download — all tickers, 1 year history
    print("[scanner] fetching universe history...", flush=True)
    universe_histories = bulk_history(tickers, period="1y")
    print(f"[scanner] got history for {len(universe_histories)} tickers", flush=True)

    # Index / cross-asset data
    index_syms = ["^GSPC", "^IXIC", "^VIX", "^VIX3M", "^VIX9D", "^VVIX",
                  "^TNX", "GC=F", "HG=F", "SPY", "RSP", "SPHB", "SPLV"]
    index_data = get_index_data(index_syms)

    # Macro data (FRED + AAII)
    fred_data = latest_fred()
    aaii_data = latest_aaii()

    # ── Market state ───────────────────────────────────────────────────────────
    print("[scanner] computing market state...", flush=True)
    state_score = compute_market_state(universe_histories, index_data, fred_data, aaii_data)
    print(f"[scanner] state score: {state_score.get('score', 'N/A')} ({state_score.get('regime', '')})", flush=True)

    # Regime shift alert
    prev_score = state.get("last_state_score")
    if prev_score is not None and abs(state_score.get("score", 50) - prev_score) >= 15:
        alert_regime_shift(
            state_score.get("summary", ""),
            prev_score, state_score.get("score", 50),
            webhook, state
        )
    state["last_state_score"] = state_score.get("score")

    # ── Watchlist & big moves ──────────────────────────────────────────────────
    # Get current prices for watchlist tickers
    watch_tickers = list(cfg["watchlist"].keys())
    watch_prices = {}
    for t in watch_tickers:
        h = universe_histories.get(t)
        if h is not None and not h.empty:
            watch_prices[t] = float(h["Close"].iloc[-1])

    watchlist_triggers = check_watchlist_triggers(
        cfg["watchlist"], watch_prices, webhook, state)

    # Check held tickers for big moves (using pct_today from universe histories)
    held_data = {}
    for t in cfg["held"]:
        h = universe_histories.get(t)
        if h is not None and len(h) >= 2:
            prev = float(h["Close"].iloc[-2])
            curr = float(h["Close"].iloc[-1])
            held_data[t] = {"pct_today": (curr - prev) / prev * 100 if prev else 0}

    big_moves = check_big_moves(
        cfg["held"], held_data, cfg["alert_move_pct"], webhook, state)

    # ── Screens (always) ──────────────────────────────────────────────────────
    print("[scanner] running candidate screens...", flush=True)
    # Deep-dive fundamentals for top candidates by momentum
    # To avoid 1000+ ticker.info calls, pre-filter: names near highs or with recent history
    deep_dive_tickers = _select_deep_dive_candidates(universe_histories, cfg["held"], n=150)
    ticker_infos = {}
    for i, t in enumerate(deep_dive_tickers):
        if i % 25 == 0:
            print(f"[scanner] deep dive {i}/{len(deep_dive_tickers)}...", flush=True)
        info = get_ticker_info(t)
        if info:
            ticker_infos[t] = info

    screen_results = run_all_screens(
        universe_histories, ticker_infos, set(cfg["held"]))

    # Discord alerts for new top candidates (Screen 1, 2, 3, 4 only for alerts)
    for screen_id in [1, 2, 3, 4]:
        for cand in screen_results.get(screen_id, [])[:3]:
            alert_screener_candidate(
                cand["ticker"],
                SCREEN_META[screen_id]["name"],
                cand.get("reason", ""),
                webhook, state
            )

    # ── Weekly-only: themes + clustering ─────────────────────────────────────
    ranked_themes, emerging_clusters, sector_rotation = [], [], []
    if mode == "weekly":
        print("[scanner] analyzing themes...", flush=True)
        theme_analysis = analyze_themes(universe_histories)
        ranked_themes = theme_analysis.get("ranked_themes", [])
        emerging_clusters = theme_analysis.get("emerging_clusters", [])
        sector_rotation = theme_analysis.get("sector_rotation", [])
        stovall_phase = theme_analysis.get("stovall_phase", "")

        for cluster in emerging_clusters:
            alert_emerging_cluster(
                cluster.get("members", []),
                cluster.get("label", ""),
                webhook, state
            )
    else:
        # Daily still gets sector rotation for the short report
        from analytics.themes import compute_sector_rotation
        spy_hist = universe_histories.get("SPY", pd.DataFrame())
        if not spy_hist.empty:
            sector_rotation = compute_sector_rotation(universe_histories, spy_hist)

    # ── Breadth summary ────────────────────────────────────────────────────────
    breadth = state_score.get("breadth", {})
    new_highs_lows = {
        "new_highs_52w": breadth.get("new_highs_52w", 0),
        "new_lows_52w": breadth.get("new_lows_52w", 0),
    }

    # ── Gap movers (for daily report) ─────────────────────────────────────────
    gap_movers = _compute_gap_movers(universe_histories, threshold=5.0)

    # ── Build HTML report ─────────────────────────────────────────────────────
    print("[scanner] building report...", flush=True)
    if mode == "weekly":
        html = build_weekly_report(
            state_score=state_score,
            sector_rotation=sector_rotation,
            ranked_themes=ranked_themes,
            emerging_clusters=emerging_clusters,
            screen_results=screen_results,
            screen_meta=SCREEN_META,
            held_tickers=set(cfg["held"]),
            watchlist=cfg["watchlist"],
            watch_prices=watch_prices,
            pre_ipo=cfg["pre_ipo"],
            new_highs_lows=new_highs_lows,
        )
    else:
        html = build_daily_report(
            state_score=state_score,
            sector_rotation=sector_rotation,
            new_highs_lows=new_highs_lows,
            gap_movers=gap_movers[:10],
            new_candidates=[c for cs in screen_results.values() for c in cs[:2]],
            earnings_reactions=[],
        )

    # Write legacy report.html at root (for GHA artifact)
    Path("report.html").write_text(html)
    print("[scanner] wrote report.html", flush=True)

    # Stage for Pages
    if not args.dry_run:
        report_type = "weekly" if mode == "weekly" else "daily"
        staged = stage_report(html, report_type)
        update_index(report_type)
        print(f"[scanner] staged to {staged}", flush=True)

    # GHA summary
    write_gha_summary(state_score, screen_results, watchlist_triggers, big_moves, mode)

    # Save state
    save_state(state)
    print("[scanner] done.", flush=True)


def _run_legacy(cfg: dict, mode: str, dry_run: bool, webhook: str) -> None:
    """
    Minimal fallback runner used when sub-modules are not yet present.
    Fetches pulse + portfolio data via yfinance directly, writes a basic report.html.
    """
    try:
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        print("[scanner] yfinance not installed — cannot run legacy mode.", flush=True)
        return

    state: dict = {}
    state_path = Path("state.json")
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}

    # Fetch market pulse
    pulse_syms = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "VIX": "^VIX", "10Y Yield": "^TNX"}
    pulse = {}
    for name, sym in pulse_syms.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                curr = float(hist["Close"].iloc[-1])
                pct = (curr - prev) / prev * 100 if prev else 0
                pulse[name] = {"value": curr, "pct": pct}
        except Exception:
            pulse[name] = {"value": 0, "pct": 0}

    # Fetch held tickers
    held = cfg.get("held", [])
    held_rows = []
    for t in held:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="5d")
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                curr = float(hist["Close"].iloc[-1])
                pct = (curr - prev) / prev * 100 if prev else 0
                held_rows.append({"ticker": t, "price": curr, "pct_today": pct})
        except Exception:
            held_rows.append({"ticker": t, "price": 0, "pct_today": 0})

    # Check watchlist
    watchlist = cfg.get("watchlist", {})
    watch_prices = {}
    for t in watchlist:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="5d")
            if not hist.empty:
                watch_prices[t] = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    state_score: dict = {"score": 50, "regime": "UNKNOWN", "summary": "Legacy mode — sub-modules not available"}
    watchlist_triggers = check_watchlist_triggers(watchlist, watch_prices, webhook, state)

    held_data = {r["ticker"]: {"pct_today": r["pct_today"]} for r in held_rows}
    big_moves = check_big_moves(held, held_data, cfg.get("alert_move_pct", 5.0), webhook, state)

    # Build minimal HTML
    rows_html = ""
    for r in held_rows:
        color = "#4caf50" if r["pct_today"] >= 0 else "#f44336"
        rows_html += (
            f"<tr><td>{r['ticker']}</td>"
            f"<td>${r['price']:.2f}</td>"
            f"<td style='color:{color}'>{r['pct_today']:+.2f}%</td></tr>\n"
        )

    pulse_html = ""
    for name, d in pulse.items():
        color = "#4caf50" if d["pct"] >= 0 else "#f44336"
        pulse_html += (
            f"<div style='display:inline-block;margin:8px 16px;'>"
            f"<b>{name}</b><br/>{d['value']:.2f} "
            f"<span style='color:{color}'>{d['pct']:+.2f}%</span></div>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset='utf-8'><title>Market Scanner — {date.today()}</title>
<style>body{{background:#0d1117;color:#e6edf3;font-family:monospace;padding:24px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #30363d;padding:8px;text-align:left}}
th{{background:#161b22}}</style></head>
<body>
<h1>Market Scanner — {date.today()} ({mode})</h1>
<p style='color:#f0a500'>Note: running in legacy mode (sub-modules not installed)</p>
<h2>Market Pulse</h2>
<div>{pulse_html}</div>
<h2>Portfolio</h2>
<table><tr><th>Ticker</th><th>Price</th><th>Today %</th></tr>
{rows_html}</table>
</body></html>"""

    Path("report.html").write_text(html)
    print("[scanner] wrote report.html (legacy mode)", flush=True)

    write_gha_summary(state_score, {}, watchlist_triggers, big_moves, mode)

    state_path.write_text(json.dumps(state, indent=2, default=str))
    print("[scanner] done (legacy mode).", flush=True)


if __name__ == "__main__":
    main()
