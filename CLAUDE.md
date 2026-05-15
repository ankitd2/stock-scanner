# Claude Code — Stock Scanner Setup

## Your one job
Create the files listed below exactly as shown, then push to `origin/main`.

```bash
git add -A
git commit -m "Add free market scanner — yfinance, matplotlib, HTML reports, no AI APIs"
git push origin main
```

---

## File 1 of 4 — `requirements.txt`

```
yfinance>=0.2.40
pandas>=2.0.0
matplotlib>=3.8.0
requests>=2.32.0
```

---

## File 2 of 4 — `config.json`

```json
{
  "portfolio": [
    "META", "NVDA", "VTI", "VRT", "MU",
    "MSFT", "AMZN", "CRWD", "ANET", "AAPL", "APP"
  ],

  "watchlist": {
    "GOOGL": {
      "buy_at": 375,
      "direction": "below",
      "note": "Cloud +63%, I/O catalysts priced in — wait for 10% pullback from $403 high"
    },
    "RKLB": {
      "buy_at": 95,
      "direction": "below",
      "note": "Above all analyst PTs — Neutron launch H2 2026 is the real re-rating event"
    },
    "NBIS": {
      "buy_at": 185,
      "direction": "below",
      "note": "+16% earnings pop — wait for consolidation"
    },
    "CRDO": {
      "buy_at": 155,
      "direction": "below",
      "note": "Only 3% PT upside at current price"
    }
  },

  "alert_move_pct": 5.0,
  "weekly_scan_weekday": 0,
  "run_screener_daily": false,
  "discord_webhook": ""
}
```

---

## File 3 of 4 — `.github/workflows/scanner.yml`

```yaml
name: Market Scanner

on:
  schedule:
    - cron: "30 14 * * 1-5"
    - cron: "0 21 * * 1"
  workflow_dispatch:

jobs:
  scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run scanner
        env:
          DISCORD_WEBHOOK: ${{ secrets.DISCORD_WEBHOOK }}
        run: python scanner.py

      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: market-report-${{ github.run_id }}
          path: report.html
          retention-days: 30
        if: always()
```

---

## File 4 of 4 — `scanner.py`

Copy the content of `scanner.py` from the same directory as this CLAUDE.md file.
It is a 750-line Python script. Do not summarise or rewrite it — copy it verbatim.

If scanner.py is not present alongside this file, implement it from the spec below.

### scanner.py spec (use only if the file is missing)

**Purpose:** free retail stock scanner, no AI APIs, no paid services.

**Inputs:** `config.json` (portfolio list, watchlist with `buy_at` targets, `alert_move_pct`, `weekly_scan_weekday`)

**Outputs:**
1. `report.html` — full newsletter-style HTML report with embedded base64 matplotlib charts
2. GitHub Actions job summary written to `$GITHUB_STEP_SUMMARY`
3. Discord webhook POST (urgent alerts only — watch price hit or screener BUY found)

**Key sections to implement:**

`get_market_pulse()` — fetch S&P 500 (^GSPC), Nasdaq (^IXIC), VIX (^VIX), 10Y yield (^TNX) via yfinance. Return dict with value and pct change.

`get_stock(ticker)` — fetch one ticker via yfinance. Return dict with: price, hi52, lo52, from_hi (% below 52wk high), pct_today, pct_ytd, tgt_mean, tgt_high, tgt_low, upside (% to mean PT), n_analysts, buy/hold/sell counts (parsed from t.recommendations last 30 rows), buy_pct, rev_growth (from info["revenueGrowth"] * 100), gross_margin, op_margin, pe_trailing, pe_forward, peg, mcap, earnings_dt (from t.calendar), rsi (14-day), news (list of {title, url} from t.news[:3]).

`calc_rsi(prices, period=14)` — standard RSI calculation.

`skill_verdict(s)` — apply 6 growth-mode criteria, return verdict BUY/WATCH/WAIT/PASS:
- Rev growth > 20% (from rev_growth field)
- Buy consensus >= 80% (buy_pct)
- Sell ratings <= 1 (hard kill at 2+)
- Price 10-40% below 52wk high (from_hi)
- PT upside >= 15% (upside); hard kill if price > tgt_high
- Market cap > $2B
- RSI > 78 AND from_hi < 5 = additional hard kill
Return dict: {verdict, passes, fails, kills, score}

`run_screener(portfolio, limit=10)` — iterate UNIVERSE list (60 AI/tech/growth tickers), skip portfolio holdings, call get_stock + skill_verdict on each, return top results sorted BUY first then by score.

UNIVERSE list includes (at minimum): MSFT, GOOGL, AMZN, META, NVDA, TSLA, CRM, NOW, SNOW, DDOG, NET, ZS, CRWD, PANW, PLTR, APP, TTD, HUBS, SHOP, RDDT, NFLX, AMD, AVGO, MU, AMAT, MRVL, ARM, TSM, ANET, CRDO, VRT, GEV, CEG, VST, NBIS, APLD, CRWV, UBER, ABNB, BKNG, ISRG, DXCM, HIMS, VEEV, INTU, ADBE, RKLB, KTOS, AXON, COIN, SOFI, MELI.

`get_earnings_calendar(tickers)` — use t.calendar to find earnings dates in next 45 days. Return list of {ticker, date} sorted by date, deduped.

**Three charts (matplotlib → base64 PNG → embedded in HTML):**
1. `chart_range(stocks)` — horizontal bars showing 52-week range with current price marker. Dark background (#0d1117).
2. `chart_watchlist(watchlist_cfg, prices)` — progress bars showing proximity to watch targets. Green when within 5%.
3. `chart_alloc(portfolio_data)` — donut chart of holdings by price weight.

**HTML report structure (dark theme, #0d1117 background):**
- Header with date, "Weekly Market Brief" label
- Market pulse strip (S&P, Nasdaq, VIX, 10Y) with color-coded % change
- 52-week range chart image
- Portfolio table: ticker, name, price, today%, PT, upside, consensus, rev growth, RSI, earnings badge
- Watch list section: chart + individual cards showing current price vs target, proximity %, note from config
- Screener results: cards with 4-stat grid (price, PT upside, rev growth, % below high) + criteria checklist
- Earnings calendar table (next 45 days)
- Allocation donut chart

**GitHub Actions summary:** write markdown to `$GITHUB_STEP_SUMMARY` env path. Include pulse table, triggered watches, screener BUYs, portfolio table, earnings.

**Discord:** POST to `DISCORD_WEBHOOK` env var only if watch price triggered or screener BUY found. Keep under 2000 chars.

**main() logic:**
1. Daily: fetch pulse, portfolio, watchlist prices, check triggers, check big moves (alert_move_pct)
2. Weekly (weekday == weekly_scan_weekday): run screener + earnings calendar
3. Always: generate charts + HTML report + GHA summary
4. If alerts: send Discord

---

## Context (why this repo exists)

Built for a growth-oriented retail investor. Current holdings, watch prices, and pre-IPO list live in `config.json`.

Screener applies a 6-criterion growth-mode framework: rev growth >20%, buy consensus ≥80%, ≤1 sell rating, price 10-40% below 52wk high, PT upside ≥15%, market cap >$2B.

