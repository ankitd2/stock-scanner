# Market Scanner

Free, automated market scanner. No AI APIs. No paid services.

## What you get

- **Daily (9:30 AM ET, weekdays):** price check against your watch targets
- **Monday (4 PM ET):** full screener across 60+ tickers + portfolio red-flag scan
- **Any time a watch price is hit:** Discord alert (if configured)
- **Always:** full HTML report saved as GitHub Actions artifact

## Setup — 10 minutes

### 1. Add this to your existing repo (or new one)
Copy all files in as-is.

### 2. Edit `config.json`
Update your portfolio tickers, watchlist targets, and optional Discord webhook.

### 3. Optional: Discord alerts
Create a webhook (Discord channel → Integrations → Webhooks → copy URL).
Add as GitHub secret `DISCORD_WEBHOOK`.

### 4. Enable Actions
Actions tab → enable workflows if prompted. Done.

### 5. Test manually
Actions → Market Scanner → Run workflow → check the artifact for `report.html`.

## Viewing the report
Actions tab → click any run → scroll to Artifacts → download `market-report-*` → open `report.html` in browser.

## Config reference

| Key | Default | Description |
|---|---|---|
| `portfolio` | — | Tickers to track daily |
| `watchlist` | — | Stocks with entry targets |
| `alert_move_pct` | 5.0 | Alert if any holding moves ±this% in a day |
| `weekly_scan_weekday` | 0 | 0=Monday, 4=Friday |
| `run_screener_daily` | false | Run screener every day (slower, more API calls) |
| `discord_webhook` | "" | Paste URL or set as GitHub secret |

## Cost: $0
- yfinance = free (Yahoo Finance data)
- GitHub Actions = free tier (2,000 min/month, scanner uses ~5 min/week)
- Discord webhook = free
- No paid data subscriptions, no AI APIs

---
*Not financial advice. Data via yfinance / Yahoo Finance.*
