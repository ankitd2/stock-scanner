"""
Plain-English explanations for every signal, indicator, and screen used in the
market intelligence scanner. These appear as collapsible <details> sections in
the HTML report so non-finance readers can understand what they are looking at.
"""

EXPLAINERS: dict[str, str] = {
    # ── Screens ───────────────────────────────────────────────────────────────
    "52wH_proximity": (
        "This screen looks for stocks trading within 10–40% of their 52-week high. "
        "A stock near a 52-week high is showing strength — buyers are in control — "
        "but if it is too close (within 10%), the easy gains may already be taken. "
        "The sweet spot is a stock that has pulled back enough to offer a better "
        "entry price while still being in an uptrend. Stocks far below (more than "
        "40% off) are usually in serious trouble and need more investigation before "
        "buying."
    ),
    "quality_pullback": (
        "A quality pullback screen finds strong, fundamentally sound companies "
        "whose share price has temporarily dropped — for example after a market "
        "sell-off or an overreaction to minor news. The idea is to buy a high-quality "
        "business at a temporary discount rather than chasing it at peak prices. "
        "Candidates must have growing revenue, strong analyst consensus, and a "
        "healthy balance sheet — the drop is the opportunity, not a warning sign."
    ),
    "risk_adj_momentum": (
        "Risk-adjusted momentum ranks stocks by how much they have gone up relative "
        "to how volatile (choppy) that rise has been. A stock that climbs steadily "
        "scores higher than one that spikes and crashes, even if the raw gain is the "
        "same. This matters because smooth momentum tends to persist — erratic moves "
        "often reverse. Stocks in the top quintile of risk-adjusted momentum have "
        "historically outperformed over 3–12 month horizons."
    ),
    "quality_momentum": (
        "Quality momentum combines two proven investment factors: quality (profitable, "
        "growing businesses with strong balance sheets) and momentum (stocks that have "
        "recently outperformed). Either factor alone works well; together they have "
        "historically produced the best risk-adjusted returns. Think of it as "
        "buying the best companies while they are already going up — not trying to "
        "catch falling knives or overpaying for mediocre businesses."
    ),
    "pead": (
        "PEAD stands for Post-Earnings Announcement Drift. Academic research going "
        "back to the 1960s shows that stocks that beat earnings expectations tend to "
        "keep drifting upward for weeks or months after the announcement — markets are "
        "slow to fully price in good news. This screen identifies recent positive "
        "earnings surprises where the initial reaction looks like just the beginning "
        "of a longer move. A high revenue beat combined with raised guidance is the "
        "strongest signal."
    ),
    "analyst_revision": (
        "When Wall Street analysts upgrade their price targets or move a stock from "
        "Hold to Buy, it often precedes a multi-week rally. This screen tracks "
        "positive analyst estimate revisions — upward changes to revenue or earnings "
        "forecasts — as a leading indicator. Multiple analysts revising estimates "
        "higher in the same direction is stronger than a single upgrade. The signal "
        "is most powerful when combined with recent earnings beats."
    ),
    "insider_buys": (
        "Corporate insiders (executives and board members) must publicly report when "
        "they buy or sell their own company's stock. While insider selling can happen "
        "for many reasons (taxes, diversification), insider buying is almost always a "
        "sign that management believes the stock is undervalued. This screen flags "
        "clusters of open-market purchases — insiders spending their own money, not "
        "just exercising options — within the past 90 days."
    ),
    "quality_oversold": (
        "This screen finds high-quality businesses that have been heavily sold off — "
        "typically RSI below 35 — often due to broader market fear rather than "
        "company-specific problems. When a fundamentally strong stock gets caught in "
        "a market downdraft and becomes technically oversold, it creates a mean-"
        "reversion opportunity. The key filter is quality: a cheap stock is only "
        "interesting if the underlying business is healthy and growing."
    ),

    # ── Market state indicators ────────────────────────────────────────────────
    "market_state_score": (
        "The market state score is a single 0–100 number that summarises the current "
        "health of the broad market by combining multiple independent signals: "
        "volatility (VIX), credit spreads, breadth, sentiment surveys, and sector "
        "leadership. A score above 70 is 'risk-on' — conditions favor owning growth "
        "stocks aggressively. Below 35 is 'risk-off' — defensive positioning or cash "
        "is appropriate. The score helps size positions: go bigger in good markets, "
        "smaller in bad ones."
    ),
    "vix": (
        "VIX is the CBOE Volatility Index, often called the 'fear gauge'. It measures "
        "how much the options market expects the S&P 500 to move over the next 30 days. "
        "A VIX below 15 signals calm, low-fear markets where investors are comfortable "
        "taking risk. Above 25 signals elevated fear and potential turbulence. Above 40 "
        "is crisis territory — but historically these spikes have also been the best "
        "buying opportunities, since fear peaks tend to coincide with market bottoms."
    ),
    "vix_term_structure": (
        "The VIX term structure compares short-term fear (VIX9D, 9-day) to longer-term "
        "fear (VIX3M, 3-month). In normal markets the long-term VIX is higher than the "
        "short-term VIX — this is called 'contango'. When the short-term VIX spikes "
        "above the long-term VIX ('backwardation'), it signals acute stress or a "
        "specific near-term fear event. A return from backwardation to contango often "
        "marks the end of a market pullback and a resumption of the uptrend."
    ),
    "hy_oas": (
        "HY OAS stands for High-Yield Option-Adjusted Spread — the extra interest rate "
        "that risky ('junk') corporate bonds pay over safe US Treasury bonds. When this "
        "spread widens (gets bigger), credit markets are worried about defaults and "
        "recession, which is bad for stocks. When it tightens (gets smaller), credit "
        "markets are relaxed, which is good for risk assets. Think of it as the credit "
        "market's confidence thermometer: tight spreads = confidence, wide spreads = "
        "worry."
    ),
    "rsp_spy": (
        "RSP/SPY ratio compares the equal-weighted S&P 500 ETF (RSP, where every stock "
        "counts equally) to the market-cap-weighted S&P 500 ETF (SPY, where Apple and "
        "Microsoft dominate). When RSP outperforms SPY, the average stock is doing well "
        "— the rally is 'broad'. When SPY outperforms RSP, only a handful of mega-caps "
        "are dragging the index higher while most stocks lag. Broad participation is "
        "healthier and more sustainable."
    ),
    "sphb_splv": (
        "SPHB/SPLV ratio compares high-beta stocks (those that move more than the "
        "market, like tech and growth) to low-volatility stocks (utilities, consumer "
        "staples). When this ratio rises, investors are reaching for risk — a bullish "
        "sign. When it falls, money is rotating into defensive stocks — a sign of "
        "caution or fear. This ratio is sometimes called the 'risk appetite' gauge for "
        "the equity market."
    ),
    "aaii": (
        "The AAII Sentiment Survey polls individual investors weekly: are you bullish, "
        "bearish, or neutral for the next 6 months? It is a classic contrarian "
        "indicator: when retail investors are extremely bullish, markets often struggle "
        "(everyone who wants to buy has already bought). When sentiment is extremely "
        "bearish, it often marks a bottom. The historical average is about 38% bulls. "
        "Above 55% bulls or below 20% bulls are the contrarian extremes to watch."
    ),
    "breadth_50dma": (
        "This measures what percentage of S&P 500 stocks are trading above their "
        "50-day moving average (a short-term trend indicator). When more than 70% of "
        "stocks are above their 50 DMA, the market has broad momentum — healthy. "
        "When fewer than 30% are above it, most stocks are in short-term downtrends — "
        "a warning. This matters because index-level gains can mask underlying weakness "
        "if only a few large-cap stocks are holding things up."
    ),
    "breadth_200dma": (
        "Similar to the 50 DMA breadth, but using the 200-day moving average, which "
        "represents the long-term trend. A stock above its 200 DMA is considered in a "
        "long-term uptrend; below it is in a downtrend. When 70%+ of S&P 500 stocks "
        "are above their 200 DMA, the market is broadly healthy. This is a slower-"
        "moving, more reliable indicator than the 50 DMA version and rarely gives "
        "false signals in either direction."
    ),

    # ── Themes ────────────────────────────────────────────────────────────────
    "theme_strength": (
        "Theme strength measures how well a group of related stocks (e.g. 'AI "
        "Infrastructure', 'Cybersecurity', 'GLP-1 Drugs') are performing together "
        "relative to the broad market. A high theme score means the theme's stocks "
        "are outperforming as a group — institutional money is flowing into the sector. "
        "A falling theme score means rotation is happening out of it. Tracking themes "
        "helps identify where the next big moves are likely to come from before they "
        "become obvious."
    ),
    "sector_rotation": (
        "Sector rotation is the process by which investment capital moves from one "
        "industry sector to another over the economic cycle. For example, early in a "
        "recovery, Financials and Consumer Discretionary tend to lead; in a slowdown, "
        "Healthcare and Utilities hold up better. This scanner tracks relative "
        "strength across all 11 GICS sectors (Technology, Healthcare, Energy, etc.) "
        "and ranks them to show which sectors are gaining institutional momentum right "
        "now — the ones to focus new money on."
    ),
    "emerging_clusters": (
        "An emerging cluster is a group of small or mid-cap stocks in a specific niche "
        "that have started outperforming the broader market in recent weeks, but have "
        "not yet attracted mainstream attention. These are often early signals of a new "
        "theme gaining traction — for example, quantum computing stocks or a specific "
        "regional energy play. Identifying clusters early allows positioning before "
        "large institutions discover and bid up the names."
    ),

    # ── General indicators ────────────────────────────────────────────────────
    "rsi": (
        "RSI stands for Relative Strength Index, a momentum indicator on a 0–100 scale. "
        "It measures how fast a stock has been moving up or down over the past 14 days. "
        "Above 70 is considered overbought — the stock has moved up very quickly and "
        "may be due for a pause or pullback. Below 30 is considered oversold — it has "
        "dropped quickly and may bounce. RSI is most useful as a timing tool: avoid "
        "buying above 75 and consider adding to quality names when RSI dips below 35."
    ),
    "50dma": (
        "The 50-day moving average (50 DMA) is the average closing price of a stock "
        "over the past 50 trading days — roughly 10 weeks. It acts as a dynamic support "
        "or resistance level. When a stock is above its 50 DMA, it is in a short-to-"
        "medium-term uptrend; below it suggests weakness. A pullback to the 50 DMA in "
        "an uptrending stock is often a buying opportunity. Breaking below the 50 DMA "
        "with high volume is a warning to be cautious."
    ),
    "200dma": (
        "The 200-day moving average (200 DMA) is the average closing price over the "
        "past 200 trading days — roughly 40 weeks. It is the key dividing line between "
        "bull and bear markets for individual stocks. Most professional investors "
        "require a stock to be above its 200 DMA before initiating a long position. "
        "A stock that loses the 200 DMA often experiences significant further selling "
        "as momentum and trend-following algorithms sell automatically."
    ),
    "golden_cross": (
        "A golden cross occurs when a stock's 50-day moving average rises above its "
        "200-day moving average. It signals that recent momentum has shifted decisively "
        "to the upside and the short-term trend is now stronger than the long-term "
        "trend — often interpreted as a medium-term bullish signal. The opposite is a "
        "'death cross' (50 DMA crossing below 200 DMA), which signals the start of a "
        "potential longer-term downtrend. Golden crosses have historically preceded "
        "above-average 6–12 month returns."
    ),
    "volume_ratio": (
        "Volume ratio compares today's trading volume to the stock's average daily "
        "volume over the past 20 days. A ratio above 1.5x means unusually high "
        "interest — perhaps from institutional buyers, news, or a breakout. Volume "
        "above average on a up day (accumulation) is bullish; above average on a down "
        "day (distribution) is bearish. Price moves on low volume are less trustworthy "
        "because they can reverse easily; moves on high volume tend to stick."
    ),
}


def explainer_html(key: str, title: str = None) -> str:
    """
    Return a <details><summary> collapsible block for the given key,
    or an empty string if the key is not found in EXPLAINERS.

    Args:
        key:   Key into the EXPLAINERS dict.
        title: Optional display title. Defaults to a title-cased version of the key.

    Returns:
        HTML string or '' if key is missing.
    """
    text = EXPLAINERS.get(key, "")
    if not text:
        return ""
    display = title if title else key.replace("_", " ").title()
    return (
        f'<details class="explainer">'
        f'<summary>&#128269; What is {display}?</summary>'
        f"<p>{text}</p>"
        f"</details>"
    )


def glossary_html() -> str:
    """
    Returns a full glossary section HTML with all terms from EXPLAINERS,
    sorted alphabetically by key.
    """
    items = sorted(EXPLAINERS.items(), key=lambda kv: kv[0].lower())
    rows = ""
    for key, text in items:
        display = key.replace("_", " ").title()
        rows += (
            f'<div class="glossary-term">'
            f'<dt><strong>{display}</strong></dt>'
            f"<dd>{text}</dd>"
            f"</div>"
        )
    return (
        f'<section class="glossary">'
        f'<h2 class="sec-title">Glossary</h2>'
        f"<dl>{rows}</dl>"
        f"</section>"
    )
