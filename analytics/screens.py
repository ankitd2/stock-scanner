"""
analytics/screens.py — 8 academically-validated candidate screens.

Screens implemented:
  1. 52wH Proximity       (George-Hwang 2004)
  2. Quality Pullback     (Mean reversion within uptrend)
  3. Risk-Adjusted Momentum (Barroso-Santa-Clara 2015)
  4. Quality-Momentum     (AQR Quality-Minus-Junk + momentum)
  5. Post-Earnings Drift  (Ball-Brown + FRBSF 2024)
  6. Analyst Revision Momentum (Stickel-Womack)
  7. Insider Cluster Buys (Lakonishok-Lee 2001)
  8. Quality Oversold     (Connors RSI-2, quality-gated)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from typing import Optional

try:
    import pandas as pd
    import numpy as np
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from analytics.indicators import compute_all


def _scalar(v):
    """Extract latest scalar from a pandas Series, or return as-is if already scalar."""
    if v is None:
        return None
    if hasattr(v, "iloc"):
        try:
            v = v.iloc[-1]
        except Exception:
            return None
    if v is None:
        return None
    try:
        import math
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Screen metadata
# ---------------------------------------------------------------------------

SCREEN_META = {
    1: {
        "name": "52wH Proximity",
        "description": (
            "Names near 52-week highs on elevated volume — "
            "momentum continuation signal (George-Hwang 2004)"
        ),
        "evidence": (
            "George-Hwang (2004), Journal of Finance. Proximity to 52wH is one "
            "of the most robust momentum anomalies."
        ),
        "hold_period": "1-3 months",
    },
    2: {
        "name": "Quality Pullback",
        "description": (
            "Quality names pulling back to 50-day moving average "
            "in an established uptrend"
        ),
        "evidence": (
            "Mean reversion within trend — tactical entry point for names "
            "with strong fundamentals."
        ),
        "hold_period": "2-6 weeks",
    },
    3: {
        "name": "Risk-Adjusted Momentum",
        "description": "Top-decile momentum names with volatility scaling",
        "evidence": (
            "Barroso-Santa-Clara (2015): vol-scaling lifts momentum Sharpe "
            "from 0.53 → 0.97"
        ),
        "hold_period": "1-3 months",
    },
    4: {
        "name": "Quality-Momentum",
        "description": "High-quality, high-growth names with positive momentum",
        "evidence": (
            "AQR Quality-Minus-Junk + momentum: most-validated retail combination"
        ),
        "hold_period": "1-6 months",
    },
    5: {
        "name": "Post-Earnings Drift",
        "description": "Recent earnings beats that are still trending up",
        "evidence": (
            "Ball-Brown (1968) PEAD + FRBSF 2024: ~5% 3-month excess return "
            "for SUE > 1.5"
        ),
        "hold_period": "1-3 months",
    },
    6: {
        "name": "Analyst Revision Momentum",
        "description": "Strong analyst consensus with significant price target upside",
        "evidence": (
            "Stickel (1995), Womack (1996): analyst upgrades predict "
            "subsequent returns"
        ),
        "hold_period": "1-3 months",
    },
    7: {
        "name": "Insider Cluster Buys",
        "description": "Multiple insiders buying within 30 days",
        "evidence": (
            "Lakonishok-Lee (2001): insider cluster buys generate "
            "~4.8% excess 6-month return"
        ),
        "hold_period": "3-6 months",
    },
    8: {
        "name": "Quality Oversold",
        "description": "Large-cap quality names with RSI < 30",
        "evidence": (
            "Connors RSI-2 mean reversion, quality-gated to avoid value traps"
        ),
        "hold_period": "1-4 weeks",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_candidate(
    ticker: str,
    screen_id: int,
    reason: str,
    held: bool,
    info: dict,
    ind: dict,
) -> dict:
    """Assemble the standard candidate dict."""
    price = info.get("price") or ind.get("latest_close")
    tgt_mean = info.get("tgt_mean")
    upside_to_pt = None
    if tgt_mean and price:
        upside_to_pt = round((tgt_mean - price) / price * 100, 1)

    return {
        "ticker": ticker,
        "screen_id": screen_id,
        "screen_name": SCREEN_META[screen_id]["name"],
        "reason": reason,
        "held": held,
        "price": price,
        "mcap": info.get("mcap"),
        "rsi": ind.get("latest_rsi"),
        "pct_from_52wh": _scalar(ind.get("pct_from_52wh")),
        "rev_growth": info.get("rev_growth"),
        "buy_pct": info.get("buy_pct"),
        "upside_to_pt": upside_to_pt,
        "vol_adj_mom": ind.get("vol_adj_mom"),
        "hi52": info.get("hi52"),
        "lo52": info.get("lo52"),
        "name": info.get("name"),
        "sector": info.get("sector"),
    }


def _hard_kill(info: dict, ind: dict, mcap_floor: float = 2e9) -> bool:
    """
    Universal hard kills applied to all screens.

    Returns True if the candidate should be skipped.
    """
    mcap = info.get("mcap") or 0
    if mcap < mcap_floor:
        return True

    sell_count = info.get("sell_count") or info.get("sell") or 0
    if sell_count >= 2:
        return True

    rsi = ind.get("latest_rsi")
    pct_from_52wh = ind.get("pct_from_52wh")
    if rsi is not None and pct_from_52wh is not None:
        if rsi > 80 and pct_from_52wh > -5:
            return True

    return False


# ---------------------------------------------------------------------------
# Screen implementations
# ---------------------------------------------------------------------------

def _screen_1_52wh_proximity(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
) -> Optional[dict]:
    """
    Screen 1: 52wH Proximity (George-Hwang 2004)
    - Within 5% of 52-week high: pct_from_52wh >= -5
    - Volume ratio >= 1.5
    - RSI(14) < 80
    - mcap >= 2B
    """
    if _hard_kill(info, ind):
        return None

    pct_from_52wh = ind.get("pct_from_52wh")
    volume_ratio = ind.get("volume_ratio")
    rsi = ind.get("latest_rsi")

    if pct_from_52wh is None or volume_ratio is None or rsi is None:
        return None

    if pct_from_52wh < -5:
        return None
    if volume_ratio < 1.5:
        return None
    if rsi >= 80:
        return None

    reason = (
        f"{abs(pct_from_52wh):.1f}% from 52wk high with {volume_ratio:.1f}x volume "
        f"and RSI {rsi:.0f} — momentum continuation setup (George-Hwang)"
    )
    return _build_candidate(ticker, 1, reason, held, info, ind)


def _screen_2_quality_pullback(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
) -> Optional[dict]:
    """
    Screen 2: Pullback to 50DMA (quality-gated)
    - 50DMA > 200DMA (golden cross)
    - Drawdown 10-30% from 52w high
    - gross_margin >= 30% OR rev_growth >= 10%
    - mcap >= 2B
    """
    if _hard_kill(info, ind):
        return None

    golden_cross = ind.get("golden_cross", False)
    above_50dma = ind.get("above_50dma", False)
    above_200dma = ind.get("above_200dma", False)
    pct_from_52wh = ind.get("pct_from_52wh")

    gross_margin = info.get("gross_margin")
    rev_growth = info.get("rev_growth")

    if not golden_cross:
        return None

    if pct_from_52wh is None:
        return None

    # Drawdown 10-30% from 52w high
    if not (-30 <= pct_from_52wh <= -10):
        return None

    # Quality proxy: gross margin >= 30% OR rev growth >= 10%
    quality_ok = (
        (gross_margin is not None and gross_margin >= 30)
        or (rev_growth is not None and rev_growth >= 10)
    )
    if not quality_ok:
        return None

    quality_str = []
    if gross_margin is not None:
        quality_str.append(f"{gross_margin:.0f}% gross margin")
    if rev_growth is not None:
        quality_str.append(f"{rev_growth:.0f}% rev growth")

    reason = (
        f"{abs(pct_from_52wh):.1f}% below 52wk high in golden-cross uptrend "
        f"with {', '.join(quality_str)} — quality pullback entry"
    )
    return _build_candidate(ticker, 2, reason, held, info, ind)


def _screen_3_risk_adj_momentum(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
    mom_threshold: float = 0.0,
) -> Optional[dict]:
    """
    Screen 3: Risk-Adjusted Momentum (Barroso-Santa-Clara 2015)
    - vol_adj_mom in top 15% of universe (threshold passed in)
    - Above 200DMA
    - RSI < 80
    - mcap >= 2B
    """
    if _hard_kill(info, ind):
        return None

    vol_adj_mom = ind.get("vol_adj_mom")
    above_200dma = ind.get("above_200dma", False)
    rsi = ind.get("latest_rsi")

    if vol_adj_mom is None or rsi is None:
        return None

    if vol_adj_mom < mom_threshold:
        return None
    if not above_200dma:
        return None
    if rsi >= 80:
        return None

    reason = (
        f"Vol-adjusted momentum score {vol_adj_mom:.2f} (above 85th pct threshold), "
        f"above 200DMA, RSI {rsi:.0f} — Barroso-Santa-Clara momentum signal"
    )
    return _build_candidate(ticker, 3, reason, held, info, ind)


def _screen_4_quality_momentum(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
) -> Optional[dict]:
    """
    Screen 4: Quality-Momentum Composite (AQR)
    - vol_adj_mom > 0 (positive momentum)
    - gross_margin >= 40%
    - rev_growth >= 15%
    - buy_pct >= 60%
    - mcap >= 2B
    """
    if _hard_kill(info, ind):
        return None

    vol_adj_mom = ind.get("vol_adj_mom")
    gross_margin = info.get("gross_margin")
    rev_growth = info.get("rev_growth")
    buy_pct = info.get("buy_pct")

    if vol_adj_mom is None or vol_adj_mom <= 0:
        return None
    if gross_margin is None or gross_margin < 40:
        return None
    if rev_growth is None or rev_growth < 15:
        return None
    if buy_pct is None or buy_pct < 60:
        return None

    reason = (
        f"{gross_margin:.0f}% gross margin, {rev_growth:.0f}% rev growth, "
        f"{buy_pct:.0f}% analyst buy consensus with positive momentum "
        f"— AQR quality-momentum composite"
    )
    return _build_candidate(ticker, 4, reason, held, info, ind)


def _screen_5_pead(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
) -> Optional[dict]:
    """
    Screen 5: Post-Earnings Drift (Ball-Brown + FRBSF 2024)
    - Earnings reported within last 30 days
    - pct_today > 0 AND pct_from_52wh > -15 (held the gap)
    - buy_pct >= 60%
    - mcap >= 2B
    """
    if _hard_kill(info, ind):
        return None

    earnings_dt = info.get("earnings_dt")
    pct_today = info.get("pct_today")
    pct_from_52wh = ind.get("pct_from_52wh")
    buy_pct = info.get("buy_pct")

    if earnings_dt is None:
        return None

    today = date.today()
    # Check earnings was within last 30 days (past earnings, not upcoming)
    days_since_earnings = (today - earnings_dt).days
    if not (0 <= days_since_earnings <= 30):
        return None

    if pct_today is None or pct_today <= 0:
        return None
    if pct_from_52wh is None or pct_from_52wh <= -15:
        return None
    if buy_pct is None or buy_pct < 60:
        return None

    reason = (
        f"Earnings {days_since_earnings}d ago, price holding gains "
        f"({pct_today:+.1f}% today, {abs(pct_from_52wh):.1f}% below 52wk high), "
        f"{buy_pct:.0f}% analyst buy — PEAD drift signal"
    )
    return _build_candidate(ticker, 5, reason, held, info, ind)


def _screen_6_analyst_revision(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
) -> Optional[dict]:
    """
    Screen 6: Analyst Revision Momentum (Stickel-Womack)
    - buy_pct >= 75%
    - tgt_mean upside >= 15%
    - sell_count <= 1
    - mcap >= 2B
    """
    if _hard_kill(info, ind):
        return None

    buy_pct = info.get("buy_pct")
    tgt_mean = info.get("tgt_mean")
    price = info.get("price") or ind.get("latest_close")
    sell_count = info.get("sell_count") or info.get("sell") or 0

    if buy_pct is None or buy_pct < 75:
        return None

    if tgt_mean is None or price is None or price == 0:
        return None

    upside = (tgt_mean - price) / price * 100
    if upside < 15:
        return None

    if sell_count > 1:
        return None

    reason = (
        f"{buy_pct:.0f}% analyst buy consensus, {upside:.0f}% upside to mean PT "
        f"${tgt_mean:.0f}, {sell_count} sell rating(s) — Stickel-Womack revision momentum"
    )
    return _build_candidate(ticker, 6, reason, held, info, ind)


def _screen_7_insider_cluster(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
) -> Optional[dict]:
    """
    Screen 7: Insider Cluster Buys (Lakonishok-Lee 2001)
    Uses SEC EDGAR Form 4. Returns stub / empty if edgar unavailable.
    - >= 3 insiders buying in last 30 days
    - Total purchases > $500k
    - No selling
    - mcap >= 2B
    """
    if _hard_kill(info, ind):
        return None

    try:
        from data import edgar  # noqa: F401
        # If edgar module exists, query it
        cluster = edgar.get_insider_cluster(ticker, days=30)
        if not cluster:
            return None

        n_buyers = cluster.get("n_buyers", 0)
        total_value = cluster.get("total_value", 0)
        has_sells = cluster.get("has_sells", True)

        if n_buyers < 3:
            return None
        if total_value < 500_000:
            return None
        if has_sells:
            return None

        reason = (
            f"{n_buyers} insiders bought ${total_value/1e6:.1f}M in last 30 days "
            f"with no selling — Lakonishok-Lee insider cluster signal"
        )
        return _build_candidate(ticker, 7, reason, held, info, ind)

    except ImportError:
        # edgar module not yet available — screen returns nothing gracefully
        return None
    except Exception:
        return None


def _screen_8_quality_oversold(
    ticker: str,
    info: dict,
    ind: dict,
    held: bool,
) -> Optional[dict]:
    """
    Screen 8: Mean Reversion in Quality (Connors RSI-2, quality-gated)
    - RSI(14) < 30
    - mcap >= 5B
    - rev_growth >= 10%
    - 200DMA still rising (golden_cross or above_200dma)
    - pct_from_52wh > -40
    """
    # Screen 8 has a higher mcap floor of 5B
    if _hard_kill(info, ind, mcap_floor=5e9):
        return None

    rsi = ind.get("latest_rsi")
    above_200dma = ind.get("above_200dma", False)
    golden_cross = ind.get("golden_cross", False)
    pct_from_52wh = ind.get("pct_from_52wh")
    rev_growth = info.get("rev_growth")

    if rsi is None or rsi >= 30:
        return None
    if rev_growth is None or rev_growth < 10:
        return None

    # 200DMA still rising: golden cross recently true or above_200dma with recovery
    dma_ok = golden_cross or above_200dma
    if not dma_ok:
        return None

    if pct_from_52wh is None or pct_from_52wh <= -40:
        return None

    reason = (
        f"RSI {rsi:.0f} (oversold) with {rev_growth:.0f}% revenue growth, "
        f"{abs(pct_from_52wh):.1f}% below 52wk high, 200DMA supportive "
        f"— Connors quality oversold mean reversion"
    )
    return _build_candidate(ticker, 8, reason, held, info, ind)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_screen(
    screen_id: int,
    universe_histories: dict,
    ticker_infos: dict,
    held_tickers: set,
) -> list:
    """
    Run one screen. Returns candidates:
    [
      {
        ticker: str,
        screen_id: int,
        screen_name: str,
        reason: str,
        held: bool,
        price: float,
        mcap: float,
        rsi: float,
        pct_from_52wh: float,
        rev_growth: float | None,
        buy_pct: float | None,
        upside_to_pt: float | None,
        vol_adj_mom: float | None,
      }
    ]
    Returns [] on any exception.
    """
    if not _PANDAS_AVAILABLE:
        return []
    if screen_id not in SCREEN_META:
        return []
    if held_tickers is None:
        held_tickers = set()

    # For Screen 3, compute universe-level 85th percentile of vol_adj_mom
    mom_threshold = 0.0
    if screen_id == 3:
        all_moms = []
        for tkr, df in universe_histories.items():
            try:
                ind = compute_all(df)
                v = ind.get("vol_adj_mom")
                if v is not None:
                    all_moms.append(v)
            except Exception:
                pass
        if len(all_moms) >= 5:
            mom_threshold = float(np.percentile(all_moms, 85))

    candidates = []

    screen_fns = {
        1: _screen_1_52wh_proximity,
        2: _screen_2_quality_pullback,
        3: _screen_3_risk_adj_momentum,
        4: _screen_4_quality_momentum,
        5: _screen_5_pead,
        6: _screen_6_analyst_revision,
        7: _screen_7_insider_cluster,
        8: _screen_8_quality_oversold,
    }

    try:
        for ticker, df in universe_histories.items():
            try:
                # Screen 7 doesn't need price history for its core logic
                info = ticker_infos.get(ticker, {})
                if not info and screen_id not in (7,):
                    continue

                ind = compute_all(df)
                held = ticker in held_tickers

                if screen_id == 3:
                    candidate = _screen_3_risk_adj_momentum(
                        ticker, info, ind, held, mom_threshold
                    )
                else:
                    fn = screen_fns.get(screen_id)
                    if fn is None:
                        continue
                    candidate = fn(ticker, info, ind, held)

                if candidate is not None:
                    candidates.append(candidate)

            except Exception:
                continue

    except Exception:
        return []

    return candidates


def run_all_screens(
    universe_histories: dict,
    ticker_infos: dict,
    held_tickers: set = None,
) -> dict:
    """
    Run all 8 screens. Returns {screen_id: [candidates]}.

    For Screen 3 (vol-adj momentum), first computes universe-wide percentile
    so "top 15%" threshold is meaningful.
    """
    if held_tickers is None:
        held_tickers = set()

    results = {}

    # Pre-compute vol-adj mom for all tickers once (used by screen 3)
    all_moms = {}
    if _PANDAS_AVAILABLE:
        for tkr, df in universe_histories.items():
            try:
                ind = compute_all(df)
                v = ind.get("vol_adj_mom")
                if v is not None:
                    all_moms[tkr] = v
            except Exception:
                pass

    mom_threshold_85 = 0.0
    if len(all_moms) >= 5:
        mom_threshold_85 = float(np.percentile(list(all_moms.values()), 85))

    for screen_id in range(1, 9):
        candidates = []
        screen_fns = {
            1: _screen_1_52wh_proximity,
            2: _screen_2_quality_pullback,
            4: _screen_4_quality_momentum,
            5: _screen_5_pead,
            6: _screen_6_analyst_revision,
            7: _screen_7_insider_cluster,
            8: _screen_8_quality_oversold,
        }

        try:
            for ticker, df in universe_histories.items():
                try:
                    info = ticker_infos.get(ticker, {})
                    ind = compute_all(df)
                    held = ticker in held_tickers

                    if screen_id == 3:
                        candidate = _screen_3_risk_adj_momentum(
                            ticker, info, ind, held, mom_threshold_85
                        )
                    else:
                        fn = screen_fns.get(screen_id)
                        if fn is None:
                            continue
                        candidate = fn(ticker, info, ind, held)

                    if candidate is not None:
                        candidates.append(candidate)

                except Exception:
                    continue

        except Exception:
            pass

        results[screen_id] = candidates

    return results
