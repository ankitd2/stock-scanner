"""
analytics/market_state.py
=========================
Market State Analysis Module

Computes 9 market-state indicators and synthesizes them into a single
0-100 Market State Score (0 = maximum stress, 100 = maximum risk-on).

Indicator groups
----------------
1. Breadth          — % above 50/200 DMA, 52-week new highs/lows
2. VIX signals      — spot VIX, VIX3M, VIX9D, VVIX, term structure, percentile
3. Cross-asset      — RSP/SPY ratio, SPHB/SPLV, copper/gold, HY OAS, AAII sentiment

Score formula (sigmoid of weighted z-scores):
  raw = (
    -0.20 * z(vix_percentile)
    -0.15 * z(hy_oas)
    -0.15 * z(vix_term_structure)
    +0.10 * z(pct_above_200dma/100)
    +0.10 * z(pct_above_50dma/100)
    +0.10 * z(sphb_splv_ratio)
    +0.08 * z(rsp_spy_30d_delta)
    -0.07 * z(vvix)
    -0.05 * abs(z(aaii_bull_bear_zscore))
  )
  score = int(100 * sigmoid(raw * 2))
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Ensure project root is on sys.path regardless of invocation method ──────
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Import data layer with fallback for both absolute and relative imports ──
try:
    from data.fetcher import get_index_data, bulk_history          # noqa: F401
    from data.fred import latest_fred, fred_zscore, get_fred_series  # noqa: F401
    from data.aaii import latest_aaii                               # noqa: F401
except ImportError:
    try:
        from .fetcher import get_index_data, bulk_history           # type: ignore
        from .fred import latest_fred, fred_zscore, get_fred_series  # type: ignore
        from .aaii import latest_aaii                                # type: ignore
    except ImportError:
        # Define no-op stubs so the module is importable without the data layer
        def get_index_data(symbols):  # type: ignore
            return {}

        def bulk_history(tickers, period="1y", interval="1d"):  # type: ignore
            return {}

        def latest_fred():  # type: ignore
            return {"hy_oas": None, "t10y2y": None, "dgs10": None}

        def fred_zscore(series, window=260):  # type: ignore
            return 0.0

        def get_fred_series():  # type: ignore
            return {}

        def latest_aaii():  # type: ignore
            return {
                "bullish": None, "bearish": None, "neutral": None,
                "bull_bear_spread": None, "bull_bear_zscore_5y": None,
            }

try:
    import pandas as pd
    import numpy as np
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ───────────────────────────────────────────────────────────────────────────
# Internal helpers
# ───────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid, numerically stable."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _safe_float(val, default: Optional[float] = None) -> Optional[float]:
    """Convert val to float safely, returning default on failure."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _threshold_z(value: Optional[float], mean: float, std: float) -> float:
    """Convert a value to a z-score using fixed mean/std thresholds.
    Returns 0.0 if value is None or std == 0.
    """
    v = _safe_float(value)
    if v is None or std == 0:
        return 0.0
    return (v - mean) / std


def _series_z(series, value: Optional[float]) -> float:
    """Compute z-score of value against a pandas Series. Returns 0.0 on failure."""
    if not _PANDAS_AVAILABLE or series is None:
        return 0.0
    v = _safe_float(value)
    if v is None:
        return 0.0
    try:
        s = pd.Series(series).dropna()
        if len(s) < 10:
            return 0.0
        mean = float(s.mean())
        std = float(s.std())
        if std == 0:
            return 0.0
        return (v - mean) / std
    except Exception:
        return 0.0


def _percentile_rank(series, value: Optional[float]) -> Optional[float]:
    """Return percentile rank (0-100) of value within series."""
    if not _PANDAS_AVAILABLE or series is None:
        return None
    v = _safe_float(value)
    if v is None:
        return None
    try:
        s = pd.Series(series).dropna()
        if len(s) < 10:
            return None
        return float((s <= v).mean() * 100)
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────────────────
# 1. Breadth
# ───────────────────────────────────────────────────────────────────────────

def compute_breadth(universe_histories: Dict[str, "pd.DataFrame"]) -> dict:
    """
    Given bulk_history output for the full universe (~1000 tickers),
    compute breadth indicators across the universe.

    Returns:
      {
        pct_above_50dma:  float (0-100),
        pct_above_200dma: float (0-100),
        new_highs_52w:    int,
        new_lows_52w:     int,
        new_highs_count:  int,  # alias for new_highs_52w
        new_lows_count:   int,
      }
    """
    default = {
        "pct_above_50dma": None,
        "pct_above_200dma": None,
        "new_highs_52w": None,
        "new_lows_52w": None,
        "new_highs_count": None,
        "new_lows_count": None,
    }

    if not _PANDAS_AVAILABLE or not universe_histories:
        return default

    above_50 = above_200 = new_hi = new_lo = total = 0

    for ticker, df in universe_histories.items():
        try:
            if df is None or df.empty:
                continue
            # Normalise column names to lowercase
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            close_col = next(
                (c for c in ("close", "adj close", "adjclose") if c in df.columns), None
            )
            if close_col is None:
                continue
            close = df[close_col].dropna()
            if len(close) < 50:
                continue

            total += 1
            latest = float(close.iloc[-1])

            # 50-DMA
            if len(close) >= 50:
                sma50 = float(close.tail(50).mean())
                if latest > sma50:
                    above_50 += 1

            # 200-DMA
            if len(close) >= 200:
                sma200 = float(close.tail(200).mean())
                if latest > sma200:
                    above_200 += 1

            # 52-week high/low (252 trading days)
            if len(close) >= 252:
                hi252 = float(close.tail(252).max())
                lo252 = float(close.tail(252).min())
                if latest >= hi252:
                    new_hi += 1
                if latest <= lo252:
                    new_lo += 1
        except Exception:
            continue

    if total == 0:
        return default

    # Only compute percentages where we have enough data points
    above_50_total = sum(
        1 for ticker, df in universe_histories.items()
        if df is not None and not df.empty and len(df) >= 50
    )
    above_200_total = sum(
        1 for ticker, df in universe_histories.items()
        if df is not None and not df.empty and len(df) >= 200
    )

    pct_50 = (above_50 / above_50_total * 100) if above_50_total > 0 else None
    pct_200 = (above_200 / above_200_total * 100) if above_200_total > 0 else None

    return {
        "pct_above_50dma": pct_50,
        "pct_above_200dma": pct_200,
        "new_highs_52w": new_hi,
        "new_lows_52w": new_lo,
        "new_highs_count": new_hi,   # alias
        "new_lows_count": new_lo,
    }


# ───────────────────────────────────────────────────────────────────────────
# 2. VIX Signals
# ───────────────────────────────────────────────────────────────────────────

def compute_vix_signals(index_data: dict) -> dict:
    """
    index_data = get_index_data(['^VIX', '^VIX3M', '^VIX9D', '^VVIX'])

    Returns:
      {
        vix:              float | None,
        vix3m:            float | None,
        vix9d:            float | None,
        vvix:             float | None,
        vix_term_structure: float | None,  # vix / vix3m
        vix9d_ratio:      float | None,    # vix9d / vix
        vix_pct_1y:       float | None,    # VIX percentile vs last 252 days
      }
    """
    out: Dict[str, Optional[float]] = {
        "vix": None, "vix3m": None, "vix9d": None, "vvix": None,
        "vix_term_structure": None, "vix9d_ratio": None, "vix_pct_1y": None,
    }
    try:
        vix   = _safe_float(index_data.get("^VIX", {}).get("price"))
        vix3m = _safe_float(index_data.get("^VIX3M", {}).get("price"))
        vix9d = _safe_float(index_data.get("^VIX9D", {}).get("price"))
        vvix  = _safe_float(index_data.get("^VVIX", {}).get("price"))

        out["vix"] = vix
        out["vix3m"] = vix3m
        out["vix9d"] = vix9d
        out["vvix"] = vvix

        # Term structure: < 1 = contango (calm), > 1 = backwardation (stress)
        if vix is not None and vix3m is not None and vix3m > 0:
            out["vix_term_structure"] = vix / vix3m

        # Short-term stress: vix9d / vix
        if vix9d is not None and vix is not None and vix > 0:
            out["vix9d_ratio"] = vix9d / vix

        # VIX percentile vs 252-day history — requires fetching history
        try:
            vix_hist = bulk_history(["^VIX"], period="1y", interval="1d")
            if vix_hist and "^VIX" in vix_hist:
                df = vix_hist["^VIX"]
                if df is not None and not df.empty:
                    close_col = next(
                        (c for c in df.columns if c.lower() in ("close", "adj close", "adjclose")),
                        df.columns[0],
                    )
                    closes = df[close_col].dropna()
                    pct = _percentile_rank(closes, vix)
                    out["vix_pct_1y"] = pct
        except Exception:
            pass

    except Exception:
        pass

    return out


# ───────────────────────────────────────────────────────────────────────────
# 3. Cross-Asset
# ───────────────────────────────────────────────────────────────────────────

def compute_cross_asset(index_data: dict, histories: Dict[str, "pd.DataFrame"]) -> dict:
    """
    Returns:
      {
        rsp_spy_ratio:         float | None,
        rsp_spy_30d_delta:     float | None,
        sphb_splv_ratio:       float | None,
        copper_gold_ratio:     float | None,
        hy_oas:                float | None,
        hy_oas_zscore:         float | None,
        aaii_bull_bear_zscore: float | None,
      }
    """
    out: Dict[str, Optional[float]] = {
        "rsp_spy_ratio": None,
        "rsp_spy_30d_delta": None,
        "sphb_splv_ratio": None,
        "copper_gold_ratio": None,
        "hy_oas": None,
        "hy_oas_zscore": None,
        "aaii_bull_bear_zscore": None,
    }
    try:
        def _price(sym: str) -> Optional[float]:
            return _safe_float(index_data.get(sym, {}).get("price"))

        # RSP / SPY equal-weight vs cap-weight breadth
        rsp = _price("RSP")
        spy = _price("SPY")
        if rsp is not None and spy is not None and spy > 0:
            out["rsp_spy_ratio"] = rsp / spy

        # RSP/SPY 30-day delta from history
        try:
            rsp_hist = histories.get("RSP")
            spy_hist = histories.get("SPY")
            if rsp_hist is not None and spy_hist is not None:
                def _get_close(df):
                    df = df.copy()
                    df.columns = [c.lower() for c in df.columns]
                    col = next((c for c in ("close", "adj close", "adjclose") if c in df.columns), None)
                    return df[col].dropna() if col else None

                rsp_close = _get_close(rsp_hist)
                spy_close = _get_close(spy_hist)
                if rsp_close is not None and spy_close is not None and len(rsp_close) >= 30 and len(spy_close) >= 30:
                    ratio_now = float(rsp_close.iloc[-1]) / float(spy_close.iloc[-1])
                    # Align 30 days ago
                    rsp_30 = float(rsp_close.iloc[-30]) if len(rsp_close) >= 30 else float(rsp_close.iloc[0])
                    spy_30 = float(spy_close.iloc[-30]) if len(spy_close) >= 30 else float(spy_close.iloc[0])
                    ratio_30d = rsp_30 / spy_30 if spy_30 > 0 else None
                    if ratio_30d is not None:
                        out["rsp_spy_30d_delta"] = ratio_now - ratio_30d
        except Exception:
            pass

        # SPHB / SPLV (high beta / low vol — risk appetite)
        sphb = _price("SPHB")
        splv = _price("SPLV")
        if sphb is not None and splv is not None and splv > 0:
            out["sphb_splv_ratio"] = sphb / splv

        # Copper / Gold (global growth appetite)
        copper = _price("HG=F")
        gold   = _price("GC=F")
        if copper is not None and gold is not None and gold > 0:
            out["copper_gold_ratio"] = copper / gold

        # HY OAS from FRED
        try:
            fred = latest_fred()
            hy_oas = _safe_float(fred.get("hy_oas"))
            out["hy_oas"] = hy_oas
            if hy_oas is not None:
                # Attempt series-based z-score first
                try:
                    fred_series = get_fred_series()
                    hy_series = fred_series.get("hy_oas")
                    if hy_series is not None and len(hy_series) > 20:
                        out["hy_oas_zscore"] = _series_z(hy_series, hy_oas)
                    else:
                        # Fall back to fixed thresholds: mean=350, std=150
                        out["hy_oas_zscore"] = _threshold_z(hy_oas, mean=350, std=150)
                except Exception:
                    out["hy_oas_zscore"] = _threshold_z(hy_oas, mean=350, std=150)
        except Exception:
            pass

        # AAII sentiment z-score
        try:
            aaii = latest_aaii()
            z = _safe_float(aaii.get("bull_bear_zscore_5y"))
            out["aaii_bull_bear_zscore"] = z
        except Exception:
            pass

    except Exception:
        pass

    return out


# ───────────────────────────────────────────────────────────────────────────
# 4. Composite score helpers
# ───────────────────────────────────────────────────────────────────────────

def _score_to_regime(score: int) -> str:
    """Map 0-100 score to regime label."""
    if score >= 70:
        return "Risk-On"
    if score >= 50:
        return "Neutral"
    if score >= 35:
        return "Caution"
    return "Risk-Off"


def _build_summary(score: int, drivers: List[dict], cross_asset: dict, breadth: dict, vix: dict) -> str:
    """Generate a 1-sentence plain-English summary of market state."""
    regime = _score_to_regime(score)

    # Identify the biggest positive and negative drivers
    if not drivers:
        return f"Market state score is {score}/100 ({regime}) with insufficient data to characterize drivers."

    top_pos = [d for d in drivers if d["contribution"] > 0]
    top_neg = [d for d in drivers if d["contribution"] < 0]

    strengths = [d["name"] for d in top_pos[:2]] if top_pos else []
    weaknesses = [d["name"] for d in top_neg[:2]] if top_neg else []

    parts = []
    if strengths:
        parts.append(f"{' and '.join(strengths)} {'are' if len(strengths) > 1 else 'is'} supportive")
    if weaknesses:
        w_str = ' and '.join(weaknesses)
        parts.append(f"{w_str} {'are' if len(weaknesses) > 1 else 'is'} a headwind")

    if parts:
        sentence = "; ".join(parts) + f" (score: {score}/100, {regime})."
        return sentence[0].upper() + sentence[1:]
    return f"Market state score is {score}/100 ({regime})."


# ───────────────────────────────────────────────────────────────────────────
# 5. Master function
# ───────────────────────────────────────────────────────────────────────────

def compute_market_state(
    universe_histories: Dict[str, "pd.DataFrame"],
    index_data: dict,
    fred_data: dict,
    aaii_data: dict,
) -> dict:
    """
    Master function. Computes composite 0-100 Market State Score.

    Parameters
    ----------
    universe_histories : bulk_history output for the full universe (~1000 tickers)
    index_data         : get_index_data output (must include VIX, RSP, SPY, SPHB, SPLV, HG=F, GC=F)
    fred_data          : latest_fred() output  {hy_oas, t10y2y, dgs10}
    aaii_data          : latest_aaii() output  {bull_bear_zscore_5y, ...}

    Returns
    -------
    {
      score:       int (0-100),
      regime:      str,
      breadth:     dict,
      vix:         dict,
      cross_asset: dict,
      drivers:     list[dict],
      summary:     str,
    }
    """
    # ── Defaults ─────────────────────────────────────────────────────────────
    default_result = {
        "score": 50,
        "regime": "Neutral",
        "breadth": {},
        "vix": {},
        "cross_asset": {},
        "drivers": [],
        "summary": "Insufficient data to compute market state.",
    }

    try:
        # ── Step 1: Sub-indicators ──────────────────────────────────────────
        breadth     = compute_breadth(universe_histories)
        vix_signals = compute_vix_signals(index_data)

        # Merge cross-asset data: prefer passed-in fred_data/aaii_data over live fetch
        # but fall back to live fetch if not provided.
        merged_index = dict(index_data) if index_data else {}

        # Inject FRED HY OAS into index_data if provided externally
        # (compute_cross_asset fetches FRED internally, but we can override with passed data)
        cross = compute_cross_asset(merged_index, universe_histories)

        # Override HY OAS with passed-in fred_data if available
        if fred_data:
            hy_oas_ext = _safe_float(fred_data.get("hy_oas"))
            if hy_oas_ext is not None:
                cross["hy_oas"] = hy_oas_ext
                # Recompute z-score with threshold method if no series available
                if cross.get("hy_oas_zscore") is None:
                    cross["hy_oas_zscore"] = _threshold_z(hy_oas_ext, mean=350, std=150)

        # Override AAII with passed-in aaii_data if available
        if aaii_data:
            bb_z = _safe_float(aaii_data.get("bull_bear_zscore_5y"))
            if bb_z is not None:
                cross["aaii_bull_bear_zscore"] = bb_z

        # ── Step 2: Z-scores for each factor ───────────────────────────────
        # Factor: vix_percentile
        vix_pct = vix_signals.get("vix_pct_1y")
        vix_val = vix_signals.get("vix")
        if vix_pct is not None:
            # Already 0-100; map to z-score: mean=50, std=25
            z_vix = _threshold_z(vix_pct, mean=50, std=25)
        else:
            # Fall back to raw VIX thresholds: mean=20, std=7
            z_vix = _threshold_z(vix_val, mean=20, std=7)

        # Factor: hy_oas
        hy_oas_z = cross.get("hy_oas_zscore")
        if hy_oas_z is None:
            hy_oas_val = cross.get("hy_oas")
            hy_oas_z = _threshold_z(hy_oas_val, mean=350, std=150)

        # Factor: vix_term_structure
        vts = vix_signals.get("vix_term_structure")
        # Typical range: 0.7-1.3; mean=0.9, std=0.15
        z_vts = _threshold_z(vts, mean=0.9, std=0.15)

        # Factor: pct_above_200dma (normalised to 0-1)
        p200 = breadth.get("pct_above_200dma")
        # mean=60% → 0.6, std=0.20
        z_p200 = _threshold_z(p200 / 100 if p200 is not None else None, mean=0.6, std=0.20)

        # Factor: pct_above_50dma (normalised to 0-1)
        p50 = breadth.get("pct_above_50dma")
        # mean=55% → 0.55, std=0.22
        z_p50 = _threshold_z(p50 / 100 if p50 is not None else None, mean=0.55, std=0.22)

        # Factor: sphb_splv_ratio
        sphb_splv = cross.get("sphb_splv_ratio")
        # Typical range: 0.5-1.5; mean=0.90, std=0.15
        z_sphb_splv = _threshold_z(sphb_splv, mean=0.90, std=0.15)

        # Factor: rsp_spy_30d_delta
        rsp_spy_delta = cross.get("rsp_spy_30d_delta")
        # Typical range: -0.05 to +0.05; mean=0, std=0.02
        z_rsp_spy = _threshold_z(rsp_spy_delta, mean=0.0, std=0.02)

        # Factor: vvix
        vvix_val = vix_signals.get("vvix")
        # Typical range: 80-140; mean=105, std=15
        z_vvix = _threshold_z(vvix_val, mean=105, std=15)

        # Factor: aaii_bull_bear_zscore (use absolute value — both extremes are contrarian warnings)
        bb_z = cross.get("aaii_bull_bear_zscore")
        z_aaii_abs = abs(_safe_float(bb_z, 0.0))

        # ── Step 3: Weighted sum ────────────────────────────────────────────
        def _w(weight: float, z: float) -> float:
            """Clip z to ±3 for robustness then apply weight."""
            cz = max(-3.0, min(3.0, z))
            return weight * cz

        # Build driver contributions (before scoring)
        driver_raw = [
            ("VIX Percentile",    -0.20, z_vix),
            ("HY OAS",            -0.15, hy_oas_z),
            ("VIX Term Structure", -0.15, z_vts),
            ("% Above 200 DMA",   +0.10, z_p200),
            ("% Above 50 DMA",    +0.10, z_p50),
            ("SPHB/SPLV Ratio",   +0.10, z_sphb_splv),
            ("RSP/SPY Delta",     +0.08, z_rsp_spy),
            ("VVIX",              -0.07, z_vvix),
            ("AAII Sentiment",    -0.05, z_aaii_abs),  # always negative (contrarian)
        ]

        raw_total = 0.0
        drivers: List[dict] = []

        for name, weight, z in driver_raw:
            # For AAII, contribution is always -(0.05 * z_abs) regardless of z sign
            if name == "AAII Sentiment":
                contribution = _w(-0.05, z_aaii_abs)
            else:
                contribution = _w(weight, z)
            raw_total += contribution
            direction = "positive" if contribution > 0 else "negative" if contribution < 0 else "neutral"
            drivers.append({"name": name, "contribution": round(contribution, 4), "direction": direction})

        # Sort by absolute contribution descending
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)

        # ── Step 4: Score ───────────────────────────────────────────────────
        score = int(round(100 * _sigmoid(raw_total * 2)))
        score = max(0, min(100, score))
        regime = _score_to_regime(score)

        summary = _build_summary(score, drivers, cross, breadth, vix_signals)

        return {
            "score": score,
            "regime": regime,
            "breadth": breadth,
            "vix": vix_signals,
            "cross_asset": cross,
            "drivers": drivers,
            "summary": summary,
        }

    except Exception as exc:
        default_result["summary"] = f"Error computing market state: {exc}"
        return default_result
