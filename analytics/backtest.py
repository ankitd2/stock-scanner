"""
analytics/backtest.py — walk-forward backtest of the 8 candidate screens.

Replays each price-based screen on a monthly cadence over the lookback window
(default 5 years), records the names each screen would have surfaced, measures
forward 1m/3m/6m returns vs SPY, and reports aggregate stats per screen.

Output is cached to disk (.cache/backtest.json) and refreshed on a 14-day TTL.
The HTML report displays the most recent backtest stats inline under each
screen card.

CRITICAL: this module is INFORMATIONAL only. It must never block live screens.
All public entry points catch exceptions and return empty/stub results rather
than raising. The cache is best-effort.

Screens that need point-in-time fundamentals or alt-data (5 PEAD, 6 analyst
revisions, 7 insider buys) are marked skipped — yfinance does not expose
historical snapshots of estimates / earnings dates / Form 4 filings in the
bulk universe path we use here.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from analytics.screens import SCREEN_META
except Exception:  # pragma: no cover — defensive fallback
    SCREEN_META = {}


CACHE_FILE = Path(__file__).parent.parent / ".cache" / "backtest.json"
CACHE_TTL_DAYS = 14

# Screens we can replay with price/volume only.
PRICE_ONLY_SCREENS = {1, 2, 3, 4, 8}

# Screens that cannot be backtested with bulk yfinance history.
SKIPPED_SCREENS = {
    5: "Requires historical earnings dates and SUE snapshots (not in yfinance bulk).",
    6: "Requires point-in-time analyst estimate snapshots.",
    7: "Requires SEC Form 4 historical filings (separate data flow).",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def screen_can_be_backtested(screen_id: int) -> tuple[bool, str]:
    """
    Returns (can_backtest, reason_if_not).
    Screens 1, 2, 3, 4, 8 — backtestable (price/technical only).
    Screen 5 (PEAD), 6 (analyst revisions), 7 (insider buys) — skipped.
    """
    if screen_id in PRICE_ONLY_SCREENS:
        return True, ""
    return False, SKIPPED_SCREENS.get(
        screen_id, "Screen requires non-price data unavailable to the backtester."
    )


def _screen_name(screen_id: int) -> str:
    meta = SCREEN_META.get(screen_id, {}) if isinstance(SCREEN_META, dict) else {}
    return meta.get("name", f"Screen {screen_id}")


# ---------------------------------------------------------------------------
# Universe preparation
# ---------------------------------------------------------------------------

def _to_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return Close series with normalized title-case column lookup."""
    if df is None or len(df) == 0:
        return None
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close")
    if close_col is None:
        return None
    try:
        s = df[close_col].astype(float)
        return s.dropna()
    except Exception:
        return None


def _to_volume_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    cols = {c.lower(): c for c in df.columns}
    vol_col = cols.get("volume")
    if vol_col is None:
        return None
    try:
        return df[vol_col].astype(float)
    except Exception:
        return None


def _build_panels(
    universe_histories: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build aligned close & volume panels across the universe.
    Rows = dates, cols = tickers. Returns (close_panel, volume_panel).
    """
    close_series: dict[str, pd.Series] = {}
    vol_series: dict[str, pd.Series] = {}

    for tkr, df in (universe_histories or {}).items():
        c = _to_close_series(df)
        v = _to_volume_series(df)
        if c is None or len(c) < 60:
            continue
        close_series[tkr] = c
        if v is not None:
            vol_series[tkr] = v

    if not close_series:
        return pd.DataFrame(), pd.DataFrame()

    close_panel = pd.DataFrame(close_series).sort_index()
    # Forward fill across business days so missing days don't produce NaN
    # explosions in rolling windows.
    close_panel = close_panel.ffill(limit=5)

    vol_panel = pd.DataFrame(vol_series).reindex(close_panel.index).ffill(limit=5)
    return close_panel, vol_panel


def _subsample_liquid(
    close_panel: pd.DataFrame,
    vol_panel: pd.DataFrame,
    n_top: int = 300,
) -> list[str]:
    """Top N tickers by 252d-avg dollar volume."""
    if close_panel.empty:
        return []
    try:
        avg_window = min(252, len(close_panel))
        recent_close = close_panel.tail(avg_window)
        recent_vol = vol_panel.reindex(recent_close.index)
        dollar_vol = (recent_close * recent_vol).mean(axis=0)
        dollar_vol = dollar_vol.dropna()
        if dollar_vol.empty:
            # Fallback: rank by data availability
            return list(close_panel.columns[:n_top])
        top = dollar_vol.sort_values(ascending=False).head(n_top).index.tolist()
        return list(top)
    except Exception:
        return list(close_panel.columns[:n_top])


# ---------------------------------------------------------------------------
# Indicator helpers (vectorized over the panel)
# ---------------------------------------------------------------------------

def _rsi_at(close: pd.Series, idx: int, period: int = 14) -> Optional[float]:
    """Compute RSI(period) on close series up to and including idx."""
    if idx < period + 1 or idx >= len(close):
        return None
    window = close.iloc[: idx + 1]
    delta = window.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    if avg_loss.iloc[-1] == 0 or math.isnan(avg_loss.iloc[-1]):
        return None
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    if math.isnan(rs):
        return None
    return float(100 - (100 / (1 + rs)))


def _ticker_metrics_at(
    close: pd.Series,
    volume: Optional[pd.Series],
    as_of_idx: int,
) -> Optional[dict]:
    """
    Compute the price-only metrics needed by all backtestable screens at a
    single point in time. Returns None if data is insufficient.
    """
    if close is None or as_of_idx < 200 or as_of_idx >= len(close):
        return None

    cur = float(close.iloc[as_of_idx])
    if not np.isfinite(cur) or cur <= 0:
        return None

    # 52-week (252d) high
    win252 = close.iloc[max(0, as_of_idx - 251): as_of_idx + 1]
    if len(win252) < 50:
        return None
    hi252 = float(win252.max())
    if not np.isfinite(hi252) or hi252 <= 0:
        return None
    pct_from_52wh = (cur / hi252 - 1.0) * 100.0

    # 50DMA, 200DMA
    sma50 = float(close.iloc[max(0, as_of_idx - 49): as_of_idx + 1].mean())
    sma200 = float(close.iloc[max(0, as_of_idx - 199): as_of_idx + 1].mean())
    sma200_prev = float(close.iloc[max(0, as_of_idx - 200): as_of_idx].mean()) \
        if as_of_idx >= 200 else float("nan")

    above_50dma = cur > sma50
    above_200dma = cur > sma200
    golden_cross = sma50 > sma200
    sma200_rising = (
        not math.isnan(sma200_prev) and sma200 > sma200_prev
    )

    # RSI
    rsi_val = _rsi_at(close, as_of_idx, period=14)

    # Volume ratio: today's vol / 20d avg vol
    vol_ratio: Optional[float] = None
    if volume is not None and as_of_idx < len(volume):
        win_vol = volume.iloc[max(0, as_of_idx - 19): as_of_idx + 1]
        try:
            avg_vol = float(win_vol.mean())
            today_vol = float(volume.iloc[as_of_idx])
            if avg_vol > 0 and np.isfinite(avg_vol) and np.isfinite(today_vol):
                vol_ratio = today_vol / avg_vol
        except Exception:
            vol_ratio = None

    # Vol-adjusted momentum (Barroso-Santa-Clara): 252d return / annualized vol
    vol_adj_mom: Optional[float] = None
    if as_of_idx >= 252:
        win_ret = close.iloc[as_of_idx - 251: as_of_idx + 1]
        try:
            daily_ret = win_ret.pct_change().dropna()
            ann_vol = float(daily_ret.std() * math.sqrt(252))
            if ann_vol > 0 and np.isfinite(ann_vol):
                # Use the (-252, -21) gap convention to match indicators.py
                start = float(close.iloc[as_of_idx - 251])
                end = float(close.iloc[as_of_idx - 20]) if as_of_idx >= 20 else cur
                if start > 0 and np.isfinite(start) and np.isfinite(end):
                    vol_adj_mom = (end / start - 1.0) / ann_vol
        except Exception:
            vol_adj_mom = None

    return {
        "price": cur,
        "pct_from_52wh": pct_from_52wh,
        "above_50dma": above_50dma,
        "above_200dma": above_200dma,
        "golden_cross": golden_cross,
        "sma200_rising": sma200_rising,
        "sma50": sma50,
        "rsi": rsi_val,
        "vol_ratio": vol_ratio,
        "vol_adj_mom": vol_adj_mom,
    }


# ---------------------------------------------------------------------------
# Simplified screen replay
# ---------------------------------------------------------------------------

def _simplified_screen(
    screen_id: int,
    universe_state: dict[str, pd.DataFrame],
    spy_state: pd.DataFrame,
    as_of_idx: int,
) -> list[str]:
    """
    Simplified historical replay of a price-only screen.

    `universe_state` and `spy_state` are the *full* close/volume panels — the
    caller passes both panels as a dict so the function can compute relative
    metrics. as_of_idx indexes into the close panel's row axis.

    Returns list of tickers that would have qualified at the given index.
    Returns [] for non-backtestable screens or on error.
    """
    if screen_id not in PRICE_ONLY_SCREENS:
        return []

    close_panel = universe_state.get("close") if isinstance(universe_state, dict) else None
    vol_panel = universe_state.get("volume") if isinstance(universe_state, dict) else None
    if close_panel is None or not isinstance(close_panel, pd.DataFrame):
        return []
    if as_of_idx < 200 or as_of_idx >= len(close_panel):
        return []

    # First pass: gather metrics for every ticker
    metrics: dict[str, dict] = {}
    for ticker in close_panel.columns:
        try:
            close = close_panel[ticker]
            if close.iloc[: as_of_idx + 1].dropna().shape[0] < 200:
                continue
            volume = vol_panel[ticker] if vol_panel is not None and ticker in vol_panel.columns else None
            m = _ticker_metrics_at(close, volume, as_of_idx)
            if m is None:
                continue
            metrics[ticker] = m
        except Exception:
            continue

    if not metrics:
        return []

    # Compute universe-wide 85th percentile of vol_adj_mom for screen 3 / top quartile for screen 4
    moms = [m["vol_adj_mom"] for m in metrics.values() if m.get("vol_adj_mom") is not None]
    mom_85 = float(np.percentile(moms, 85)) if len(moms) >= 5 else 0.0
    mom_75 = float(np.percentile(moms, 75)) if len(moms) >= 5 else 0.0

    qualified: list[str] = []

    for ticker, m in metrics.items():
        try:
            if screen_id == 1:
                # 52wH proximity: within 5% of high, RSI<80, vol_ratio>=1.5
                if (
                    m["pct_from_52wh"] is not None
                    and m["pct_from_52wh"] >= -5
                    and m["rsi"] is not None and m["rsi"] < 80
                    and m["vol_ratio"] is not None and m["vol_ratio"] >= 1.5
                ):
                    qualified.append(ticker)

            elif screen_id == 2:
                # Pullback to 50DMA: golden cross, 10-30% from high,
                # price within 3% of 50DMA from above
                if not m["golden_cross"]:
                    continue
                p = m["price"]
                sma50 = m["sma50"]
                if sma50 <= 0:
                    continue
                if not (-30 <= m["pct_from_52wh"] <= -10):
                    continue
                # within 3% of 50DMA (from above)
                near_50 = (p >= sma50) and ((p - sma50) / sma50 * 100 <= 3.0)
                if near_50:
                    qualified.append(ticker)

            elif screen_id == 3:
                # Risk-adjusted momentum: top 15% by vol-adj mom, above 200DMA,
                # RSI < 80
                if m["vol_adj_mom"] is None:
                    continue
                if m["vol_adj_mom"] < mom_85:
                    continue
                if not m["above_200dma"]:
                    continue
                if m["rsi"] is None or m["rsi"] >= 80:
                    continue
                qualified.append(ticker)

            elif screen_id == 4:
                # Quality-momentum proxy: top 25% by vol-adj mom AND price
                # above sma50 AND price above sma200. (No fundamentals.)
                if m["vol_adj_mom"] is None:
                    continue
                if m["vol_adj_mom"] < mom_75:
                    continue
                if not (m["above_50dma"] and m["above_200dma"]):
                    continue
                qualified.append(ticker)

            elif screen_id == 8:
                # Quality oversold: RSI < 30, 200DMA still rising, drawdown < 40%
                if m["rsi"] is None or m["rsi"] >= 30:
                    continue
                if not m["sma200_rising"]:
                    continue
                if m["pct_from_52wh"] is None or m["pct_from_52wh"] <= -40:
                    continue
                qualified.append(ticker)
        except Exception:
            continue

    return qualified


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def _fwd_return(series: pd.Series, idx: int, horizon_days: int) -> Optional[float]:
    """Forward return from idx to idx+horizon_days, as a percent."""
    if idx < 0 or idx >= len(series):
        return None
    target_idx = idx + horizon_days
    if target_idx >= len(series):
        return None
    p0 = float(series.iloc[idx])
    p1 = float(series.iloc[target_idx])
    if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
        return None
    return (p1 / p0 - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def _empty_stats(screen_id: int, lookback_weeks: int) -> dict:
    return {
        "name": _screen_name(screen_id),
        "n_observations": 0,
        "n_unique_tickers": 0,
        "median_fwd_1m": 0.0,
        "median_fwd_3m": 0.0,
        "median_fwd_6m": 0.0,
        "spy_baseline_1m": 0.0,
        "spy_baseline_3m": 0.0,
        "spy_baseline_6m": 0.0,
        "alpha_3m": 0.0,
        "alpha_6m": 0.0,
        "hit_rate_3m": 0.0,
        "sharpe_3m": None,
        "max_drawdown": None,
        "lookback_weeks": lookback_weeks,
        "as_of": date.today().isoformat(),
    }


def _skipped_stats(screen_id: int) -> dict:
    return {
        "name": _screen_name(screen_id),
        "skipped": True,
        "reason": SKIPPED_SCREENS.get(
            screen_id, "Screen requires non-price data unavailable to the backtester."
        ),
    }


def run_walkforward_backtest(
    screen_ids: list[int],
    universe_histories: dict[str, pd.DataFrame],
    spy_history: pd.DataFrame,
    lookback_weeks: int = 260,
    rebalance_freq_weeks: int = 4,
) -> dict[int, dict]:
    """
    Walk-forward backtest of `screen_ids` over the last `lookback_weeks` weeks.

    At each rebalance date:
      1. Reconstruct universe state up to that date.
      2. Run a simplified (price-only) version of each screen.
      3. Compute forward 1m/3m/6m returns for surfaced tickers and SPY.

    Returns: {screen_id: stats_dict}.

    Screens that cannot be backtested receive a {"skipped": True, "reason": ...}
    stub.
    """
    results: dict[int, dict] = {}

    # Always emit a row per requested screen
    for sid in screen_ids:
        if sid in SKIPPED_SCREENS:
            results[sid] = _skipped_stats(sid)

    # Build the close/volume panels for the universe
    close_panel, vol_panel = _build_panels(universe_histories)
    if close_panel.empty:
        for sid in screen_ids:
            if sid not in results:
                results[sid] = _empty_stats(sid, lookback_weeks)
        return results

    # Subsample to most-liquid 300 tickers
    liquid = _subsample_liquid(close_panel, vol_panel, n_top=300)
    if liquid:
        close_panel = close_panel[liquid]
        vol_panel = vol_panel[[c for c in liquid if c in vol_panel.columns]] \
            if not vol_panel.empty else vol_panel

    # Build the SPY series aligned to the panel
    spy_close = _to_close_series(spy_history) if spy_history is not None else None
    if spy_close is None or spy_close.empty:
        # Fall back: use mean of universe as a poor-man's baseline
        spy_close = close_panel.mean(axis=1).dropna()
    spy_close = spy_close.reindex(close_panel.index).ffill()

    # Trading-day horizons
    H_1M = 21
    H_3M = 63
    H_6M = 126

    n_rows = len(close_panel)
    if n_rows < 252 + H_3M:
        for sid in screen_ids:
            if sid not in results:
                results[sid] = _empty_stats(sid, lookback_weeks)
        return results

    # Rebalance indices: every `rebalance_freq_weeks * 5` trading days
    step = max(5, rebalance_freq_weeks * 5)
    # Earliest replay index: need 252 days of history + ability to compute
    # forward 6m return (so don't go past n_rows - H_6M).
    start_idx = max(252, n_rows - lookback_weeks * 5)
    end_idx = n_rows - H_1M  # need at least 1m forward window
    rebalance_indices = list(range(start_idx, end_idx, step))

    panel_state = {"close": close_panel, "volume": vol_panel}

    # Per-screen aggregations
    agg: dict[int, dict] = {}
    for sid in screen_ids:
        if sid in SKIPPED_SCREENS:
            continue
        agg[sid] = {
            "fwd_1m": [],
            "fwd_3m": [],
            "fwd_6m": [],
            "spy_1m": [],
            "spy_3m": [],
            "spy_6m": [],
            "tickers": set(),
            "obs": 0,
            "wins_3m": 0,
            "wins_total_3m": 0,
        }

    for idx in rebalance_indices:
        spy_fwd_1m = _fwd_return(spy_close, idx, H_1M)
        spy_fwd_3m = _fwd_return(spy_close, idx, H_3M)
        spy_fwd_6m = _fwd_return(spy_close, idx, H_6M)

        for sid in agg.keys():
            try:
                names = _simplified_screen(sid, panel_state, spy_close, idx)
            except Exception:
                names = []
            for tkr in names:
                try:
                    s = close_panel[tkr]
                    r1 = _fwd_return(s, idx, H_1M)
                    r3 = _fwd_return(s, idx, H_3M)
                    r6 = _fwd_return(s, idx, H_6M)
                    if r1 is not None:
                        agg[sid]["fwd_1m"].append(r1)
                        if spy_fwd_1m is not None:
                            agg[sid]["spy_1m"].append(spy_fwd_1m)
                    if r3 is not None:
                        agg[sid]["fwd_3m"].append(r3)
                        agg[sid]["obs"] += 1
                        if spy_fwd_3m is not None:
                            agg[sid]["spy_3m"].append(spy_fwd_3m)
                            agg[sid]["wins_total_3m"] += 1
                            if r3 > spy_fwd_3m:
                                agg[sid]["wins_3m"] += 1
                    if r6 is not None:
                        agg[sid]["fwd_6m"].append(r6)
                        if spy_fwd_6m is not None:
                            agg[sid]["spy_6m"].append(spy_fwd_6m)
                    agg[sid]["tickers"].add(tkr)
                except Exception:
                    continue

    today_iso = date.today().isoformat()
    for sid, data in agg.items():
        if not data["fwd_3m"]:
            stats = _empty_stats(sid, lookback_weeks)
            results[sid] = stats
            continue

        med_1m = float(np.median(data["fwd_1m"])) if data["fwd_1m"] else 0.0
        med_3m = float(np.median(data["fwd_3m"]))
        med_6m = float(np.median(data["fwd_6m"])) if data["fwd_6m"] else 0.0

        spy_1m = float(np.median(data["spy_1m"])) if data["spy_1m"] else 0.0
        spy_3m = float(np.median(data["spy_3m"])) if data["spy_3m"] else 0.0
        spy_6m = float(np.median(data["spy_6m"])) if data["spy_6m"] else 0.0

        # Sharpe (3m): annualized
        arr3 = np.asarray(data["fwd_3m"], dtype=float)
        if arr3.size >= 2 and arr3.std(ddof=0) > 0:
            # Each obs is a 3m return; annualize by sqrt(4)
            sharpe = float((arr3.mean() / arr3.std(ddof=0)) * math.sqrt(4))
        else:
            sharpe = None

        max_dd = float(arr3.min()) if arr3.size > 0 else None

        hit_rate = (
            (data["wins_3m"] / data["wins_total_3m"]) * 100
            if data["wins_total_3m"] > 0
            else 0.0
        )

        results[sid] = {
            "name": _screen_name(sid),
            "n_observations": int(data["obs"]),
            "n_unique_tickers": int(len(data["tickers"])),
            "median_fwd_1m": round(med_1m, 2),
            "median_fwd_3m": round(med_3m, 2),
            "median_fwd_6m": round(med_6m, 2),
            "spy_baseline_1m": round(spy_1m, 2),
            "spy_baseline_3m": round(spy_3m, 2),
            "spy_baseline_6m": round(spy_6m, 2),
            "alpha_3m": round(med_3m - spy_3m, 2),
            "alpha_6m": round(med_6m - spy_6m, 2),
            "hit_rate_3m": round(hit_rate, 1),
            "sharpe_3m": round(sharpe, 3) if sharpe is not None else None,
            "max_drawdown": round(max_dd, 2) if max_dd is not None else None,
            "lookback_weeks": lookback_weeks,
            "as_of": today_iso,
        }

    # Ensure all requested screens are represented
    for sid in screen_ids:
        if sid not in results:
            results[sid] = _empty_stats(sid, lookback_weeks)

    return results


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cached_backtest(cache_path: Path | None = None) -> dict[int, dict]:
    """
    Load .cache/backtest.json if present and <CACHE_TTL_DAYS days old.

    Returns {} if missing, stale, or unparseable.
    """
    path = cache_path or CACHE_FILE
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text())
        cached_on = raw.get("_cached_on")
        if cached_on:
            try:
                cached_dt = datetime.fromisoformat(cached_on)
                age_days = (datetime.now() - cached_dt).days
                if age_days >= CACHE_TTL_DAYS:
                    return {}
            except Exception:
                return {}
        results = raw.get("results", {})
        # Convert string keys back to ints
        return {int(k): v for k, v in results.items()}
    except Exception:
        return {}


def save_backtest_cache(
    results: dict[int, dict],
    cache_path: Path | None = None,
) -> None:
    """Persist results to .cache/backtest.json atomically."""
    path = cache_path or CACHE_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_cached_on": datetime.now().isoformat(timespec="seconds"),
            "_ttl_days": CACHE_TTL_DAYS,
            "results": {str(k): v for k, v in (results or {}).items()},
        }
        # Atomic write: temp file in same dir, then rename
        fd, tmp_path = tempfile.mkstemp(
            prefix=".backtest_", suffix=".json", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up the temp if rename failed
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
    except Exception as e:
        print(f"[backtest] cache save failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _fetch_extended_history(
    universe_histories: dict[str, pd.DataFrame],
    period: str = "5y",
    n_top: int = 300,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    The walk-forward backtest needs 5y of history but the main scanner only
    pulls 1y. Subsample the universe to the top `n_top` most-liquid tickers
    (using the 1y panel we already have) and re-fetch their full history.

    Returns (extended_histories, spy_history). Falls back to the 1y panel
    when the extended fetch fails.
    """
    close_panel, vol_panel = _build_panels(universe_histories)
    if close_panel.empty:
        return universe_histories, pd.DataFrame()

    liquid = _subsample_liquid(close_panel, vol_panel, n_top=n_top)
    if not liquid:
        liquid = list(close_panel.columns)[:n_top]

    # Always include SPY so the baseline works
    if "SPY" not in liquid:
        liquid = ["SPY"] + liquid

    try:
        from data.fetcher import bulk_history
    except ImportError:
        return universe_histories, universe_histories.get("SPY", pd.DataFrame())

    print(f"[backtest] fetching {period} history for top {len(liquid)} "
          f"liquid tickers (cold cache)...", file=sys.stderr)
    extended = bulk_history(liquid, period=period)
    if not extended or len(extended) < 50:
        # Fetch failed catastrophically — fall back to 1y data
        print("[backtest] extended fetch returned too few tickers; "
              "falling back to 1y data", file=sys.stderr)
        return universe_histories, universe_histories.get("SPY", pd.DataFrame())

    spy_history = extended.get("SPY", universe_histories.get("SPY", pd.DataFrame()))
    return extended, spy_history


def get_or_compute_backtest(
    universe_histories: dict[str, pd.DataFrame],
    spy_history: pd.DataFrame,
    force_refresh: bool = False,
    cache_path: Path | None = None,
) -> dict[int, dict]:
    """
    Returns cached backtest if <CACHE_TTL_DAYS days old, else runs a fresh
    walk-forward and caches the result.

    The walk-forward needs 5y of data. The main scanner only pulls 1y, so on
    a cold cache we re-fetch 5y of history for the most-liquid subsample
    (~300 tickers, ~2-3 extra minutes — only once every 14 days).

    Never raises: any internal failure returns an empty stub so the report
    still renders.
    """
    if not force_refresh:
        cached = load_cached_backtest(cache_path)
        if cached:
            return cached

    try:
        # Fetch the 5y history we actually need for walk-forward
        ext_histories, ext_spy = _fetch_extended_history(
            universe_histories, period="5y", n_top=300,
        )
        # Use the extended SPY if we got one, else fall back to caller's
        spy = ext_spy if ext_spy is not None and not ext_spy.empty else spy_history

        results = run_walkforward_backtest(
            screen_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            universe_histories=ext_histories or {},
            spy_history=spy if spy is not None else pd.DataFrame(),
        )
        # Persist
        save_backtest_cache(results, cache_path=cache_path)
        return results
    except Exception as e:
        print(f"[backtest] walk-forward failed: {e}", file=sys.stderr)
        # Last-resort fallback: stub for every screen so the report still renders
        fallback: dict[int, dict] = {}
        for sid in range(1, 9):
            if sid in SKIPPED_SCREENS:
                fallback[sid] = _skipped_stats(sid)
            else:
                fallback[sid] = _empty_stats(sid, 260)
        return fallback
