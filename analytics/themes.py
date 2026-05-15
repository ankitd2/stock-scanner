"""
analytics/themes.py

Scores thematic baskets, computes sector rotation, and detects emerging clusters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import numpy as np
from analytics.clustering import detect_emerging_clusters

THEMES_FILE = Path(__file__).parent.parent / "universe" / "themes.yaml"


def load_themes() -> dict:
    """Load universe/themes.yaml. Returns {"themes": {...}, "sector_etfs": [...]}"""
    with open(THEMES_FILE, "r") as fh:
        data = yaml.safe_load(fh)
    return data


def _calc_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI(period) and return the latest value."""
    if prices is None or len(prices) < period + 1:
        return float("nan")
    delta = prices.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.iloc[:period].mean()
    avg_loss = loss.iloc[:period].mean()
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _get_close(df: pd.DataFrame) -> pd.Series | None:
    """Extract close price series from a history DataFrame."""
    for col in ["Close", "Adj Close", "close", "adj_close"]:
        if col in df.columns:
            return df[col].dropna()
    return None


def _period_return(prices: pd.Series, n_days: int) -> float | None:
    """Return % return over last n_days trading days."""
    if prices is None or len(prices) < n_days + 1:
        return None
    start = prices.iloc[-(n_days + 1)]
    end = prices.iloc[-1]
    if start == 0 or pd.isna(start):
        return None
    return float((end - start) / start * 100.0)


def compute_theme_strength(
    theme_key: str,
    members: list[str],
    histories: dict[str, pd.DataFrame],
    spy_history: pd.DataFrame,
) -> dict:
    """
    Compute strength score for one thematic basket.

    Components (all z-scored across the 16 baskets — caller does the z-scoring):
      rel_strength_4w:    equal-weight basket return (4w) - SPY return (4w)
      rel_strength_12w:   same for 12 weeks
      pct_above_50dma:    % of members with close > 50DMA (0-100)
      internal_cohesion:  average pairwise 30d return correlation among members
      pct_near_52wh:      % of members within 5% of their 52w high (0-100)
      median_rsi:         median RSI(14) across members

    Returns dict with all 6 components + "available_members" (tickers we got data for).
    Returns {} if fewer than 3 members have data.
    """
    spy_close = _get_close(spy_history) if spy_history is not None and not spy_history.empty else None

    spy_ret_4w = _period_return(spy_close, 20) if spy_close is not None else None
    spy_ret_12w = _period_return(spy_close, 60) if spy_close is not None else None

    # 4w = ~20 trading days, 12w = ~60 trading days
    ret_4w_list: list[float] = []
    ret_12w_list: list[float] = []
    above_50dma: list[bool] = []
    near_52wh: list[bool] = []
    rsi_vals: list[float] = []
    returns_30d: dict[str, pd.Series] = {}  # for internal cohesion
    available_members: list[str] = []

    for ticker in members:
        df = histories.get(ticker)
        if df is None or df.empty:
            continue
        prices = _get_close(df)
        if prices is None or len(prices) < 20:
            continue

        available_members.append(ticker)

        # 4w return
        r4 = _period_return(prices, 20)
        if r4 is not None:
            ret_4w_list.append(r4)

        # 12w return
        r12 = _period_return(prices, 60)
        if r12 is not None:
            ret_12w_list.append(r12)

        # 50 DMA
        if len(prices) >= 50:
            dma50 = prices.iloc[-50:].mean()
            above_50dma.append(bool(prices.iloc[-1] > dma50))

        # 52-week high
        if len(prices) >= 252:
            hi52 = prices.iloc[-252:].max()
        else:
            hi52 = prices.max()
        if hi52 > 0:
            near_52wh.append(bool(prices.iloc[-1] >= hi52 * 0.95))

        # RSI
        if len(prices) >= 15:
            rsi = _calc_rsi(prices, 14)
            if not pd.isna(rsi):
                rsi_vals.append(rsi)

        # 30d log-returns for cohesion
        if len(prices) >= 32:
            log_ret = np.log(prices / prices.shift(1)).dropna().iloc[-30:]
            returns_30d[ticker] = log_ret

    if len(available_members) < 3:
        return {}

    # Relative strength (4w)
    basket_4w = float(np.mean(ret_4w_list)) if ret_4w_list else 0.0
    rs_4w = basket_4w - (spy_ret_4w if spy_ret_4w is not None else 0.0)

    # Relative strength (12w)
    basket_12w = float(np.mean(ret_12w_list)) if ret_12w_list else 0.0
    rs_12w = basket_12w - (spy_ret_12w if spy_ret_12w is not None else 0.0)

    # % above 50 DMA
    pct_above_50dma = (sum(above_50dma) / len(above_50dma) * 100.0) if above_50dma else 50.0

    # Internal cohesion: avg pairwise 30d correlation
    internal_cohesion = 0.0
    if len(returns_30d) >= 2:
        ret_df = pd.DataFrame(returns_30d)
        corr_mat = ret_df.corr()
        iu = np.triu_indices(corr_mat.shape[0], k=1)
        vals = corr_mat.values[iu]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            internal_cohesion = float(np.mean(valid))

    # % near 52-week high
    pct_near_52wh = (sum(near_52wh) / len(near_52wh) * 100.0) if near_52wh else 0.0

    # Median RSI
    median_rsi = float(np.median(rsi_vals)) if rsi_vals else 50.0

    return {
        "rel_strength_4w": rs_4w,
        "rel_strength_12w": rs_12w,
        "pct_above_50dma": pct_above_50dma,
        "internal_cohesion": internal_cohesion,
        "pct_near_52wh": pct_near_52wh,
        "median_rsi": median_rsi,
        "available_members": available_members,
    }


def _zscore_column(values: list[float]) -> list[float]:
    """Z-score a list of values. Returns 0.0 for all elements if std == 0."""
    arr = np.array(values, dtype=float)
    std = arr.std()
    if std == 0 or np.isnan(std):
        return [0.0] * len(values)
    return list((arr - arr.mean()) / std)


def rank_themes(
    histories: dict[str, pd.DataFrame],
    spy_history: pd.DataFrame,
) -> list[dict]:
    """
    Score all 16 baskets, z-score each component across baskets, compute composite:
      theme_score = 0.30*z(rs_4w) + 0.20*z(rs_12w) + 0.15*z(pct_above_50dma)
                  + 0.15*z(internal_cohesion) + 0.10*z(pct_near_52wh) + 0.10*z(median_rsi)

    Returns list of dicts sorted descending by theme_score:
      [{key, name, description, score, components, rank, members, available_members}, ...]
    """
    data = load_themes()
    themes_cfg = data.get("themes", {})

    raw_results: list[dict] = []

    for theme_key, theme_meta in themes_cfg.items():
        name = theme_meta.get("name", theme_key)
        description = theme_meta.get("description", "")
        members = theme_meta.get("members", [])

        components = compute_theme_strength(
            theme_key, members, histories, spy_history
        )
        if not components:
            continue

        raw_results.append(
            {
                "key": theme_key,
                "name": name,
                "description": description,
                "members": members,
                "available_members": components.pop("available_members"),
                "_components": components,
            }
        )

    if not raw_results:
        return []

    # Z-score each component across baskets
    component_keys = [
        "rel_strength_4w",
        "rel_strength_12w",
        "pct_above_50dma",
        "internal_cohesion",
        "pct_near_52wh",
        "median_rsi",
    ]
    weights = {
        "rel_strength_4w": 0.30,
        "rel_strength_12w": 0.20,
        "pct_above_50dma": 0.15,
        "internal_cohesion": 0.15,
        "pct_near_52wh": 0.10,
        "median_rsi": 0.10,
    }

    # Extract raw component values for z-scoring
    for ck in component_keys:
        raw_vals = [r["_components"].get(ck, 0.0) for r in raw_results]
        z_vals = _zscore_column(raw_vals)
        for r, z in zip(raw_results, z_vals):
            r["_components"][f"z_{ck}"] = z

    # Compute composite theme_score
    for r in raw_results:
        score = sum(
            weights[ck] * r["_components"].get(f"z_{ck}", 0.0)
            for ck in component_keys
        )
        r["score"] = round(float(score), 4)
        # Keep clean components dict (both raw and z-scored)
        r["components"] = r.pop("_components")

    # Sort descending by score and assign rank
    raw_results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(raw_results):
        r["rank"] = i + 1

    return raw_results


def compute_sector_rotation(
    histories: dict[str, pd.DataFrame],
    spy_history: pd.DataFrame,
    previous_ranks: dict[str, int] | None = None,
) -> list[dict]:
    """
    Faber-style momentum ranking of 11 SPDR sector ETFs.
    rank_score = 0.6 * RS_3m + 0.4 * RS_6m
    where RS_Nm = (sector_return_N - SPY_return_N)

    Returns list sorted descending by rank_score:
      [{symbol, name, rank_score, rs_3m, rs_6m, rank, rank_delta, ytd_return}, ...]
    rank_delta: vs previous week's rank (pass previous_ranks dict or None)

    Also includes a "rotation_call" string:
      e.g. "XLU +4 ranks — defensive rotation signal"
      Only generated if any sector moved 3+ ranks.
    """
    data = load_themes()
    sector_etfs = data.get("sector_etfs", [])

    spy_close = _get_close(spy_history) if spy_history is not None and not spy_history.empty else None
    spy_ret_3m = _period_return(spy_close, 63) if spy_close is not None else None
    spy_ret_6m = _period_return(spy_close, 126) if spy_close is not None else None

    # YTD: approximate as Jan 1 = ~252 * (days_into_year/365) trading days
    # Use a simple fixed 126 for ~6m as proxy, or find YTD from year start price
    spy_ret_ytd: float | None = None
    if spy_close is not None and len(spy_close) >= 252:
        # Use first available price of the year (252 days back as approximation)
        spy_ret_ytd = _period_return(spy_close, 252)

    results: list[dict] = []

    for etf in sector_etfs:
        symbol = etf["symbol"]
        name = etf["name"]

        df = histories.get(symbol)
        if df is None or df.empty:
            continue

        prices = _get_close(df)
        if prices is None or len(prices) < 30:
            continue

        rs_3m = None
        ret_3m = _period_return(prices, 63)
        if ret_3m is not None and spy_ret_3m is not None:
            rs_3m = ret_3m - spy_ret_3m
        elif ret_3m is not None:
            rs_3m = ret_3m

        rs_6m = None
        ret_6m = _period_return(prices, 126)
        if ret_6m is not None and spy_ret_6m is not None:
            rs_6m = ret_6m - spy_ret_6m
        elif ret_6m is not None:
            rs_6m = ret_6m

        # YTD return
        ytd_return = None
        if len(prices) >= 252:
            ytd_return = _period_return(prices, 252)

        # rank_score: use what's available
        if rs_3m is not None and rs_6m is not None:
            rank_score = 0.6 * rs_3m + 0.4 * rs_6m
        elif rs_3m is not None:
            rank_score = rs_3m
        elif rs_6m is not None:
            rank_score = rs_6m
        else:
            continue

        results.append(
            {
                "symbol": symbol,
                "name": name,
                "rank_score": round(float(rank_score), 4),
                "rs_3m": round(float(rs_3m), 4) if rs_3m is not None else None,
                "rs_6m": round(float(rs_6m), 4) if rs_6m is not None else None,
                "ytd_return": round(float(ytd_return), 4) if ytd_return is not None else None,
                "rank": None,
                "rank_delta": None,
            }
        )

    # Sort and assign rank
    results.sort(key=lambda x: x["rank_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
        if previous_ranks and r["symbol"] in previous_ranks:
            r["rank_delta"] = previous_ranks[r["symbol"]] - (i + 1)

    return results


def _infer_stovall_phase(top3_symbols: list[str]) -> str:
    """
    Infer Stovall cycle phase from top-3 sectors.
    Tech+Disc+Comm in top-3 → "Mid Cycle"
    Energy+Materials+Industrials → "Late Cycle"
    Utilities+Staples+Health in top-3 → "Recession"
    Otherwise → "Early Cycle"
    """
    tech_disc_comm = {"XLK", "XLY", "XLC"}
    energy_mat_ind = {"XLE", "XLB", "XLI"}
    def_staples_health = {"XLU", "XLP", "XLV"}

    top3_set = set(top3_symbols)

    overlap_tdc = len(top3_set & tech_disc_comm)
    overlap_emi = len(top3_set & energy_mat_ind)
    overlap_dsh = len(top3_set & def_staples_health)

    if overlap_tdc >= 2:
        return "Mid Cycle"
    if overlap_emi >= 2:
        return "Late Cycle"
    if overlap_dsh >= 2:
        return "Recession"
    return "Early Cycle"


def analyze_themes(
    universe_histories: dict[str, pd.DataFrame],
) -> dict:
    """
    Master function. Loads themes, ranks all baskets, runs sector rotation,
    calls detect_emerging_clusters.

    Returns:
      {
        ranked_themes: list[dict],       # all 16, sorted by score
        top_themes: list[dict],          # top 5
        sector_rotation: list[dict],     # 11 sectors, sorted
        rotation_call: str | None,       # "XLU +4 ranks — defensive rotation" or None
        emerging_clusters: list[dict],   # from clustering module
        stovall_phase: str,              # "Early Cycle" | "Mid Cycle" | "Late Cycle" | "Recession"
      }
    """
    # SPY is the benchmark
    spy_history = universe_histories.get("SPY", pd.DataFrame())

    # Rank themes
    ranked_themes = rank_themes(universe_histories, spy_history)
    top_themes = ranked_themes[:5]

    # Sector rotation
    sector_rotation = compute_sector_rotation(universe_histories, spy_history)

    # Rotation call: any sector moved 3+ ranks vs previous? (no prev here — skip)
    rotation_call: str | None = None
    # rotation_call generation relies on rank_delta; since no prev ranks passed here,
    # we leave it None at this level. Callers can pass previous_ranks separately.

    # Emerging clusters
    emerging_clusters = detect_emerging_clusters(universe_histories)

    # Stovall phase from top-3 sectors
    top3_symbols = [s["symbol"] for s in sector_rotation[:3]] if len(sector_rotation) >= 3 else []
    stovall_phase = _infer_stovall_phase(top3_symbols)

    return {
        "ranked_themes": ranked_themes,
        "top_themes": top_themes,
        "sector_rotation": sector_rotation,
        "rotation_call": rotation_call,
        "emerging_clusters": emerging_clusters,
        "stovall_phase": stovall_phase,
    }
