"""
analytics/clustering.py

Auto-discovers emerging thematic clusters using Mantegna correlation distance +
hierarchical agglomerative clustering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Optional


def compute_correlation_matrix(
    histories: dict[str, pd.DataFrame],
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation of daily log-returns over last `lookback_days`.
    Returns square DataFrame with tickers as index/columns.
    Only includes tickers with >= lookback_days * 0.8 observations.
    """
    if not histories:
        return pd.DataFrame()

    min_obs = int(lookback_days * 0.8)
    returns_dict: dict[str, pd.Series] = {}

    for ticker, df in histories.items():
        if df is None or df.empty:
            continue
        # Use Close column (or Adj Close); take last lookback_days rows
        close_col = None
        for col in ["Close", "Adj Close", "close", "adj_close"]:
            if col in df.columns:
                close_col = col
                break
        if close_col is None:
            continue

        prices = df[close_col].dropna()
        prices = prices.iloc[-lookback_days:] if len(prices) >= lookback_days else prices

        if len(prices) < min_obs:
            continue

        log_ret = np.log(prices / prices.shift(1)).dropna()
        if len(log_ret) < min_obs - 1:
            continue

        returns_dict[ticker] = log_ret

    if len(returns_dict) < 2:
        return pd.DataFrame()

    # Align on common index
    returns_df = pd.DataFrame(returns_dict)
    # Drop columns (tickers) still below min_obs after alignment
    valid_cols = returns_df.columns[returns_df.count() >= (min_obs - 1)]
    returns_df = returns_df[valid_cols]

    if returns_df.shape[1] < 2:
        return pd.DataFrame()

    corr = returns_df.corr(method="pearson")
    return corr


def mantegna_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Mantegna correlation distance: d(i,j) = sqrt(2 * (1 - rho(i,j)))
    Returns distance matrix (same shape as corr_matrix).
    """
    if corr_matrix.empty:
        return pd.DataFrame()

    # Clip correlations to [-1, 1] to avoid numerical issues with sqrt
    rho = corr_matrix.values.clip(-1.0, 1.0)
    # Replace NaN with 0 correlation (maximum uncertainty => distance ~1.41)
    rho = np.where(np.isnan(rho), 0.0, rho)
    dist = np.sqrt(2.0 * (1.0 - rho))
    # Ensure diagonal is exactly 0
    np.fill_diagonal(dist, 0.0)

    return pd.DataFrame(dist, index=corr_matrix.index, columns=corr_matrix.columns)


def cluster_universe(
    distance_matrix: pd.DataFrame,
    min_size: int = 4,
    max_size: int = 30,
    min_internal_corr: float = 0.55,
) -> list[list[str]]:
    """
    Hierarchical agglomerative clustering (average linkage) on distance matrix.
    Cut tree at distance threshold 0.9 (corresponds to ~rho=0.6 in Mantegna distance).
    Filter clusters: size in [min_size, max_size] AND mean internal correlation >= min_internal_corr.
    Returns list of ticker lists (each list is one cluster), sorted by cluster size desc.
    """
    if distance_matrix is None or distance_matrix.empty:
        return []

    tickers = list(distance_matrix.index)
    n = len(tickers)

    if n < min_size:
        return []

    dist_vals = distance_matrix.values.copy().astype(float)
    # Ensure symmetry and zero diagonal
    np.fill_diagonal(dist_vals, 0.0)
    dist_vals = (dist_vals + dist_vals.T) / 2.0

    # Replace any NaN with max distance
    dist_vals = np.where(np.isnan(dist_vals), np.sqrt(4.0), dist_vals)

    try:
        condensed = squareform(dist_vals, checks=False)
        Z = linkage(condensed, method="average")
    except Exception:
        return []

    # Cut threshold 0.9 corresponds to rho = 1 - (0.9^2 / 2) = 1 - 0.405 = 0.595 ~ 0.6
    labels = fcluster(Z, t=0.9, criterion="distance")

    # Group tickers by cluster label
    cluster_map: dict[int, list[str]] = {}
    for ticker, label in zip(tickers, labels):
        cluster_map.setdefault(int(label), []).append(ticker)

    # Reverse-map: build correlation matrix from distance for filtering
    # d = sqrt(2*(1-rho)) => rho = 1 - d^2/2
    corr_from_dist = 1.0 - (dist_vals ** 2) / 2.0

    valid_clusters: list[list[str]] = []
    for label, members in cluster_map.items():
        size = len(members)
        if size < min_size or size > max_size:
            continue

        # Compute mean internal correlation
        idx = [tickers.index(m) for m in members]
        sub_corr = corr_from_dist[np.ix_(idx, idx)]
        # Only upper-triangle off-diagonal elements
        iu = np.triu_indices(len(idx), k=1)
        if len(iu[0]) == 0:
            continue
        mean_corr = float(np.mean(sub_corr[iu]))

        if mean_corr >= min_internal_corr:
            valid_clusters.append(members)

    # Sort by cluster size descending
    valid_clusters.sort(key=lambda c: len(c), reverse=True)
    return valid_clusters


def _mean_internal_corr(
    members: list[str],
    corr_matrix: pd.DataFrame,
) -> float:
    """Return mean pairwise correlation for a set of tickers in a correlation matrix."""
    available = [m for m in members if m in corr_matrix.index]
    if len(available) < 2:
        return 0.0
    sub = corr_matrix.loc[available, available]
    iu = np.triu_indices(len(available), k=1)
    vals = sub.values[iu]
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def detect_emerging_clusters(
    histories: dict[str, pd.DataFrame],
    lookback_days: int = 60,
    delta_threshold: float = 0.20,
    min_now_corr: float = 0.60,
    max_prior_corr: float = 0.40,
) -> list[dict]:
    """
    Compares correlation structure of last 60d vs prior 60d.
    An "emerging" cluster is one where:
      - Mean internal correlation rose by > delta_threshold (default 0.20)
      - Current mean internal correlation > min_now_corr (default 0.60)
      - Prior mean internal correlation < max_prior_corr (default 0.40)

    Algorithm:
    1. C_now = correlation matrix for last `lookback_days` days
    2. C_prior = correlation matrix for days [-2*lookback_days : -lookback_days]
    3. cluster_now = cluster_universe(C_now)
    4. For each cluster in cluster_now, check if it qualifies as "emerging"
    5. Return top 3 emerging clusters (by delta magnitude)

    Returns list of:
      {
        members: list[str],          # ticker symbols
        corr_now: float,             # current mean pairwise correlation
        corr_prior: float,           # prior mean pairwise correlation
        delta: float,                # corr_now - corr_prior
        label: str,                  # largest 3 tickers joined, e.g. "NVDA / AVGO / AMD"
        size: int,
      }
    """
    if not histories:
        return []

    required_history = lookback_days * 2 + 5  # a small buffer

    # Split histories into now (last lookback_days) and prior (preceding lookback_days)
    now_histories: dict[str, pd.DataFrame] = {}
    prior_histories: dict[str, pd.DataFrame] = {}

    for ticker, df in histories.items():
        if df is None or df.empty:
            continue

        close_col = None
        for col in ["Close", "Adj Close", "close", "adj_close"]:
            if col in df.columns:
                close_col = col
                break
        if close_col is None:
            continue

        prices = df[[close_col]].dropna()
        total_rows = len(prices)

        if total_rows < required_history:
            # Still include in now_histories for C_now if enough data
            if total_rows >= int(lookback_days * 0.8):
                now_histories[ticker] = prices.iloc[-lookback_days:]
            continue

        now_histories[ticker] = prices.iloc[-lookback_days:]
        prior_slice = prices.iloc[-2 * lookback_days: -lookback_days]
        if len(prior_slice) >= int(lookback_days * 0.8):
            prior_histories[ticker] = prior_slice

    if len(now_histories) < 4:
        return []

    # Compute correlation matrices
    c_now = compute_correlation_matrix(now_histories, lookback_days=lookback_days)
    if c_now.empty or c_now.shape[0] < 4:
        return []

    dist_now = mantegna_distance(c_now)
    clusters_now = cluster_universe(dist_now, min_size=4, max_size=30, min_internal_corr=0.55)

    if not clusters_now:
        return []

    # Need prior correlation matrix for delta computation
    c_prior = (
        compute_correlation_matrix(prior_histories, lookback_days=lookback_days)
        if len(prior_histories) >= 4
        else pd.DataFrame()
    )

    emerging: list[dict] = []

    for cluster_members in clusters_now:
        corr_now_val = _mean_internal_corr(cluster_members, c_now)
        if corr_now_val < min_now_corr:
            continue

        if c_prior.empty:
            # No prior data — cannot compute delta, skip
            continue

        corr_prior_val = _mean_internal_corr(cluster_members, c_prior)
        delta = corr_now_val - corr_prior_val

        if delta > delta_threshold and corr_prior_val < max_prior_corr:
            # Label: up to 3 tickers (use first 3 by list order as proxy for largest)
            label_tickers = cluster_members[:3]
            label = " / ".join(label_tickers)

            emerging.append(
                {
                    "members": cluster_members,
                    "corr_now": round(corr_now_val, 4),
                    "corr_prior": round(corr_prior_val, 4),
                    "delta": round(delta, 4),
                    "label": label,
                    "size": len(cluster_members),
                }
            )

    # Sort by delta magnitude descending, return top 3
    emerging.sort(key=lambda x: x["delta"], reverse=True)
    return emerging[:3]
