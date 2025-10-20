from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

EPSILON = 1e-6


def apply_single_name_cap(weights: pd.Series, cap: float = 0.10) -> pd.Series:
    """Apply a deterministic post-optimisation 10% single name cap.

    The optimiser may occasionally allocate more than the configured cap to a
    single security (e.g. due to turnover blending or infeasible solutions). We
    therefore re-project the weights onto the capped simplex following the
    procedure described in the stabilisation brief for v0.4.9.

    Parameters
    ----------
    weights:
        Candidate portfolio weights indexed by ticker.
    cap:
        Maximum allocation per security (default 10%).

    Returns
    -------
    pandas.Series
        Rebalanced weights respecting the cap, non-negative and ordered from
        highest to lowest allocation.
    """

    if weights is None:
        return pd.Series(dtype="float64")

    w = pd.Series(weights, dtype="float64").clip(lower=0.0)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return w.sort_values(ascending=False)

    w = w / total
    cap = float(max(0.0, cap))
    if cap == 0:
        return pd.Series(0.0, index=w.index)

    excess = (w - cap).clip(lower=0.0).sum()
    w = w.clip(upper=cap)

    if w.sum() < 1.0 and excess > 0:
        room = (cap - w).clip(lower=0.0)
        room_total = room.sum()
        if room_total > 0:
            allocatable = min(excess, room_total)
            if allocatable > 0:
                w += room * (allocatable / room_total)

    final_total = w.sum()
    if final_total <= 0:
        return pd.Series(0.0, index=w.index)

    return (w / final_total).sort_values(ascending=False)


def inverse_vol_weights(
    returns_252: pd.DataFrame,
    tickers: list,
    cap_single: float = 0.10,
    k: int = 15,
) -> pd.Series:
    vol = returns_252.std().replace(0, np.nan).dropna()
    vol = vol.loc[vol.index.intersection(tickers)]
    top = vol.nsmallest(k).index
    inv = 1.0 / vol.loc[top]
    w = inv / inv.sum()
    w = apply_single_name_cap(w, cap_single)
    return w

def apply_sector_caps(weights: pd.Series, sector_map: pd.Series, cap: float = 0.35) -> pd.Series:
    w = weights.copy()
    sectors = sector_map.loc[w.index]
    for sec, sw in w.groupby(sectors).sum().items():
        if sw > cap:
            scale = cap / sw
            w.loc[sectors == sec] *= scale
    return w / w.sum()

def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    if prev_w is None or prev_w.empty:
        return new_w.abs().sum()
    idx = prev_w.index.union(new_w.index)
    return (prev_w.reindex(idx, fill_value=0) - new_w.reindex(idx, fill_value=0)).abs().sum()

def enforce_turnover(prev_w: pd.Series, cand_w: pd.Series, t_cap: float = 0.30) -> pd.Series:
    if prev_w is None or prev_w.empty:
        return cand_w
    t = turnover(prev_w, cand_w)
    if t <= t_cap:
        return cand_w
    # Linearly interpolate toward prev_w to meet turnover cap
    lam = t_cap / t
    blended = lam * cand_w + (1 - lam) * prev_w.reindex(cand_w.index, fill_value=0)
    return (blended / blended.sum()).clip(lower=0)


def apply_turnover_controls(
    prev_w: pd.Series | None,
    target_w: pd.Series,
    *,
    turnover_cap: float = 0.40,
    rebalance_band: float = 0.25,
) -> pd.Series:
    """Apply rebalance banding and turnover cap controls."""

    target = pd.Series(target_w).astype(float).clip(lower=0.0)
    if target.empty:
        return target
    target_sum = target.sum()
    if target_sum <= 0:
        return target
    target = target / target_sum

    if prev_w is None or len(prev_w) == 0:
        return target.sort_values(ascending=False)

    prev = pd.Series(prev_w).astype(float)
    union_index = prev.index.union(target.index)
    prev = prev.reindex(union_index, fill_value=0.0)
    working = target.reindex(union_index, fill_value=0.0)

    band = max(0.0, float(rebalance_band))
    if band > 0:
        thresholds = band * working.abs()
        diff = (working - prev).abs()
        keep_mask = diff <= thresholds
        working = working.where(~keep_mask, prev)

    working = working.clip(lower=0.0)
    if working.sum() > 0:
        working = working / working.sum()

    cap = max(0.0, float(turnover_cap))
    current_turnover = turnover(prev, working)
    if cap > 0 and current_turnover > cap:
        lam = cap / current_turnover
        blended = prev + lam * (working - prev)
        blended = blended.clip(lower=0.0)
        if blended.sum() > 0:
            working = blended / blended.sum()
        else:
            working = blended

    return working.sort_values(ascending=False)


def risk_adjust_scores(
    scores: pd.Series,
    returns_window: pd.DataFrame,
    *,
    window: int = 20,
    epsilon: float = EPSILON,
) -> pd.Series:
    """Scale scores by realised volatility to emphasise risk efficiency."""

    if scores is None or scores.empty:
        return pd.Series(dtype=float)

    vol = (
        returns_window.astype(float)
        .rolling(window, min_periods=max(2, window // 2))
        .std()
        .iloc[-1]
        if not returns_window.empty
        else pd.Series(dtype=float)
    )
    vol = vol.reindex(scores.index).fillna(vol.median() if not vol.empty else 0.0)
    denom = vol.abs().clip(lower=epsilon)
    adjusted = pd.Series(scores, dtype=float) / denom
    adjusted = adjusted.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return adjusted.sort_values(ascending=False)


def select_dynamic_topk(
    history: Iterable[pd.Series],
    *,
    min_k: int = 10,
    max_k: int = 30,
    default_k: int = 20,
) -> int:
    """Determine an optimal Top-K based on historical score concentration."""

    min_k = int(max(1, min_k))
    max_k = int(max(min_k, max_k))
    default_k = int(min(max_k, max(min_k, default_k)))

    history_list = [pd.Series(s).dropna().sort_values(ascending=False) for s in history]
    history_list = [s for s in history_list if not s.empty]
    if not history_list:
        return default_k

    metrics: Dict[int, float] = {}
    for k in range(min_k, max_k + 1):
        diffs: List[float] = []
        for series in history_list:
            if len(series) < 2:
                continue
            top_vals = series.head(k)
            if top_vals.empty:
                continue
            mean_top = float(top_vals.mean())
            mean_all = float(series.mean()) if len(series) else 0.0
            diffs.append(mean_top - mean_all)
        if diffs:
            metrics[k] = float(np.mean(diffs))

    if not metrics:
        return default_k

    best_score = max(metrics.values())
    best_options = [k for k, v in metrics.items() if np.isclose(v, best_score)]
    return min(best_options) if best_options else default_k


def soft_diversified_selection(
    scores: pd.Series,
    corr_matrix: pd.DataFrame | None,
    *,
    k: int,
    penalty: float = 0.15,
) -> pd.Index:
    """Select tickers greedily with a soft correlation penalty."""

    working = pd.Series(scores).dropna().sort_values(ascending=False)
    if working.empty or k <= 0:
        return pd.Index([], dtype=object)

    selected: List[str] = []
    corr_matrix = corr_matrix if corr_matrix is not None else pd.DataFrame()
    for _ in range(min(k, len(working))):
        best_name = None
        best_score = -np.inf
        for name, base_score in working.items():
            if name in selected:
                continue
            penalty_term = 0.0
            if selected and not corr_matrix.empty and name in corr_matrix.index:
                correlations = corr_matrix.loc[name, corr_matrix.columns.intersection(selected)]
                if not correlations.empty:
                    penalty_term = float(correlations.mean())
            adjusted = float(base_score) - penalty * penalty_term
            if adjusted > best_score:
                best_score = adjusted
                best_name = name
        if best_name is None:
            break
        selected.append(best_name)
    return pd.Index(selected)


def tune_rebalance_band(
    history: Iterable[float],
    current_band: float,
    *,
    min_band: float = 0.20,
    max_band: float = 0.35,
    step: float = 0.05,
) -> float:
    """Auto-tune the rebalance band based on realised turnover history."""

    values = [float(v) for v in history if np.isfinite(v)]
    if not values:
        return float(min(max_band, max(min_band, current_band)))

    median_val = float(np.median(values))
    band = float(current_band)
    if median_val > 0.45:
        band += step
    elif median_val < 0.25:
        band -= step
    band = float(min(max_band, max(min_band, band)))
    return band


def apply_exposure_clamp(weights: pd.Series, multiplier: float) -> pd.Series:
    """Scale weights by a gross exposure multiplier, preserving direction."""

    mult = float(multiplier)
    if not np.isfinite(mult) or mult <= 0:
        return pd.Series(weights, dtype=float)
    w = pd.Series(weights).astype(float)
    return w * mult
