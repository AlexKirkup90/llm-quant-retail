from __future__ import annotations

import numpy as np
import pandas as pd


def blend_weights(
    trend_w: pd.Series,
    mr_w: pd.Series,
    recent_perf: dict | None,
) -> pd.Series:
    """Blend trend and mean-reversion sleeves using softmax on Sharpe estimates."""

    trend_series = pd.Series(trend_w).astype(float)
    mr_series = pd.Series(mr_w).astype(float)
    union_index = trend_series.index.union(mr_series.index)
    trend_series = trend_series.reindex(union_index, fill_value=0.0)
    mr_series = mr_series.reindex(union_index, fill_value=0.0)

    recent_perf = recent_perf or {}
    trend_perf = float(recent_perf.get("trend", 0.0))
    mr_perf = float(recent_perf.get("mean_reversion", 0.0))
    scale = float(recent_perf.get("scale", 3.0)) if recent_perf else 3.0
    scale = max(1.0, scale)
    perf_array = np.array([trend_perf, mr_perf], dtype=float) * scale
    perf_array = perf_array - np.nanmax(perf_array)
    weights = np.exp(perf_array)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.array([0.5, 0.5], dtype=float)
    else:
        weights = weights / weights.sum()

    blended = trend_series * weights[0] + mr_series * weights[1]
    blended = blended.clip(lower=0.0)
    total = blended.sum()
    if total > 0:
        blended = blended / total
    return blended.sort_values(ascending=False)
