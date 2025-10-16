from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _as_panel(base_features: pd.DataFrame) -> pd.DataFrame:
    if base_features is None or base_features.empty:
        return pd.DataFrame()
    if isinstance(base_features.index, pd.MultiIndex):
        panel = base_features.copy()
    elif "date" in base_features.columns:
        panel = base_features.set_index(["date", base_features.index.name or "symbol"])
    else:
        raise ValueError("base_features must have a MultiIndex or a 'date' column")
    panel.index = panel.index.set_names(["date", "symbol"])
    return panel.sort_index()


def _cross_sectional_zscore(values: pd.Series) -> pd.Series:
    if values.empty:
        return values

    def _z(col: pd.Series) -> pd.Series:
        std = col.std(ddof=0)
        if std and not np.isclose(std, 0.0):
            return (col - col.mean()) / std
        return pd.Series(0.0, index=col.index)

    return values.groupby(level=0).transform(_z)


def build_velocity_features(base_features: pd.DataFrame, windows: Dict[str, int]) -> pd.DataFrame:
    """Create velocity/acceleration style transforms for a feature panel."""

    panel = _as_panel(base_features)
    if panel.empty:
        return panel

    mom_window = int(max(1, windows.get("mom_accel", 6))) if isinstance(windows, dict) else 6
    vol_window = int(max(1, windows.get("vol_window", 5))) if isinstance(windows, dict) else 5
    delta_window = int(max(1, windows.get("delta_z", 4))) if isinstance(windows, dict) else 4

    output = pd.DataFrame(index=panel.index)

    if "mom_6m" in panel.columns:
        mom_diff = panel["mom_6m"].groupby(level=1).diff(mom_window)
        output["mom_6m_velocity"] = _cross_sectional_zscore(mom_diff.fillna(0.0))

    vol_col = None
    for candidate in ("realized_vol_20d", "vol_20d", "volatility_20d"):
        if candidate in panel.columns:
            vol_col = candidate
            break
    if vol_col:
        vol_change = panel[vol_col].groupby(level=1).diff(vol_window)
        output["vol_20d_compression"] = _cross_sectional_zscore(vol_change.fillna(0.0))

    for col, name in (("eps_rev_3m", "eps_rev_delta"), ("news_sent", "news_sent_delta")):
        if col in panel.columns:
            delta = panel[col].groupby(level=1).diff(delta_window)
            output[name] = _cross_sectional_zscore(delta.fillna(0.0))

    if "mom_6m" in panel.columns:
        cross_ranks = panel.groupby(level=0)["mom_6m"].rank(ascending=False, method="average")
        rank_change = cross_ranks.groupby(level=1).diff(delta_window)
        output["rank_stability"] = -_cross_sectional_zscore(rank_change.fillna(0.0))

    output = output.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return output
