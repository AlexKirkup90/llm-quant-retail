from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _as_series(values: pd.Series | Sequence[float] | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype="float64")
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype="float64")


def coef_attribution(weights: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Return the top features ordered by absolute weight."""

    series = _as_series(weights).dropna()
    if series.empty:
        return pd.DataFrame(columns=["weight", "importance", "sign", "importance_pct"])

    df = pd.DataFrame({"feature": series.index, "weight": series.values})
    df["importance"] = df["weight"].abs()
    df["sign"] = np.sign(df["weight"]).astype(int)
    df = df.sort_values("importance", ascending=False).head(max(1, int(top_n)))
    total = df["importance"].sum()
    if total > 0:
        df["importance_pct"] = df["importance"] / total
    else:
        df["importance_pct"] = 0.0
    return df.set_index("feature")


def shap_like_contributions(
    features: pd.DataFrame,
    weights: pd.Series,
    top_features: int = 5,
) -> pd.DataFrame:
    """Approximate feature contributions per holding via weight × feature values."""

    if features is None or features.empty:
        return pd.DataFrame()

    weight_series = _as_series(weights)
    if weight_series.empty:
        return pd.DataFrame()

    aligned_cols = [c for c in features.columns if c in weight_series.index]
    if not aligned_cols:
        return pd.DataFrame()

    ordered_cols = (
        weight_series.reindex(aligned_cols).abs().sort_values(ascending=False).index
    )
    full_working = features.reindex(columns=aligned_cols).fillna(0.0)
    full_weights = weight_series.reindex(aligned_cols).fillna(0.0)
    full_contrib = full_working.mul(full_weights, axis=1)
    totals = full_contrib.sum(axis=1)

    if top_features > 0:
        display_cols = list(ordered_cols[: int(top_features)])
    else:
        display_cols = list(ordered_cols)

    contributions = full_contrib.reindex(columns=display_cols)
    contributions["total_contribution"] = totals
    return contributions


def ic_ema_table(ic_ema: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Return a table of top and bottom feature IC_EMA values."""

    series = _as_series(ic_ema).dropna()
    if series.empty:
        return pd.DataFrame(columns=["ic_ema", "group"])
    top = series.nlargest(max(1, top_n))
    bottom = series.nsmallest(max(1, top_n))
    combined = pd.concat([top, bottom])
    groups = ["top"] * len(top) + ["bottom"] * len(bottom)
    table = pd.DataFrame({"ic_ema": combined.values, "group": groups}, index=combined.index)
    return table
