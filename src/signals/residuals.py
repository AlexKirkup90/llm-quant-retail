from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_design(sector_map: pd.Series, market_ret: float) -> pd.DataFrame:
    sectors = sector_map.fillna("UNKNOWN").astype(str)
    dummies = pd.get_dummies(sectors, drop_first=True, dtype=float)
    design = pd.DataFrame({"market": float(market_ret)}, index=sectors.index)
    if not dummies.empty:
        design = pd.concat([design, dummies], axis=1)
    design.insert(0, "intercept", 1.0)
    return design


def compute_residual_returns(
    rets: pd.DataFrame,
    sector_map: pd.Series | None,
    bench: pd.Series,
) -> pd.Series:
    """Regress weekly symbol returns on market/sector exposures and return residuals."""

    if rets is None or rets.empty:
        return pd.Series(dtype=float)
    returns = rets.copy()
    returns.index = pd.to_datetime(returns.index)
    latest_idx = returns.index[-1]
    cross_section = returns.loc[latest_idx].dropna()
    if cross_section.empty:
        return pd.Series(dtype=float)

    bench_series = pd.Series(bench).astype(float) if bench is not None else pd.Series(dtype=float)
    if bench_series.index.dtype != returns.index.dtype:
        bench_series.index = pd.to_datetime(bench_series.index)
    market_ret = float(bench_series.reindex(returns.index, method="ffill").fillna(0.0).loc[latest_idx]) if not bench_series.empty else 0.0

    if sector_map is None:
        sector_series = pd.Series("UNKNOWN", index=cross_section.index)
    else:
        sector_series = pd.Series(sector_map).reindex(cross_section.index).fillna("UNKNOWN")

    design = _prepare_design(sector_series, market_ret)
    design = design.reindex(cross_section.index).fillna(0.0)

    y = cross_section.astype(float).values
    X = design.values
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(X.shape[1])
    fitted = design.dot(coeffs)
    residuals = cross_section - fitted
    residuals.name = "residual_return"
    return residuals
