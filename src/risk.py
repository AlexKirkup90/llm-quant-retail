from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd


def transaction_cost(turnover: float, adv: float | pd.Series | Mapping[str, float], base_bps: float = 5) -> float:
    """Quadratic transaction cost model with ADV conditioning."""

    turnover = float(max(0.0, turnover))
    base = float(base_bps) / 10000.0

    if isinstance(adv, Mapping):
        values = [float(v) for v in adv.values() if v is not None]
        adv_val = float(np.median(values)) if values else 0.0
    elif isinstance(adv, pd.Series):
        values = adv.dropna().astype(float)
        adv_val = float(values.median()) if not values.empty else 0.0
    else:
        adv_val = float(adv or 0.0)

    if adv_val <= 0:
        return turnover * base

    impact = 1.0 + (turnover / adv_val)
    return turnover * base * impact


def annualised_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return float("nan")
    vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
    return float(vol)


def estimate_asset_betas(returns: pd.DataFrame, benchmark_col: str = "SPY") -> pd.Series:
    """Estimate single-factor betas versus the benchmark column."""

    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    if benchmark_col not in returns.columns:
        return pd.Series(0.0, index=returns.columns)

    bench = returns[benchmark_col]
    var_bench = bench.var(ddof=0)
    if not np.isfinite(var_bench) or np.isclose(var_bench, 0.0):
        return pd.Series(0.0, index=returns.columns)

    cov = returns.covwith(bench) if hasattr(returns, "covwith") else returns.apply(lambda col: np.cov(col, bench)[0, 1])
    betas = cov / var_bench
    return pd.Series(betas).fillna(0.0)


def _clip_scale(scale: float, floor: float = 0.0, ceiling: float = 3.0) -> float:
    return float(min(max(scale, floor), ceiling))


def apply_vol_target(weights: pd.Series, portfolio_returns: pd.Series, target_vol: float | None) -> Tuple[pd.Series, float]:
    if target_vol is None or target_vol <= 0:
        return weights, 1.0

    realised = annualised_volatility(portfolio_returns)
    if not np.isfinite(realised) or realised <= 0:
        return weights, 1.0

    scale = _clip_scale(target_vol / realised, floor=0.0, ceiling=3.0)
    return weights * scale, scale


def enforce_beta_limit(weights: pd.Series, asset_betas: pd.Series, limit: float | None) -> Tuple[pd.Series, float]:
    if limit is None or limit <= 0:
        return weights, 1.0

    asset_betas = asset_betas.reindex(weights.index).fillna(0.0)
    port_beta = float((weights * asset_betas).sum())
    beta_abs = abs(port_beta)
    if beta_abs <= limit or beta_abs == 0:
        return weights, 1.0

    scale = _clip_scale(limit / beta_abs, floor=0.0, ceiling=1.0)
    return weights * scale, scale


def drawdown_scale(equity_curve: pd.Series, threshold: float | None = 0.15, floor: float = 0.35) -> float:
    if threshold is None or threshold <= 0:
        return 1.0
    if equity_curve is None or equity_curve.empty:
        return 1.0

    curve = equity_curve.astype(float)
    peak = curve.cummax()
    dd = (curve / peak - 1.0).min()
    depth = abs(float(dd))
    if depth <= threshold:
        return 1.0

    overshoot = depth - threshold
    scale = 1.0 - overshoot / max(1e-6, 1.0 - threshold)
    return _clip_scale(scale, floor=floor, ceiling=1.0)


@dataclass
class RiskOverlayState:
    vol_scale: float = 1.0
    beta_scale: float = 1.0
    drawdown_scale: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "vol_scale": float(self.vol_scale),
            "beta_scale": float(self.beta_scale),
            "drawdown_scale": float(self.drawdown_scale),
        }


def apply_overlays(
    weights: pd.Series,
    portfolio_returns: pd.Series,
    asset_betas: pd.Series,
    equity_curve: pd.Series,
    target_vol: float | None = None,
    beta_limit: float | None = None,
    drawdown_limit: float | None = None,
) -> Tuple[pd.Series, RiskOverlayState]:
    state = RiskOverlayState()

    adj_weights, state.vol_scale = apply_vol_target(weights, portfolio_returns, target_vol)
    adj_weights, state.beta_scale = enforce_beta_limit(adj_weights, asset_betas, beta_limit)
    scale_dd = drawdown_scale(equity_curve, threshold=drawdown_limit) if drawdown_limit else 1.0
    state.drawdown_scale = scale_dd
    adj_weights = adj_weights * scale_dd

    return adj_weights, state
