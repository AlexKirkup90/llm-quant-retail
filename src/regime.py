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


def _moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(5, window // 2)).mean()


def _realised_volatility(returns: pd.Series, lookback: int = 20) -> pd.Series:
    return returns.rolling(lookback, min_periods=max(5, lookback // 2)).std()


def market_risk_flag(
    spy_prices: pd.Series | pd.DataFrame,
    *,
    breadth: pd.Series | None = None,
    vol_window: int = 20,
    vol_history: int = 252,
    breadth_threshold: float = 0.45,
) -> bool:
    """Return True when multiple market stress signals are active."""

    if isinstance(spy_prices, pd.DataFrame):
        if "SPY" in spy_prices.columns:
            price_series = spy_prices["SPY"].dropna()
        else:
            price_series = spy_prices.iloc[:, 0].dropna()
    else:
        price_series = pd.Series(spy_prices).dropna()

    if price_series.empty:
        return False

    price_series = price_series.astype(float)
    ma50 = _moving_average(price_series, 50)
    ma200 = _moving_average(price_series, 200)
    crossover_down = False
    if not ma50.dropna().empty and not ma200.dropna().empty:
        crossover_down = bool(ma50.iloc[-1] < ma200.iloc[-1])

    returns = price_series.pct_change().dropna()
    vol_series = _realised_volatility(returns, lookback=vol_window)
    vol_tail = vol_series.tail(vol_history)
    vol_trigger = False
    if not vol_tail.dropna().empty:
        threshold = float(vol_tail.quantile(0.80))
        current_vol = float(vol_series.iloc[-1]) if not vol_series.dropna().empty else 0.0
        vol_trigger = bool(current_vol > threshold and threshold > 0)

    breadth_trigger = False
    if breadth is not None:
        breadth_series = pd.Series(breadth).dropna().astype(float)
        if not breadth_series.empty:
            breadth_trigger = bool(breadth_series.iloc[-1] < breadth_threshold)

    return crossover_down or vol_trigger or breadth_trigger


def exposure_multiplier(flagged: bool, minimum: float = 0.7) -> float:
    """Return exposure multiplier given risk flag."""

    if not flagged:
        return 1.0
    return float(max(minimum, 0.0))
