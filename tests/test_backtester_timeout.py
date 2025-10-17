import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.backtester import BacktestConfig, WalkForwardBacktester


def _build_prices() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=90, freq="B")
    base = np.linspace(0, 1, len(dates))
    data = {
        "AAA": 50 + 2 * base,
        "BBB": 45 + 1.5 * base,
        "SPY": 200 + 0.5 * base,
    }
    prices = pd.DataFrame(data, index=dates)
    prices += np.cos(np.linspace(0, 3, len(dates)))[:, None]
    return prices


def test_backtester_respects_timeouts(caplog):
    prices = _build_prices()

    def strategy(window: pd.DataFrame) -> pd.Series:
        cols = [c for c in window.columns if c != "SPY"]
        weights = pd.Series(1.0, index=cols)
        return weights / weights.sum()

    def slow_residual(returns: pd.DataFrame, bench: Optional[pd.Series]) -> pd.Series:
        time.sleep(0.2)
        return pd.Series(dtype=float)

    def slow_regime(weights: pd.Series, start: pd.Timestamp, window: pd.DataFrame):
        time.sleep(0.2)
        return weights, {"trend": 1.0, "mean_reversion": 0.0}

    config = BacktestConfig(
        universe="TEST",
        spec_version="0.7",
        lookback_days=60,
        target_vol=0.1,
        beta_limit=1.0,
        drawdown_limit=0.2,
        base_bps=0.0,
        benchmark="SPY",
    )

    engine = WalkForwardBacktester(
        prices,
        strategy,
        config,
        residual_fn=slow_residual,
        regime_blend_fn=slow_regime,
        timeout_seconds=0.05,
    )

    caplog.set_level(logging.WARNING)
    result = engine.run(log=False)

    warnings = caplog.text
    assert "Residual regression timed out" in warnings
    assert "Regime blend timed out" in warnings
    assert not result["net_returns"].empty
