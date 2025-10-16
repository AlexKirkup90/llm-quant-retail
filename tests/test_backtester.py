import json
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtester import BacktestConfig, WalkForwardBacktester


def _build_prices() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=160, freq="B")
    base = np.linspace(0, 1, len(dates))
    data = {
        "AAA": 100 + 5 * base,
        "BBB": 80 + 4 * base,
        "SPY": 200 + 3 * base,
    }
    prices = pd.DataFrame(data, index=dates)
    prices += np.sin(np.linspace(0, 6, len(dates)))[:, None]
    return prices


def test_walk_forward_backtester_runs(tmp_path, monkeypatch):
    prices = _build_prices()

    def strategy(window: pd.DataFrame) -> pd.Series:
        cols = [c for c in window.columns if c != "SPY"]
        weights = pd.Series(1.0, index=cols)
        return weights / weights.sum()

    config = BacktestConfig(
        universe="TEST",
        spec_version="0.6",
        lookback_days=60,
        target_vol=0.2,
        beta_limit=1.2,
        drawdown_limit=0.10,
        base_bps=5.0,
        benchmark="SPY",
    )

    adv = pd.Series(5_000_000, index=prices.columns)
    engine = WalkForwardBacktester(prices, strategy, config, adv=adv)
    result = engine.run(log=True)

    summary = result["summary"]
    assert summary["net_cagr"] > -0.05
    assert summary["max_drawdown"] >= 0.0
    assert summary["avg_turnover"] >= 0.0
    assert summary["total_cost"] >= 0.0
    assert summary["gross_vs_net_spread"] >= 0.0
    assert summary["cost_bps_weekly"] >= 0.0
    assert summary["overlay_scaler_mean"] > 0.0
    assert "overlay_scalers" in result and not result["overlay_scalers"].empty
    assert summary["gross_vs_net_spread"] <= summary["total_cost"] + 1e-4

    log_path = Path(result["log_path"])
    assert log_path.exists()
    payload = json.loads(log_path.read_text())
    assert payload["config"]["universe"] == "TEST"
    assert "gross" in payload["stats"]

    assert not result["net_returns"].empty
    assert result["equity_curve"].iloc[-1] > 0


def test_gross_net_spread_zero_when_costs_disabled():
    prices = _build_prices()

    def strategy(window: pd.DataFrame) -> pd.Series:
        cols = [c for c in window.columns if c != "SPY"]
        weights = pd.Series(1.0, index=cols)
        return weights / weights.sum()

    config = BacktestConfig(
        universe="TEST",
        spec_version="v0.6",
        lookback_days=60,
        target_vol=0.2,
        beta_limit=1.2,
        drawdown_limit=0.10,
        base_bps=0.0,
        benchmark="SPY",
    )

    engine = WalkForwardBacktester(prices, strategy, config, adv=None)
    result = engine.run(log=False)

    summary = result["summary"]
    assert pytest.approx(summary["gross_vs_net_spread"], abs=1e-9) == 0.0
    assert pytest.approx(summary["total_cost"], abs=1e-9) == 0.0
