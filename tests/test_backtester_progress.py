import numpy as np
import pandas as pd

from src.backtester import BacktestConfig, WalkForwardBacktester


def _build_prices(periods: int = 252 * 3) -> pd.DataFrame:
    dates = pd.date_range("2020-01-03", periods=periods, freq="B")
    base = np.linspace(0, 1, len(dates))
    data = {
        "AAA": 100 + 2 * base,
        "BBB": 80 + 3 * base,
        "CCC": 60 + 4 * base,
        "SPY": 200 + 1.5 * base,
    }
    prices = pd.DataFrame(data, index=dates)
    prices += np.sin(np.linspace(0, 4, len(dates)))[:, None]
    return prices


def test_backtester_logs_progress(capsys):
    prices = _build_prices()

    def strategy(window: pd.DataFrame) -> pd.Series:
        cols = [c for c in window.columns if c != "SPY"]
        weights = pd.Series(1.0, index=cols)
        return weights / weights.sum()

    config = BacktestConfig(
        universe="TEST",
        spec_version="0.7",
        lookback_days=126,
        target_vol=0.15,
        beta_limit=1.0,
        drawdown_limit=0.15,
        base_bps=0.0,
        benchmark="SPY",
    )

    engine = WalkForwardBacktester(prices, strategy, config)
    engine.run(log=False)

    output = capsys.readouterr().out
    assert "[backtester] window" in output
