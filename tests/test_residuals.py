import pandas as pd

from src.signals import residuals


def test_residual_returns_remove_market_component():
    dates = pd.date_range("2024-01-01", periods=4, freq="W")
    returns = pd.DataFrame(
        {
            "AAA": [0.02, 0.01, -0.01, 0.015],
            "BBB": [0.018, 0.009, -0.008, 0.014],
            "SPY": [0.02, 0.01, -0.01, 0.015],
        },
        index=dates,
    )
    sector_map = pd.Series({"AAA": "TECH", "BBB": "TECH"})
    bench = returns["SPY"]
    residual = residuals.compute_residual_returns(returns.drop(columns=["SPY"]), sector_map, bench)
    assert residual.index.tolist() == ["AAA", "BBB"]
    assert abs(residual.mean()) < 1e-6
    assert (residual.abs() < 0.05).all()


def test_residual_returns_handles_missing_sector():
    dates = pd.date_range("2024-01-01", periods=3, freq="W")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.0],
            "BBB": [0.0, 0.01, -0.01],
        },
        index=dates,
    )
    residual = residuals.compute_residual_returns(returns, None, pd.Series([0.0, 0.0, 0.0], index=dates))
    assert residual.index.tolist() == ["AAA", "BBB"]
    assert not residual.isna().any()
