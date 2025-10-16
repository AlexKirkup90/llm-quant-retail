import pandas as pd

from src import portfolio


def test_rebalance_band_reduces_small_trades():
    prev = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    target = pd.Series({"A": 0.32, "B": 0.28, "C": 0.40})
    adjusted = portfolio.apply_turnover_controls(prev, target, turnover_cap=0.5, rebalance_band=0.25)
    # Small adjustments should keep weights close to previous due to banding
    assert abs(adjusted["A"] - prev["A"]) < 1e-6
    assert abs(adjusted["B"] - prev["B"]) < 1e-6


def test_turnover_cap_clamps_total_turnover():
    prev = pd.Series({"A": 0.6, "B": 0.4})
    target = pd.Series({"A": 0.0, "B": 1.0})
    adjusted = portfolio.apply_turnover_controls(prev, target, turnover_cap=0.4, rebalance_band=0.0)
    total_turnover = portfolio.turnover(prev, adjusted)
    assert total_turnover <= 0.401
    assert abs(adjusted.sum() - 1.0) < 1e-6
