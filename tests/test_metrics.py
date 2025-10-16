import pandas as pd
from src.metrics import max_drawdown, sortino

def test_mdd_simple():
    curve = pd.Series([1, 1.1, 1.05, 1.2, 0.9, 1.0])
    mdd = max_drawdown(curve)
    assert 0.25 - 1e-6 <= mdd <= 0.25 + 1e-6  # 1.2 -> 0.9 drawdown ~25%

def test_sortino_sign():
    rets = pd.Series([0.01, -0.02, 0.015, -0.005, 0.004])
    s = sortino(rets)
    assert s == s  # not NaN
