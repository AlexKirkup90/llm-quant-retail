import pandas as pd
import pytest

from src import regime


def test_regime_blend_softmax():
    trend = pd.Series({"A": 0.6, "B": 0.4})
    mr = pd.Series({"A": 0.2, "B": 0.8})
    blended = regime.blend_weights(trend, mr, {"trend": 1.0, "mean_reversion": 0.2})
    assert blended.sum() == pytest.approx(1.0)
    assert blended["A"] > blended["B"]


def test_regime_blend_switches_when_performance_flips():
    trend = pd.Series({"A": 0.6, "B": 0.4})
    mr = pd.Series({"A": 0.2, "B": 0.8})
    blended = regime.blend_weights(trend, mr, {"trend": -0.5, "mean_reversion": 1.0})
    assert blended["B"] > blended["A"]
