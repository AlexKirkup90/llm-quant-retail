import numpy as np
import pandas as pd

from src import regime


def test_market_risk_flag_triggers_on_downtrend_vol_and_breadth():
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    prices = pd.Series(np.linspace(400, 300, len(dates)), index=dates, name="SPY")
    breadth = pd.Series(np.linspace(0.6, 0.3, len(dates)), index=dates)
    flag = regime.market_risk_flag(prices, breadth=breadth)
    assert flag is True


def test_exposure_multiplier_clamps():
    assert regime.exposure_multiplier(True, minimum=0.7) == 0.7
    assert regime.exposure_multiplier(False, minimum=0.7) == 1.0
