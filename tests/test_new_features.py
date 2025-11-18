import pandas as pd
import numpy as np
from src.portfolio import mean_variance_weights, calculate_momentum_score
from src.metrics import calculate_sector_attribution

def test_mean_variance_weights():
    prices = pd.DataFrame(np.random.rand(100, 4), columns=list('ABCD'))
    weights = mean_variance_weights(prices)
    assert isinstance(weights, pd.Series)
    assert np.isclose(weights.sum(), 1.0)
    assert all(weights >= 0)

def test_calculate_momentum_score():
    prices = pd.DataFrame(np.random.rand(100, 4), columns=list('ABCD'))
    scores = calculate_momentum_score(prices)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 4

def test_calculate_sector_attribution():
    weights = pd.DataFrame({'2025-01-01': [0.5, 0.5]}, index=['A', 'B'])
    returns = pd.DataFrame(np.random.rand(100, 2), columns=['A', 'B'])
    sector_map = pd.Series({'A': 'TECH', 'B': 'FINANCE'})
    attribution = calculate_sector_attribution(weights, returns, sector_map)
    assert isinstance(attribution, pd.DataFrame)
    assert 'TECH' in attribution.index
    assert 'FINANCE' in attribution.index
