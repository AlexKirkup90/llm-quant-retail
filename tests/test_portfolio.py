import pandas as pd
from src.portfolio import turnover, enforce_turnover

def test_turnover():
    w1 = pd.Series({"A":0.5,"B":0.5})
    w2 = pd.Series({"A":0.2,"B":0.8})
    assert abs(turnover(w1, w2) - 0.6) < 1e-9

def test_enforce_turnover():
    w1 = pd.Series({"A":0.5,"B":0.5})
    w2 = pd.Series({"A":0.0,"B":1.0})
    w = enforce_turnover(w1, w2, 0.30)
    assert abs((w - w1).abs().sum() - 0.30) < 1e-6
