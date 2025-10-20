import numpy as np
import pandas as pd

from src import portfolio


def test_dynamic_topk_bounds_and_default():
    history = []
    idx = pd.Index([f"S{i}" for i in range(30)])
    for _ in range(12):
        values = pd.Series(np.linspace(1.0, 0.0, len(idx)), index=idx)
        history.append(values)
    k = portfolio.select_dynamic_topk(history, min_k=10, max_k=25, default_k=20)
    assert 10 <= k <= 25


def test_dynamic_topk_tie_prefers_lower():
    idx = pd.Index([f"S{i}" for i in range(5)])
    s1 = pd.Series([5, 4, 3, 2, 1], index=idx)
    s2 = pd.Series([5, 4, 3, 2, 1], index=idx)
    k = portfolio.select_dynamic_topk([s1, s2], min_k=2, max_k=4, default_k=3)
    assert k == 2


def test_risk_adjust_scores_handles_low_vol():
    scores = pd.Series({"A": 2.0, "B": 1.0})
    returns = pd.DataFrame({"A": [0.01] * 30, "B": [0.02] * 30})
    adj = portfolio.risk_adjust_scores(scores, returns)
    assert adj.loc["A"] > adj.loc["B"]
