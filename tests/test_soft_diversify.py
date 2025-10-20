import numpy as np
import pandas as pd

from src import portfolio


def test_soft_diversified_selection_reduces_high_corr():
    tickers = ["A", "B", "C", "D"]
    scores = pd.Series({"A": 1.0, "B": 0.95, "C": 0.9, "D": 0.85})
    corr = pd.DataFrame(
        [
            [1.0, 0.95, 0.2, 0.1],
            [0.95, 1.0, 0.25, 0.05],
            [0.2, 0.25, 1.0, 0.4],
            [0.1, 0.05, 0.4, 1.0],
        ],
        index=tickers,
        columns=tickers,
    )

    naive = scores.sort_values(ascending=False).head(3).index
    soft = portfolio.soft_diversified_selection(scores, corr, k=3, penalty=0.5)
    assert len(soft) == 3
    assert "C" in soft  # lower score but diversifies
    naive_pairs = corr.loc[naive, naive]
    soft_pairs = corr.loc[soft, soft]
    assert float(soft_pairs.where(~np.eye(3, dtype=bool)).stack().mean()) < float(
        naive_pairs.where(~np.eye(3, dtype=bool)).stack().mean()
    )
