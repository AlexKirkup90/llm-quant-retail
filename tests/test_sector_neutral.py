import numpy as np
import pandas as pd

from src import signals


def test_sector_neutralize_features_zero_means():
    features = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [5.0, 4.0, 3.0, 2.0],
        },
        index=["A", "B", "C", "D"],
    )
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"})

    neutralised = signals.sector_neutralize_features(features, sectors)

    for sector, members in sectors.groupby(sectors):
        sector_slice = neutralised.loc[members.index]
        assert np.isclose(sector_slice.mean().values, 0.0).all()


def test_score_current_sector_neutral_option():
    features = pd.DataFrame(
        {"f1": [1.0, 2.0], "f2": [3.0, 1.0]}, index=["A", "B"]
    )
    weights = pd.Series({"f1": 0.5, "f2": -0.5})
    sectors = pd.Series({"A": "Tech", "B": "Tech"})

    neutral_scores = signals.score_current(
        features,
        weights,
        sector_map=sectors,
        sector_neutral=True,
    )
    raw_scores = signals.score_current(features, weights)

    assert not neutral_scores.equals(raw_scores)
