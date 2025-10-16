import pandas as pd
import pytest

from src import explain


def test_coef_attribution_orders_by_importance():
    weights = pd.Series({"f1": 0.5, "f2": -0.2, "f3": 0.1})
    attr = explain.coef_attribution(weights, top_n=2)

    assert list(attr.index) == ["f1", "f2"]
    assert (attr["importance"] >= 0).all()
    assert pytest.approx(attr.loc["f1", "importance"], rel=1e-6) == 0.5


def test_shap_like_contributions_matches_weighting():
    features = pd.DataFrame(
        {"f1": [1.0, 2.0], "f2": [-1.0, 0.5], "f3": [0.0, 1.0]},
        index=["AAA", "BBB"],
    )
    weights = pd.Series({"f1": 0.4, "f2": -0.1, "f3": 0.2})

    contrib = explain.shap_like_contributions(features, weights, top_features=2)
    assert "total_contribution" in contrib.columns

    expected = (1.0 * 0.4) + (-1.0 * -0.1)
    assert contrib.loc["AAA", "total_contribution"] == pytest.approx(expected)
