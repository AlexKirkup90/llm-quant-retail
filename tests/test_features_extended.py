import numpy as np
import pandas as pd

from src import features


SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]


def test_eps_rev_3m_deterministic():
    series = features.eps_rev_3m(SYMBOLS)
    expected = pd.Series(
        {
            "AAPL": 1.0,
            "MSFT": -1.0,
            "GOOGL": 0.0,
            "AMZN": 0.0,
            "SPY": 0.0,
        },
        name="eps_rev_3m",
    )
    expected.index.name = "symbol"
    pd.testing.assert_series_equal(series.round(6), expected)
    assert np.isclose(series.mean(), 0.0, atol=1e-6)


def test_liq_adv_and_def_stability_deterministic():
    liq = features.liq_adv(SYMBOLS)
    stab = features.def_stability(SYMBOLS)

    expected_liq = pd.Series(
        {
            "AAPL": -0.046344,
            "MSFT": -1.923253,
            "GOOGL": 0.704786,
            "AMZN": 0.59812,
            "SPY": 0.666691,
        },
        name="liq_adv",
    )
    expected_stab = pd.Series(
        {
            "AAPL": 1.723073,
            "MSFT": 0.168248,
            "GOOGL": 0.098965,
            "AMZN": -0.916723,
            "SPY": -1.073563,
        },
        name="def_stability",
    )
    expected_liq.index.name = "symbol"
    expected_stab.index.name = "symbol"

    pd.testing.assert_series_equal(liq.round(6), expected_liq)
    pd.testing.assert_series_equal(stab.round(6), expected_stab)

    for series in (liq, stab):
        if len(series) > 1:
            assert np.isclose(series.mean(), 0.0, atol=1e-6)
            assert np.isclose(series.std(ddof=0), 1.0, atol=1e-6)


def test_news_sent_deterministic():
    sent = features.news_sent(SYMBOLS)
    expected = pd.Series(
        {
            "AAPL": -1.881211,
            "MSFT": 0.670741,
            "GOOGL": 0.6406,
            "AMZN": 0.753127,
            "SPY": -0.183258,
        },
        name="news_sent",
    )
    pd.testing.assert_series_equal(sent.round(6), expected)
    assert np.isclose(sent.mean(), 0.0, atol=1e-6)
    assert np.isclose(sent.std(ddof=0), 1.0, atol=1e-6)


def test_combine_features_includes_v02_columns():
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    price_data = pd.DataFrame({sym: 100 + np.linspace(0, 10, len(dates)) for sym in SYMBOLS}, index=dates)

    feats = features.combine_features(price_data)

    required_cols = {"mom_6m", "risk_beta", "eps_rev_3m", "news_sent", "liq_adv", "def_stability"}
    assert required_cols.issubset(set(feats.columns))

    subset = feats.loc[SYMBOLS]
    assert not subset[list(required_cols)].isna().any().any()
    # Columns should be z-scored
    for col in required_cols:
        if subset[col].std(ddof=0) > 0:
            assert np.isclose(subset[col].mean(), 0.0, atol=1e-6)

