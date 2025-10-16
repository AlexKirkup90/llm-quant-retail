import numpy as np
import pandas as pd

from src import dataops, features


SYMBOLS = ["AAPL", "MSFT", "GOOGL"]


def _make_price_history(symbols):
    dates = pd.date_range("2023-01-02", periods=260, freq="B")
    data = {}
    for i, sym in enumerate(symbols):
        base = 80 + i * 15
        data[sym] = base + np.linspace(0, 50, len(dates))
    return pd.DataFrame(data, index=dates)


def test_fetch_fundamentals_is_deterministic():
    first = dataops.fetch_fundamentals(SYMBOLS)
    second = dataops.fetch_fundamentals(SYMBOLS)
    pd.testing.assert_frame_equal(first, second)
    assert {"pe_ratio", "dividend_yield", "roe", "debt_to_equity"}.issubset(set(first.columns))


def test_fetch_news_sentiment_range_and_repeatable():
    first = dataops.fetch_news_sentiment(SYMBOLS)
    second = dataops.fetch_news_sentiment(SYMBOLS)
    pd.testing.assert_series_equal(first, second)
    assert first.between(-1.0, 1.0).all()


def test_combine_features_includes_fundamentals_and_sentiment():
    prices = _make_price_history(SYMBOLS + ["SPY"])
    fundamentals = dataops.fetch_fundamentals(SYMBOLS)
    sentiment = dataops.fetch_news_sentiment(SYMBOLS)
    feats = features.combine_features(prices, fundamentals=fundamentals, sentiment=sentiment)

    required_cols = {
        "mom_6m",
        "risk_beta",
        "value_score",
        "quality_score",
        "dividend_yield",
        "leverage_score",
        "news_sentiment",
    }
    assert required_cols.issubset(set(feats.columns))
    non_price_cols = [c for c in required_cols if c != "risk_beta"]
    assert not feats.loc["AAPL", non_price_cols].isna().any()


def test_news_sentiment_signal_standardization():
    sentiment = dataops.fetch_news_sentiment(SYMBOLS)
    signal = features.news_sentiment_signal(sentiment)
    assert signal.name == "news_sentiment"
    assert np.isclose(signal.mean(), 0.0, atol=1e-6)
    if len(signal) > 1:
        assert np.isclose(signal.std(ddof=0), 1.0)

