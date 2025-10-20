import pandas as pd

import src.universe_registry as registry


def test_load_universe_normalized_with_columns(monkeypatch):
    df = pd.DataFrame({"Ticker": ["spy", "aapl"], "Security": ["SPDR", "Apple Inc"]})
    df.attrs["universe_filter_meta"] = {"raw_count": 2, "filtered_count": 2}

    def fake_loader(name: str):
        return df

    monkeypatch.setattr(registry, "load_universe", fake_loader)

    normalized = registry.load_universe_normalized("SP500_FULL")

    assert list(normalized.columns) == ["symbol", "name", "sector"]
    assert normalized["symbol"].tolist() == ["SPY", "AAPL"]
    assert normalized["name"].tolist() == ["SPDR", "Apple Inc"]
    assert "universe_filter_meta" in normalized.attrs


def test_load_universe_normalized_with_index(monkeypatch):
    df = pd.DataFrame({"sector": ["Technology", "Financials"]}, index=pd.Index(["msft", "jpm"], name="Ticker"))

    def fake_loader(name: str):
        return df

    monkeypatch.setattr(registry, "load_universe", fake_loader)

    normalized = registry.load_universe_normalized("R1000")

    assert list(normalized.columns) == ["symbol", "name", "sector"]
    assert normalized["symbol"].tolist() == ["MSFT", "JPM"]
    assert normalized["name"].tolist() == ["", ""]
    assert normalized["sector"].tolist() == ["Technology", "Financials"]
