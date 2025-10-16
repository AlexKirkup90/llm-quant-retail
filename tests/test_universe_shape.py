import pandas as pd

from app import _resolve_symbols_from_universe
from src import universe


def test_universe_loader_handles_index_symbols(monkeypatch):
    frame = pd.DataFrame(
        {"name": ["Alpha", "Beta"], "sector": ["Tech", "Health"]},
        index=[" aapl ", "msft"],
    )

    monkeypatch.setattr(universe, "_load_base", lambda mode: frame.copy())
    monkeypatch.setattr(universe, "_ensure_mini_cache", lambda df: None)
    monkeypatch.setattr(universe, "_log_universe", lambda mode, symbols: None)
    monkeypatch.setattr(
        universe.universe_registry,
        "expected_min_constituents",
        lambda mode: 1,
    )

    loaded = universe.load_universe("TEST", apply_filters=False)
    assert list(loaded.columns[:3]) == ["symbol", "name", "sector"]
    assert loaded["symbol"].tolist() == ["AAPL", "MSFT"]

    symbols = _resolve_symbols_from_universe(loaded)
    assert symbols == ["AAPL", "MSFT"]
