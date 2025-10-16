from pathlib import Path

import pandas as pd
import pytest

from src import universe, universe_registry


@pytest.fixture
def sample_base(monkeypatch):
    base_df = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"]})

    def _mock_base(mode: str) -> pd.DataFrame:
        return base_df.copy()

    monkeypatch.setattr(universe, "_load_base", _mock_base)
    monkeypatch.setattr(universe, "_ensure_mini_cache", lambda df: None)
    monkeypatch.setattr(universe, "_log_universe", lambda mode, symbols: None)

    def _expected(name: str, spec_path: Path = Path("spec/current_spec.json")) -> int:
        return len(base_df)

    monkeypatch.setattr(universe_registry, "expected_min_constituents", _expected)
    return base_df


def test_filters_skipped_when_snapshot_missing(sample_base, monkeypatch, tmp_path):
    missing_snapshot = tmp_path / "missing.csv"
    monkeypatch.setattr(universe, "OHLCV_FILE", missing_snapshot)

    df, meta = universe.load_universe_with_meta("TEST", apply_filters=True)

    assert df.equals(sample_base)
    assert meta["filters_applied"] is False
    assert meta["raw_count"] == len(sample_base)
    assert meta["filtered_count"] == len(sample_base)
    assert "missing" in meta["reason"].lower()


def test_filters_applied_with_valid_snapshot(sample_base, monkeypatch, tmp_path):
    snapshot = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "close": [10.0, 2.0, 20.0],
            "adv_usd": [10_000_000.0, 10_000_000.0, 1_000_000.0],
        }
    )
    snapshot_path = tmp_path / "ohlcv.csv"
    snapshot.to_csv(snapshot_path, index=False)
    monkeypatch.setattr(universe, "OHLCV_FILE", snapshot_path)

    df, meta = universe.load_universe_with_meta("TEST", apply_filters=True)

    assert meta["filters_applied"] is True
    assert meta["reason"] == ""
    assert meta["raw_count"] == len(sample_base)
    assert meta["filtered_count"] < len(sample_base)
    assert "adv_usd" in df.columns
    assert "last_price" in df.columns
    assert set(df["symbol"]) == {"AAA"}


def test_app_bypass_checkbox(sample_base, monkeypatch):
    df, meta = universe.load_universe_with_meta("TEST", apply_filters=False)

    assert df.equals(sample_base)
    assert meta["filters_applied"] is False
    assert meta["filtered_count"] == meta["raw_count"]
    assert "bypass" in meta["reason"].lower()
