from pathlib import Path

import pandas as pd

from src import universe_registry

SNAPSHOT_DIR = Path("data/reference/snapshots")


def test_registry_known_universes():
    expected = {"SP500_MINI", "SP500_FULL", "R1000", "NASDAQ_100", "FTSE_350"}
    assert set(universe_registry.registry_list()) == expected


def test_parse_snapshots_offline():
    sp500 = universe_registry.fetch_sp500_full(SNAPSHOT_DIR / "sp500_wikipedia.html")
    nasdaq = universe_registry.fetch_nasdaq_100(SNAPSHOT_DIR / "nasdaq100_wikipedia.html")
    ftse = universe_registry.fetch_ftse_350(SNAPSHOT_DIR / "ftse350_wikipedia.html")
    r1000 = universe_registry.fetch_r1000(SNAPSHOT_DIR / "r1000_wikipedia.html")

    for frame in [sp500, nasdaq, ftse, r1000]:
        assert isinstance(frame, pd.DataFrame)
        assert set(frame.columns) == {"symbol", "name", "sector"}
        assert not frame.empty

    assert ftse["symbol"].str.endswith(".L").all()


def test_refresh_writes_csvs(tmp_path, monkeypatch):
    monkeypatch.setattr(universe_registry, "REF_DIR", tmp_path)

    providers = {
        "SP500_FULL": (universe_registry.fetch_sp500_full, "sp500_wikipedia.html"),
        "R1000": (universe_registry.fetch_r1000, "r1000_wikipedia.html"),
        "NASDAQ_100": (universe_registry.fetch_nasdaq_100, "nasdaq100_wikipedia.html"),
        "FTSE_350": (universe_registry.fetch_ftse_350, "ftse350_wikipedia.html"),
    }

    for name, (fetcher, filename) in providers.items():
        snapshot_file = SNAPSHOT_DIR / filename

        def _provider(_html_path=None, *, path=snapshot_file, func=fetcher):
            return func(path)

        monkeypatch.setattr(
            universe_registry._UNIVERSES[name],  # type: ignore[attr-defined]
            "provider",
            _provider,
            raising=False,
        )
        df, source = universe_registry.refresh_universe(name, force=True)
        assert source == "live"
        assert not df.empty
        expected_path = tmp_path / universe_registry._UNIVERSES[name].csv_filename  # type: ignore[attr-defined]
        assert expected_path.exists()
