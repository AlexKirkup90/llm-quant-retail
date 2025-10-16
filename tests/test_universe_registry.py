import json
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from src import universe_registry

SNAPSHOT_DIR = Path("data/reference/snapshots")


def test_registry_known_universes():
    expected = {"SP500_MINI", "SP500_FULL", "R1000", "NASDAQ_100", "FTSE_350"}
    assert set(universe_registry.registry_list()) == expected


def test_parse_snapshots_offline():
    snapshots = {
        "SP500_FULL": universe_registry.fetch_sp500_full(
            SNAPSHOT_DIR / "sp500_wikipedia.html"
        ),
        "NASDAQ_100": universe_registry.fetch_nasdaq_100(
            SNAPSHOT_DIR / "nasdaq100_wikipedia.html"
        ),
        "FTSE_350": universe_registry.fetch_ftse_350(
            html_path_100=SNAPSHOT_DIR / "ftse100_wikipedia.html",
            html_path_250=SNAPSHOT_DIR / "ftse250_wikipedia.html",
        ),
        "R1000": universe_registry.fetch_r1000(
            SNAPSHOT_DIR / "r1000_wikipedia.html"
        ),
    }

    for name, frame in snapshots.items():
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == ["symbol", "name", "sector"]
        assert not frame.empty
        assert len(frame) >= 3

    assert snapshots["FTSE_350"]["symbol"].str.endswith(".L").all()


def test_ftse350_composite_offline():
    df = universe_registry.fetch_ftse_350(
        html_path_100=SNAPSHOT_DIR / "ftse100_wikipedia.html",
        html_path_250=SNAPSHOT_DIR / "ftse250_wikipedia.html",
    )

    assert list(df.columns) == ["symbol", "name", "sector"]
    assert not df.empty
    assert len(df) >= 4
    assert df["symbol"].str.endswith(".L").all()
    assert df["symbol"].is_unique


def test_ftse_suffixed_and_string_types():
    df = universe_registry.fetch_ftse_350(
        html_path_100=SNAPSHOT_DIR / "ftse100_wikipedia.html",
        html_path_250=SNAPSHOT_DIR / "ftse250_wikipedia.html",
    )

    assert ptypes.is_string_dtype(df["symbol"])
    assert df["symbol"].equals(df["symbol"].str.upper())
    assert df["symbol"].str.endswith(".L").all()


def test_expected_min_constituents_defaults(tmp_path):
    empty_spec = tmp_path / "spec_empty.json"
    empty_spec.write_text(json.dumps({}))

    assert universe_registry.expected_min_constituents(
        "SP500_FULL", spec_path=empty_spec
    ) == 450
    assert universe_registry.expected_min_constituents(
        "SP500_MINI", spec_path=empty_spec
    ) == 5

    override_spec = tmp_path / "spec_override.json"
    override_spec.write_text(
        json.dumps(
            {
                "universe_selection": {
                    "constraints": {
                        "min_constituents": {
                            "SP500_MINI": 12,
                            "default": 42,
                        }
                    }
                }
            }
        )
    )

    assert universe_registry.expected_min_constituents(
        "SP500_MINI", spec_path=override_spec
    ) == 12
    assert universe_registry.expected_min_constituents(
        "UNKNOWN", spec_path=override_spec
    ) == 42


def test_normalize_handles_integer_columns():
    raw = pd.DataFrame(
        {
            "Ticker": [101, 202, 303, None],
            "Security": ["Alpha", " beta", None, ""],
            "Sector": [1, 2, None, 3],
        }
    )

    normalized = universe_registry._normalize_universe_df(raw, "R1000")

    assert len(normalized) == 3  # row with missing ticker dropped
    assert ptypes.is_string_dtype(normalized["symbol"])
    assert normalized["symbol"].equals(normalized["symbol"].str.upper())


def test_provider_handles_integer_headers(tmp_path):
    html = """
    <table>
        <thead>
            <tr><th>0</th><th>1</th><th>2</th></tr>
            <tr><th>Symbol</th><th>Company</th><th>Sector</th></tr>
        </thead>
        <tbody>
            <tr><td>aapl</td><td>Apple Inc</td><td>Technology</td></tr>
            <tr><td>msft</td><td>Microsoft</td><td>Technology</td></tr>
        </tbody>
    </table>
    """
    html_path = tmp_path / "nasdaq_inline.html"
    html_path.write_text(html, encoding="utf-8")

    df = universe_registry.fetch_nasdaq_100(html_path)

    assert list(df.columns) == ["symbol", "name", "sector"]
    assert df.loc[0, "symbol"] == "AAPL"
    assert df.loc[1, "symbol"] == "MSFT"


def test_refresh_writes_csvs(tmp_path, monkeypatch):
    monkeypatch.setattr(universe_registry, "REF_DIR", tmp_path)

    providers = {
        "SP500_FULL": (universe_registry.fetch_sp500_full, "sp500_wikipedia.html"),
        "R1000": (universe_registry.fetch_r1000, "r1000_wikipedia.html"),
        "NASDAQ_100": (universe_registry.fetch_nasdaq_100, "nasdaq100_wikipedia.html"),
        "FTSE_350": (
            universe_registry.fetch_ftse_350,
            ("ftse100_wikipedia.html", "ftse250_wikipedia.html"),
        ),
    }

    for name, (fetcher, filename) in providers.items():
        if name == "FTSE_350":
            file_100, file_250 = filename

            def _provider(
                _html_path=None,
                *,
                func=fetcher,
                path_100=SNAPSHOT_DIR / file_100,
                path_250=SNAPSHOT_DIR / file_250,
            ):
                return func(html_path_100=path_100, html_path_250=path_250)

        else:
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
