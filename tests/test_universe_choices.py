import importlib

import app


def test_get_universe_choices_includes_all_preferred(monkeypatch):
    monkeypatch.setattr(
        app,
        "registry_list",
        lambda: ["R1000", "SP500_FULL", "SP500_MINI", "NASDAQ_100", "FTSE_350"],
        raising=False,
    )

    importlib.reload(app)

    monkeypatch.setattr(
        app,
        "registry_list",
        lambda: ["R1000", "SP500_FULL", "SP500_MINI", "NASDAQ_100", "FTSE_350"],
        raising=False,
    )

    choices = app.get_universe_choices()

    assert "NASDAQ_100" in choices
    assert "FTSE_350" in choices

    preferred = [u for u in app.PREFERRED if u in choices]
    assert choices[: len(preferred)] == preferred
