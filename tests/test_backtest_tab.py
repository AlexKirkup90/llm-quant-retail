from streamlit.testing.v1 import AppTest


ALL_UNI = ["SP500_FULL", "R1000", "NASDAQ_100", "FTSE_350", "SP500_MINI"]


def test_backtest_tab_initialises(monkeypatch):
    monkeypatch.setattr("app.registry_list", lambda: list(ALL_UNI))

    app_test = AppTest.from_file("app.py", default_timeout=20)
    app_test.run()

    expected = [
        "bt__mode",
        "bt__years",
        "bt__tc",
        "bt__vol",
        "bt__sector_neutral",
        "bt__vel",
        "bt__resid",
        "bt__ic",
        "bt__regime",
    ]
    for key in expected:
        assert key in app_test.session_state
    assert not app_test.exception
