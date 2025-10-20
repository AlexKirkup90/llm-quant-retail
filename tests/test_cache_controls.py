from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src import dataops


def test_snapshot_button_sets_cache_warm(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "ohlcv_latest.csv"
    monkeypatch.setattr(dataops, "OHLCV_LATEST_PATH", snapshot_path)
    monkeypatch.setattr(dataops, "REFERENCE_DIR", snapshot_path.parent)

    metrics_payload = {"rows": 60, "cols": 30, "path": str(snapshot_path)}
    success_messages: list[str] = []

    def _fake_build(universe: str, out_path: str):
        assert out_path == str(snapshot_path)
        dates = pd.date_range("2024-01-01", periods=metrics_payload["rows"], freq="B")
        columns = [f"T{i:03d}" for i in range(metrics_payload["cols"])]
        frame = pd.DataFrame(1.0, index=dates, columns=columns)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path)
        return metrics_payload

    monkeypatch.setattr(dataops, "build_ohlcv_snapshot", _fake_build)
    monkeypatch.setattr("streamlit.success", lambda message, *_, **__: success_messages.append(message))

    app_test = AppTest.from_file("app.py", default_timeout=20)
    app_test.run()

    rebuild_button = app_test.button("weekly__rebuild_snapshot")
    rebuild_button.click().run()

    assert app_test.session_state["weekly__cache_warm"] is True
    assert any(
        "Warm price cache detected: 60 rows × 30 symbols" in message
        for message in success_messages
    )
