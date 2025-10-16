import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src import signals


def _make_dates(n):
    base = datetime(2024, 1, 5)
    return [base + timedelta(days=7 * i) for i in range(n)]


def test_fit_rolling_ridge_applies_smoothing_and_scalers(monkeypatch, tmp_path):
    dates = _make_dates(2)
    tickers = ["AAA", "BBB", "CCC"]
    returns = pd.DataFrame(
        [[0.02, -0.01, 0.015], [0.03, 0.01, -0.005]],
        index=pd.to_datetime(dates),
        columns=tickers,
    )
    returns.attrs["vol_regime"] = "high"

    panels = {}
    for ts in dates:
        frame = pd.DataFrame(
            {
                "mom_6m": [1.0, 0.5, 0.3],
                "risk_beta": [0.8, 1.2, 0.7],
                "quality_roic": [0.4, 0.6, 0.5],
                "def_stability": [0.3, 0.9, 0.4],
                "eps_rev_3m": [0.2, 0.1, 0.05],
            },
            index=tickers,
        )
        frame.attrs["analyst_scores"] = [0.6, 0.4, 0.5]
        panels[pd.Timestamp(ts)] = frame

    weights_sequence = [
        pd.Series(
            {
                "mom_6m": 1.0,
                "risk_beta": 0.0,
                "quality_roic": 0.0,
                "def_stability": 0.0,
                "eps_rev_3m": 0.0,
            }
        ),
        pd.Series(
            {
                "mom_6m": 0.0,
                "risk_beta": 1.0,
                "quality_roic": 0.0,
                "def_stability": 0.0,
                "eps_rev_3m": 0.0,
            }
        ),
    ]

    def fake_fit_ridge(features_hist, fwd_returns, alpha=signals.RIDGE_ALPHA):
        return weights_sequence.pop(0)

    monkeypatch.setattr(signals, "fit_ridge", fake_fit_ridge)

    # Redirect RUNS_DIR to temp to avoid polluting repo
    monkeypatch.setattr(signals, "RUNS_DIR", tmp_path)

    weights = signals.fit_rolling_ridge(returns, panels, weeks=signals.ROLLING_WEEKS)

    assert pytest.approx(weights["mom_6m"], rel=1e-5) == 0.9043062201
    assert pytest.approx(weights["risk_beta"], rel=1e-5) == 0.0956937799
    assert weights["quality_roic"] == 0.0
    assert weights["def_stability"] == 0.0

    out_file = tmp_path / "feature_weights.json"
    assert out_file.exists()
    saved = json.loads(out_file.read_text())
    assert saved["vol_regime"] == "high"
    assert saved["market_sentiment"] == "pos"
    assert "mom_6m" in saved["weights"]
