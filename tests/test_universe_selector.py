import json

import pandas as pd
import pytest

from src.universe_selector import (
    choose_universe,
    compute_universe_metrics,
    score_universes,
)


def _build_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    records = []
    for idx, dt in enumerate(dates):
        records.append(
            {
                "date": dt,
                "universe": "U1",
                "alpha": 0.10 + idx * 0.01,
                "sortino": 1.5 + 0.1 * idx,
                "mdd": 0.15 + 0.01 * idx,
                "hit_rate": 0.55,
                "val_alpha": 0.08,
                "val_sortino": 1.1,
                "coverage": 0.95,
                "turnover_cost": 0.01,
            }
        )
    for idx, dt in enumerate(dates[:2]):
        records.append(
            {
                "date": dt,
                "universe": "U2",
                "alpha": 0.05 + idx * 0.02,
                "sortino": 1.1 + 0.05 * idx,
                "mdd": 0.10 + 0.02 * idx,
                "hit_rate": 0.52,
                "val_alpha": 0.04,
                "val_sortino": 0.9,
                "turnover_cost": 0.015,
            }
        )
    for idx, dt in enumerate(dates[:4]):
        records.append(
            {
                "date": dt,
                "universe": "U3",
                "alpha": 0.03 + idx * 0.015,
                "sortino": 0.9 + 0.05 * idx,
                "mdd": 0.18 + 0.015 * idx,
                "hit_rate": 0.5,
                "val_alpha": 0.02,
                "val_sortino": 0.8,
                "coverage": None,
                "turnover_cost": 0.02,
            }
        )
    return pd.DataFrame.from_records(records)


def test_compute_universe_metrics_respects_lookback():
    history = _build_history()
    metrics = compute_universe_metrics(history, lookback_weeks=4, min_weeks=3)

    assert set(metrics.columns) == {
        "alpha",
        "sortino",
        "mdd",
        "hit_rate",
        "val_alpha",
        "val_sortino",
        "coverage",
        "turnover_cost",
        "n_weeks",
    }
    assert metrics.loc["U1", "n_weeks"] == 4
    # Last four alpha values for U1 start at 0.12
    expected_alpha = (0.12 + 0.13 + 0.14 + 0.15) / 4
    assert metrics.loc["U1", "alpha"] == pytest.approx(expected_alpha)
    # Sparse universe still returns an entry
    assert metrics.loc["U2", "n_weeks"] == 2


def test_score_universes_softmax_and_ranking(mocker):
    mocker.patch(
        "src.sentiment.compute_universe_momentum_scores",
        return_value={"U1": 0.05, "U2": 0.02, "U3": -0.01},
    )

    scores, probs = score_universes(["U1", "U2", "U3"])

    assert pytest.approx(probs.sum(), rel=1e-6) == 1.0
    assert probs.idxmax() == "U1"
    assert scores["U1"] > scores["U2"] > scores["U3"]


def test_choose_universe_writes_log_and_returns_decision(tmp_path, mocker):
    mocker.patch(
        "src.sentiment.compute_universe_momentum_scores",
        return_value={"U1": 0.05, "U2": 0.02, "U3": -0.01},
    )
    history = _build_history()
    metrics_history_path = tmp_path / "metrics_history.json"
    metrics_history_path.write_text(history.to_json(orient="records"))

    registry_frames = {
        "U1": pd.DataFrame({"symbol": list(range(12))}),
        "U2": pd.DataFrame({"symbol": list(range(6))}),
        "U3": pd.DataFrame({"symbol": list(range(20))}),
    }

    def _registry(name: str) -> pd.DataFrame:
        return registry_frames[name]

    spec = {
        "version": "0.4",
        "universe_selection": {
            "lookback_weeks": 4,
            "min_weeks": 3,
            "constraints": {"min_constituents": {"U1": 10, "default": 8}},
            "logging": {"fields": ["as_of", "winner", "coverage_now"]},
        },
    }

    candidates = ["U1", "U2", "U3"]
    decision = choose_universe(
        candidates,
        spec["universe_selection"].get("constraints", {}),
        _registry,
        metrics_history_path,
        spec,
        "2025-01-10",
    )

    assert decision["winner"] in candidates
    assert decision["rationale"].startswith("Selected")
    assert set(decision["scores"].keys()) == set(candidates)
    assert set(decision["probabilities"].keys()) == set(candidates)
    assert decision["coverage_now"]["U2"] == pytest.approx(6 / 8)

    log_path = tmp_path / "runs" / "universe_decisions.json"
    assert log_path.exists()
    history_log = json.loads(log_path.read_text())
    assert history_log[-1]["winner"] == decision["winner"]
    assert set(history_log[-1].keys()) == {"as_of", "winner", "coverage_now"}
