import json
from datetime import date, timedelta
import pandas as pd

from src.universe_selector import (
    choose_universe,
    compute_universe_metrics,
    score_universes,
)


def _make_history(universes, weeks):
    rows = []
    start = date(2024, 1, 1)
    for uni in universes:
        for idx in range(weeks):
            rows.append(
                {
                    "date": (start + timedelta(weeks=idx)).isoformat(),
                    "universe": uni,
                    "alpha": 0.01 * (idx + 1),
                    "sortino": 1.0 + 0.1 * idx,
                    "mdd": 0.05 + 0.01 * idx,
                    "hit_rate": 0.5 + 0.02 * idx,
                    "val_alpha": 0.005 * (idx + 1),
                    "val_sortino": 0.8 + 0.05 * idx,
                    "coverage": 0.7,
                    "turnover_cost": 0.001 * idx,
                }
            )
    return pd.DataFrame(rows)


def test_compute_universe_metrics_window():
    history = _make_history(["U1", "U2", "U3"], 10)
    metrics = compute_universe_metrics(history, lookback_weeks=4, min_weeks=6)
    assert set(metrics.index) == {"U1", "U2", "U3"}
    assert set(["alpha", "sortino", "mdd", "coverage", "turnover_cost", "n_weeks"]).issubset(
        metrics.columns
    )
    assert metrics.loc["U1", "n_weeks"] == 4.0


def test_score_universes_softmax_sum_one():
    metrics = pd.DataFrame(
        {
            "alpha": [0.02, 0.01],
            "sortino": [1.5, 0.9],
            "mdd": [0.1, 0.25],
            "coverage": [0.9, 0.6],
            "turnover_cost": [0.001, 0.002],
        },
        index=["U1", "U2"],
    )
    weights = {"alpha": 1.0, "sortino": 0.5, "mdd": 0.2, "coverage": 0.3, "turnover": 0.1}
    scores, probs = score_universes(metrics, weights, temperature=0.7)
    assert list(scores.index) == ["U1", "U2"]
    assert abs(probs.sum() - 1.0) < 1e-8
    assert (probs >= 0).all()


def test_choose_universe_logs_and_rationale(tmp_path):
    metrics_path = tmp_path / "metrics_history.json"
    history = [
        {
            "date": "2024-01-05",
            "universe": "U1",
            "alpha": 0.02,
            "sortino": 1.2,
            "mdd": 0.1,
            "hit_rate": 0.55,
            "val_alpha": 0.01,
            "val_sortino": 0.9,
            "turnover_cost": 0.001,
        },
        {
            "date": "2024-01-12",
            "universe": "U2",
            "alpha": 0.015,
            "sortino": 1.0,
            "mdd": 0.12,
            "hit_rate": 0.52,
            "val_alpha": 0.008,
            "val_sortino": 0.85,
        },
        {
            "date": "2024-01-19",
            "universe": "U3",
            "alpha": 0.01,
            "sortino": 0.95,
            "mdd": 0.09,
            "hit_rate": 0.51,
            "val_alpha": 0.007,
            "val_sortino": 0.83,
            "coverage": 0.65,
        },
    ]
    metrics_path.write_text(json.dumps(history))

    universe_sizes = {
        "U1": 120,
        "U2": 80,
        "U3": 95,
    }

    def registry_fn(name: str):
        count = universe_sizes.get(name, 0)
        return pd.DataFrame({"symbol": list(range(count))})

    spec = {
        "version": "v0.4",
        "universe_selection": {
            "lookback_weeks": 6,
            "min_weeks": 2,
            "weights": {
                "alpha": 0.6,
                "sortino": 0.2,
                "mdd": 0.3,
                "coverage": 0.2,
                "turnover": 0.1,
            },
            "temperature": 0.8,
            "constraints": {"min_constituents": {"U1": 100, "U2": 100, "U3": 90}},
        },
    }

    decision = choose_universe(
        ["U1", "U2", "U3"],
        spec["universe_selection"].get("constraints", {}),
        registry_fn,
        metrics_path,
        spec,
        date(2024, 3, 1),
    )

    assert decision["winner"] in {"U1", "U2", "U3"}
    assert isinstance(decision["rationale"], str) and decision["rationale"]
    assert abs(sum(decision["probabilities"].values()) - 1.0) < 1e-8

    log_file = tmp_path / "runs" / "universe_decisions.json"
    assert log_file.exists()
    log_entries = json.loads(log_file.read_text())
    assert log_entries[-1]["winner"] == decision["winner"]
    assert log_entries[-1]["spec"] == "v0.4"
