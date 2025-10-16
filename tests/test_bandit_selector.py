import json

import numpy as np
import pandas as pd

from src import universe_selector


def test_bandit_prefers_higher_net_alpha(tmp_path):
    history = [
        {"date": "2024-01-05", "universe": "A", "net_alpha": 0.02},
        {"date": "2024-01-12", "universe": "A", "net_alpha": 0.03},
        {"date": "2024-01-19", "universe": "A", "net_alpha": 0.01},
        {"date": "2024-01-05", "universe": "B", "net_alpha": -0.02},
        {"date": "2024-01-12", "universe": "B", "net_alpha": -0.01},
        {"date": "2024-01-19", "universe": "B", "net_alpha": 0.0},
    ]
    metrics_history_path = tmp_path / "metrics_history.json"
    metrics_history_path.write_text(json.dumps(history))

    registry_frames = {
        "A": pd.DataFrame({"symbol": ["AAA", "BBB"]}),
        "B": pd.DataFrame({"symbol": ["CCC", "DDD"]}),
    }

    def registry(name: str) -> pd.DataFrame:
        return registry_frames[name]

    spec = {
        "version": "0.6",
        "universe_selection": {
            "bandit": {"enabled": True, "alpha_prior": 1.0, "beta_prior": 1.0, "min_observations": 2}
        },
    }

    decision = universe_selector.choose_universe(
        ["A", "B"],
        {},
        registry,
        metrics_history_path,
        spec,
        "2025-01-10",
        bandit_enabled=True,
        rng=np.random.default_rng(42),
    )

    assert decision["winner"] == "A"
    assert decision["bandit"]["active"] is True
    bandit_probs = decision["bandit"]["probabilities"]
    assert bandit_probs["A"] > bandit_probs["B"]
    posteriors = decision["bandit"]["posteriors"]
    assert posteriors["A"]["alpha"] > posteriors["B"]["alpha"]
