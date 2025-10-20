import json

import pytest

from src import memory, universe_selector


def _patch_bandit_paths(monkeypatch, tmp_path):
    trace_path = tmp_path / "bandit_trace.jsonl"
    bak_path = tmp_path / "bandit_trace.bak.jsonl"
    monkeypatch.setattr(memory, "BANDIT_TRACE_PATH", trace_path, raising=False)
    monkeypatch.setattr(memory, "BANDIT_TRACE_BAK_PATH", bak_path, raising=False)
    monkeypatch.setattr(universe_selector.memory, "BANDIT_TRACE_PATH", trace_path, raising=False)
    monkeypatch.setattr(universe_selector.memory, "BANDIT_TRACE_BAK_PATH", bak_path, raising=False)


def test_bandit_persist(tmp_path, monkeypatch):
    _patch_bandit_paths(monkeypatch, tmp_path)

    updated = universe_selector.update_bandit(
        "SP500_FULL",
        {"alpha": 0.01, "sortino": 0.2},
        "2024-01-05",
        spec={"bandit": {"reward": "alpha"}},
    )
    assert pytest.approx(updated["SP500_FULL"][0]) == 2.0
    assert pytest.approx(updated["SP500_FULL"][1]) == 1.0

    posteriors = memory.load_bandit_posteriors()
    assert posteriors["SP500_FULL"] == pytest.approx((2.0, 1.0))

    updated = universe_selector.update_bandit(
        "SP500_FULL",
        {"alpha": -0.002, "sortino": -0.1},
        "2024-01-12",
        spec={"bandit": {"reward": "alpha"}},
    )
    assert pytest.approx(updated["SP500_FULL"][0]) == 2.0
    assert pytest.approx(updated["SP500_FULL"][1]) == 2.0

    posteriors = memory.load_bandit_posteriors()
    assert posteriors["SP500_FULL"] == pytest.approx((2.0, 2.0))


def test_bandit_reward_shaping(tmp_path, monkeypatch):
    _patch_bandit_paths(monkeypatch, tmp_path)

    trace_path = memory.BANDIT_TRACE_PATH
    records = []
    alpha, beta = 1.0, 1.0
    for idx in range(1, 13):
        alpha += 1.0
        records.append(
            {
                "as_of": f"2024-01-{idx:02d}",
                "choice": "SP500_FULL",
                "rewards": {"alpha": 0.0, "sortino": 0.05 * idx},
                "reward_value": 0.0,
                "posteriors": {"SP500_FULL": [alpha, beta]},
            }
        )
    trace_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    updated = universe_selector.update_bandit(
        "SP500_FULL",
        {"alpha": -0.01, "sortino": 0.9},
        "2024-01-19",
        spec={"bandit": {"reward": "alpha_sortino"}},
    )
    last_record = memory.load_bandit_trace(limit=1)[-1]
    assert last_record["reward_mode"] == "alpha_sortino"
    assert last_record["reward_value"] > 0.0  # sortino contribution rescues negative alpha
    posteriors = memory.load_bandit_posteriors()
    assert pytest.approx(updated["SP500_FULL"][0], rel=1e-6) == posteriors["SP500_FULL"][0]
