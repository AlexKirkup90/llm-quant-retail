import pandas as pd

from src import metrics
from src.signals import update_feature_ic_ema


def test_spearman_ic_and_hit_rate_basic():
    predicted = pd.Series([0.9, 0.1, 0.5], index=["A", "B", "C"])
    future = pd.Series([0.08, -0.02, 0.03], index=["A", "B", "C"])
    ic = metrics.spearman_ic(predicted, future)
    hit = metrics.hit_rate(predicted, future, top_frac=0.34)
    assert -1.0 <= ic <= 1.0
    assert 0.0 <= hit <= 1.0


def test_append_and_load_ic_history(tmp_path):
    path = tmp_path / "ic_history.json"
    record = {"date": "2024-01-05", "ic": 0.2, "hit_rate": 0.6, "universe": "TEST"}
    metrics.append_ic_metric(record, path=path)
    history = metrics.load_ic_history(path=path)
    assert len(history) == 1
    assert history.iloc[0]["ic"] == 0.2


def test_feature_ic_snapshot_and_ema():
    features = pd.DataFrame(
        {
            "feat1": [0.1, 0.3, 0.2],
            "feat2": [0.5, 0.2, 0.4],
        },
        index=["A", "B", "C"],
    )
    future = pd.Series([0.04, -0.01, 0.02], index=["A", "B", "C"])
    snapshot = metrics.feature_ic_snapshot(features, future)
    ema = update_feature_ic_ema(pd.Series({"feat1": 0.0, "feat2": 0.1}), snapshot, ema_lambda=0.8)
    assert set(snapshot.index) == {"feat1", "feat2"}
    assert ema.index.equals(snapshot.index)
    assert (ema.abs() <= 1.0).all()
