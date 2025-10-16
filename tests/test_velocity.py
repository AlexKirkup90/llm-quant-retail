import pandas as pd

from src.features import velocity


def test_velocity_features_alignment():
    dates = pd.date_range("2024-01-01", periods=6, freq="W")
    symbols = ["AAA", "BBB"]
    records = {}
    for date in dates:
        df = pd.DataFrame(
            {
                "mom_6m": [0.1, 0.2],
                "realized_vol_20d": [0.3, 0.25],
                "eps_rev_3m": [0.05, 0.02],
                "news_sent": [0.1, -0.1],
            },
            index=symbols,
        )
        records[date] = df
    panel = pd.concat(records, names=["date", "symbol"])
    windows = {"mom_accel": 2, "vol_window": 2, "delta_z": 2}
    result = velocity.build_velocity_features(panel, windows)
    assert set(["mom_6m_velocity", "vol_20d_compression", "eps_rev_delta", "news_sent_delta", "rank_stability"]).issubset(result.columns)
    assert result.index.equals(panel.index)
    assert not result.isna().any().any()


def test_velocity_handles_missing_columns():
    dates = pd.date_range("2024-01-01", periods=5, freq="W")
    symbols = ["AAA", "BBB"]
    records = {}
    for date in dates:
        df = pd.DataFrame({"mom_6m": [0.1, 0.2]}, index=symbols)
        records[date] = df
    panel = pd.concat(records, names=["date", "symbol"])
    result = velocity.build_velocity_features(panel, {"mom_accel": 1})
    assert "mom_6m_velocity" in result.columns
    assert result.index.equals(panel.index)
