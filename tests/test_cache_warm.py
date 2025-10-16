import os
from pathlib import Path

import pandas as pd

from app import _apply_runtime_cap
from src import dataops


def test_warm_snapshot_triggers_bypass(tmp_path):
    snapshot_path = Path("data/reference/ohlcv_latest.csv")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"symbol": [f"T{i}" for i in range(8)], "close": 1.0})
    df.to_csv(snapshot_path, index=False)
    try:
        symbols = [f"T{i}" for i in range(10)]
        coverage = dataops.ohlcv_snapshot_coverage(dataops.load_latest_ohlcv_snapshot(), symbols)
        assert coverage == 0.8
        capped, applied = _apply_runtime_cap(symbols, cap=5, cache_warm=True, bypass_cap_if_warm=True)
        assert applied == 0
        assert len(capped) == len(symbols)
    finally:
        snapshot_path.unlink(missing_ok=True)
