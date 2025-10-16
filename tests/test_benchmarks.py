import pandas as pd

from app import _resolve_benchmark, _apply_runtime_cap
from src import risk


def test_benchmark_mapping_contains_expected_universes():
    assert _resolve_benchmark("SP500_FULL")[0] == "SPY"
    assert _resolve_benchmark("R1000")[0] == "SPY"
    assert _resolve_benchmark("NASDAQ_100")[0] == "QQQ"
    assert _resolve_benchmark("FTSE_350")[0] == "ISF.L"


def test_runtime_cap_bypassed_when_cache_warm():
    symbols = ["AAA", "BBB", "CCC"]
    capped, applied = _apply_runtime_cap(symbols, cap=1, cache_warm=True, bypass_cap_if_warm=True)
    assert capped == symbols
    assert applied == 0


def test_benchmark_beta_column_alignment():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.0, 0.02, -0.01, 0.005],
            "BBB": [0.015, -0.005, 0.01, 0.0, 0.007],
            "QQQ": [0.012, -0.002, 0.008, -0.004, 0.006],
        },
        index=idx,
    )
    betas = risk.estimate_asset_betas(returns, benchmark_col="QQQ")
    assert "AAA" in betas.index and pd.notna(betas.loc["AAA"])
    assert "BBB" in betas.index and pd.notna(betas.loc["BBB"])


def test_benchmark_not_duplicated_when_appended():
    symbols = ["AAPL", "QQQ", "MSFT"]
    bench = _resolve_benchmark("NASDAQ_100")[0]
    unique = list(dict.fromkeys(symbols + [bench]))
    assert unique.count(bench) == 1
    assert unique[-1] == "MSFT"
