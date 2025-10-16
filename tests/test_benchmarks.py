import pandas as pd

from app import _apply_runtime_cap, _ensure_benchmark_symbol
from src import risk
from src.metrics import default_benchmark


def test_default_benchmark_mapping_contains_expected_universes():
    assert default_benchmark("SP500_FULL") == "SPY"
    assert default_benchmark("R1000") == "SPY"
    assert default_benchmark("NASDAQ_100") == "QQQ"
    assert default_benchmark("FTSE_350") == "ISF.L"


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


def test_sp500_weekly_appends_spy_and_uses_beta():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    updated, bench = _ensure_benchmark_symbol(symbols, "SP500_FULL")
    assert bench == "SPY"
    assert updated[-1] == "SPY"
    assert "SPY" in updated

    idx = pd.date_range("2024-03-01", periods=6, freq="B")
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.005, 0.012, 0.004, -0.003, 0.009],
            "MSFT": [0.008, 0.002, 0.011, -0.004, 0.006, 0.010],
            "GOOGL": [0.007, -0.001, 0.009, 0.003, -0.002, 0.008],
            "SPY": [0.006, -0.002, 0.007, 0.002, -0.001, 0.005],
        },
        index=idx,
    )
    betas = risk.estimate_asset_betas(returns, benchmark_col=bench)
    assert float(betas.loc["AAPL"]) != 0.0
    assert float(betas.loc["MSFT"]) != 0.0


def test_benchmark_not_duplicated_when_appended():
    symbols = ["AAPL", "QQQ", "MSFT"]
    updated, bench = _ensure_benchmark_symbol(symbols, "NASDAQ_100")
    assert updated.count(bench) == 1
