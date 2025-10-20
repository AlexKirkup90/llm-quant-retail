import numpy as np
import pandas as pd
import pytest

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

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-03-01", periods=40, freq="B")
    returns = pd.DataFrame(
        {
            "AAPL": rng.normal(0.01, 0.02, len(idx)),
            "MSFT": rng.normal(0.008, 0.015, len(idx)),
            "GOOGL": rng.normal(0.009, 0.018, len(idx)),
            "SPY": rng.normal(0.007, 0.012, len(idx)),
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


def test_beta_requires_min_overlap(caplog):
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    returns = pd.DataFrame({"A": np.linspace(0.01, 0.02, len(idx)), "SPY": np.linspace(0.0, 0.01, len(idx))}, index=idx)
    with caplog.at_level("WARNING"):
        betas = risk.estimate_asset_betas(returns, benchmark_col="SPY")
    assert (betas == 0.0).all()


def test_beta_warns_when_benchmark_missing(caplog):
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
    with caplog.at_level("WARNING"):
        betas = risk.estimate_asset_betas(returns, benchmark_col="SPY")
    assert "missing" in caplog.text.lower()
    assert (betas == 0.0).all()
