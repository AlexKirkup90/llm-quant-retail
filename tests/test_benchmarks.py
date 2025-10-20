import numpy as np
import pandas as pd

from app import _apply_runtime_cap, _ensure_benchmark_symbol
from src.metrics import beta_vs_bench, default_benchmark


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


def test_benchmark_appended_after_cap():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    capped, _ = _apply_runtime_cap(symbols, cap=2, cache_warm=False, bypass_cap_if_warm=True)
    updated, bench = _ensure_benchmark_symbol(capped, "SP500_FULL")
    assert bench == "SPY"
    assert updated[-1] == "SPY"
    assert updated.count("SPY") == 1


def test_beta_vs_bench_returns_value_for_correlated_series():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    bench_returns = pd.Series(0.001 + 0.0005 * np.sin(np.linspace(0, 3, len(idx))), index=idx)
    bench_prices = (1 + bench_returns).cumprod()
    other_returns = bench_returns * 1.1
    other_prices = (1 + other_returns).cumprod()
    price_wide = pd.DataFrame({"SPY": bench_prices, "AAA": other_prices})
    port_returns = 0.6 * other_returns + 0.4 * bench_returns

    beta_value = beta_vs_bench(port_returns, price_wide, "SPY")
    assert pd.notna(beta_value)
    assert abs(beta_value) > 0.1


def test_beta_vs_bench_requires_overlap(caplog):
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    bench_prices = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    price_wide = pd.DataFrame({"SPY": bench_prices})
    port_returns = pd.Series(np.linspace(0.0, 0.01, 5), index=idx[:5])

    with caplog.at_level("WARNING"):
        beta_value = beta_vs_bench(port_returns, price_wide, "SPY")
    assert np.isnan(beta_value)
    assert "insufficient overlap" in caplog.text.lower()


def test_beta_vs_bench_handles_missing_benchmark(caplog):
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    price_wide = pd.DataFrame({"QQQ": np.linspace(100, 130, len(idx))}, index=idx)
    port_returns = pd.Series(np.linspace(0.0, 0.01, len(idx)), index=idx)

    with caplog.at_level("WARNING"):
        beta_value = beta_vs_bench(port_returns, price_wide, "SPY")
    assert np.isnan(beta_value)
    assert "missing" in caplog.text.lower()
