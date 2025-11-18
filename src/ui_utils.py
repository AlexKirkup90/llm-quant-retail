import json
import pathlib
from typing import List, Sequence, Tuple

import pandas as pd

from src.metrics import default_benchmark
from src.universe import ensure_universe_schema
from src.universe_registry import registry_list

PREFERRED = ["SP500_MINI", "SP500_FULL", "R1000", "NASDAQ_100", "FTSE_350"]
LABELS = {
    "SP500_MINI": "S&P 500 (Mini)",
    "SP500_FULL": "S&P 500",
    "R1000": "Russell 1000",
    "NASDAQ_100": "NASDAQ-100",
    "FTSE_350": "FTSE 350",
}

BENCHMARK_LABELS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "ISF.L": "ISF.L",
}


def _resolve_benchmark(universe_name: str) -> Tuple[str, str]:
    ticker = default_benchmark(universe_name)
    label = BENCHMARK_LABELS.get(ticker, ticker)
    return ticker, label


def _ensure_benchmark_symbol(symbols: Sequence[str], universe_name: str) -> Tuple[List[str], str]:
    bench = default_benchmark(universe_name)
    updated = list(symbols)
    if bench not in updated:
        updated.append(bench)
    return updated, bench


def get_universe_choices() -> List[str]:
    """Return preferred ordering combining registry and spec-defined universes."""

    choices = sorted(set(registry_list()))
    spec_path = pathlib.Path("spec/current_spec.json")
    try:
        spec = json.loads(spec_path.read_text())
        choices = sorted(set(choices) | set(spec.get("universes", [])))
    except Exception:
        pass
    ordered = [u for u in PREFERRED if u in choices]
    ordered.extend(u for u in choices if u not in PREFERRED)
    return ordered


def _clean_symbol_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
    )


def _resolve_symbols_from_universe(df: pd.DataFrame, universe_name: str = "UNKNOWN") -> List[str]:
    """Return deduplicated symbols regardless of column/index placement."""

    if df is None or df.empty:
        return []
    normalized = ensure_universe_schema(df, universe_name)
    cleaned = _clean_symbol_series(normalized["symbol"]).dropna()
    return pd.Index(cleaned).drop_duplicates().tolist()


def _apply_runtime_cap(
    symbols: Sequence[str],
    cap: int,
    cache_warm: bool,
    bypass_cap_if_warm: bool,
) -> Tuple[List[str], int]:
    """Return symbols subject to runtime cap plus the effective limit applied."""

    symbol_list = list(symbols)
    try:
        cap_value = int(cap)
    except Exception:
        cap_value = 0
    if cache_warm and bypass_cap_if_warm:
        return symbol_list, 0
    if cap_value <= 0:
        return symbol_list, 0
    effective = min(cap_value, len(symbol_list))
    return symbol_list[:effective], effective


def _format_cap_status(total_symbols: int, runtime_cap: int, bypass_active: bool) -> Tuple[int, str]:
    """Return the displayed count and cap label for the status line."""

    try:
        cap_value = int(runtime_cap)
    except Exception:
        cap_value = 0
    if bypass_active or cap_value <= 0:
        return total_symbols, "none"
    return min(total_symbols, cap_value), str(cap_value)


def _symbol_cache_key(universe_name: str) -> str:
    return f"prices_{str(universe_name).lower()}".replace("/", "_")
