from __future__ import annotations

import json
import logging
from typing import Iterable

import pandas as pd

from . import universe_registry
from .config import REF_DIR, RUNS_DIR
from .utils import load_sp500_symbols

LOGGER = logging.getLogger(__name__)

OHLCV_FILE = REF_DIR / "ohlcv_latest.csv"

MIN_PRICE = 3.0
MIN_ADV = 5_000_000


def _load_base(mode: str) -> pd.DataFrame:
    mode = mode.upper()
    if mode == "SP500":
        df = load_sp500_symbols()
        df["symbol"] = df["symbol"].str.upper().str.strip()
        return df
    if mode in universe_registry.registry_list():
        try:
            df = universe_registry.load_universe(mode)
        except universe_registry.UniverseRegistryError:
            if mode == "SP500_MINI":
                df = load_sp500_symbols().head(5).copy()
                df["symbol"] = df["symbol"].str.upper().str.strip()
                mini_path = REF_DIR / "sp500_mini.csv"
                mini_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(mini_path, index=False)
                return df
            raise
        df["symbol"] = df["symbol"].str.upper().str.strip()
        return df
    raise ValueError(f"Unknown universe mode: {mode}")


def _ensure_mini_cache(df: pd.DataFrame) -> None:
    mini_path = REF_DIR / "sp500_mini.csv"
    try:
        mini_path.parent.mkdir(parents=True, exist_ok=True)
        if not mini_path.exists():
            df.head(5).to_csv(mini_path, index=False)
    except Exception as exc:
        LOGGER.debug("Unable to cache mini universe: %s", exc)


def _load_ohlcv() -> pd.DataFrame:
    if not OHLCV_FILE.exists():
        LOGGER.warning("Missing OHLCV file at %s", OHLCV_FILE)
        return pd.DataFrame()
    try:
        df = pd.read_csv(OHLCV_FILE)
        df["symbol"] = df["symbol"].str.upper().str.strip()
        return df
    except Exception as exc:
        LOGGER.warning("Failed to load OHLCV data: %s", exc)
        return pd.DataFrame()


def _apply_filters(df: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if ohlcv.empty:
        LOGGER.info("No OHLCV data available; skipping liquidity filters")
        return df
    merged = df.merge(ohlcv[["symbol", "close", "adv_usd"]], on="symbol", how="left")
    mask = (merged["close"] >= MIN_PRICE) & (merged["adv_usd"] >= MIN_ADV)
    dropped = merged.loc[~mask, "symbol"].dropna().tolist()
    if dropped:
        LOGGER.info(
            "Filtered %d symbols below thresholds (price >= %.2f, ADV >= %.0f): %s",
            len(dropped),
            MIN_PRICE,
            MIN_ADV,
            ", ".join(sorted(dropped))
        )
    filtered = merged.loc[mask].copy()
    return filtered.reset_index(drop=True)


def _log_universe(mode: str, symbols: Iterable[str]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    symbols_list = list(symbols)
    payload = {
        "mode": mode,
        "count": len(symbols_list),
    }
    try:
        log_path = RUNS_DIR / "last_universe.json"
        log_path.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        LOGGER.debug("Failed to write last_universe.json: %s", exc)


def load_universe(mode: str = "SP500") -> pd.DataFrame:
    """Load the desired universe and apply liquidity/price filters.

    The registry-backed universes may raise
    :class:`src.universe_registry.UniverseRegistryError` when both live and
    cached data are unavailable. When liquidity filters remove every symbol we
    return the unfiltered base universe (or the cached ``SP500_MINI`` subset
    when that mode is explicitly requested).
    """

    mode = (mode or "SP500").upper()
    base = _load_base(mode)
    if base.empty:
        if mode != "SP500_MINI":
            raise universe_registry.UniverseRegistryError(
                f"Universe {mode} is empty after loading"
            )
        base = base.head(5)
    _ensure_mini_cache(base)
    ohlcv = _load_ohlcv()
    filtered = _apply_filters(base, ohlcv)
    if filtered.empty:
        if mode == "SP500_MINI":
            filtered = base
        else:
            LOGGER.warning(
                "All symbols filtered out for mode=%s; returning unfiltered base universe",
                mode,
            )
            filtered = base
    symbols = filtered.get("symbol", [])
    _log_universe(mode, symbols)
    return filtered
