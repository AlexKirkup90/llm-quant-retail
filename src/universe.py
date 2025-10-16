from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd

from . import universe_registry
from .config import REF_DIR, RUNS_DIR
from .utils import load_sp500_symbols

LOGGER = logging.getLogger(__name__)

OHLCV_FILE = REF_DIR / "ohlcv_latest.csv"

MIN_PRICE = 3.0
MIN_ADV = 5_000_000
_MIN_COVERAGE = 0.7


@dataclass
class FilterMetadata:
    raw_count: int
    filtered_count: int
    reason: str
    filters_applied: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw_count": self.raw_count,
            "filtered_count": self.filtered_count,
            "reason": self.reason,
            "filters_applied": self.filters_applied,
        }


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


def _load_ohlcv_snapshot(min_rows: int) -> Tuple[pd.DataFrame, str]:
    if not OHLCV_FILE.exists():
        msg = f"Liquidity filters skipped: missing OHLCV snapshot at {OHLCV_FILE}."
        LOGGER.info(msg)
        return pd.DataFrame(), msg
    try:
        df = pd.read_csv(OHLCV_FILE)
    except Exception as exc:
        msg = f"Liquidity filters skipped: failed to load OHLCV snapshot ({exc})."
        LOGGER.warning(msg)
        return pd.DataFrame(), msg
    df["symbol"] = df["symbol"].str.upper().str.strip()
    if len(df) < max(1, min_rows):
        msg = (
            "Liquidity filters skipped: OHLCV snapshot too small "
            f"({len(df)} rows < {min_rows})."
        )
        LOGGER.info(msg)
        return pd.DataFrame(), msg
    return df, ""


def _extract_adv(snapshot: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "adv_usd" in snapshot.columns:
        return snapshot["adv_usd"], ""
    volume_candidates = [
        col
        for col in snapshot.columns
        if col.lower() in {"volume", "volume_30d", "avg_volume_30d", "average_volume_30d"}
    ]
    if volume_candidates:
        volume_col = volume_candidates[0]
        try:
            adv_series = snapshot["close"] * snapshot[volume_col]
            return adv_series, ""
        except Exception as exc:  # pragma: no cover - defensive
            return pd.Series(dtype="float64"), (
                f"Liquidity filters skipped: failed to compute ADV ({exc})."
            )
    return pd.Series(dtype="float64"), "Liquidity filters skipped: ADV data missing in snapshot."


def _apply_filters(base: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, FilterMetadata]:
    meta = FilterMetadata(
        raw_count=len(base),
        filtered_count=len(base),
        reason="",
        filters_applied=False,
    )
    if base.empty:
        meta.reason = "Base universe empty; skipping liquidity filters."
        return base, meta

    min_rows = universe_registry.expected_min_constituents(mode)
    snapshot, snapshot_reason = _load_ohlcv_snapshot(min_rows)
    if snapshot.empty:
        meta.reason = snapshot_reason
        return base, meta

    required_columns = {"symbol", "close"}
    if not required_columns.issubset(snapshot.columns):
        missing = sorted(required_columns - set(snapshot.columns))
        meta.reason = (
            "Liquidity filters skipped: OHLCV snapshot missing columns "
            f"{', '.join(missing)}."
        )
        LOGGER.info(meta.reason)
        return base, meta

    adv_series, adv_reason = _extract_adv(snapshot)
    if adv_reason:
        meta.reason = adv_reason
        LOGGER.info(meta.reason)
        return base, meta

    working = base.copy()
    working["symbol"] = working["symbol"].str.upper().str.strip()
    snapshot = snapshot.copy()
    snapshot = snapshot[["symbol", "close"]].copy().rename(columns={"close": "last_price"})
    snapshot["adv_usd"] = adv_series

    merged = working.merge(snapshot, on="symbol", how="left")
    coverage_mask = merged["last_price"].notna() & merged["adv_usd"].notna()
    coverage = float(coverage_mask.mean()) if len(merged) else 0.0
    if coverage < _MIN_COVERAGE:
        meta.reason = (
            "Liquidity filters skipped: insufficient OHLCV coverage "
            f"({coverage:.0%} < {_MIN_COVERAGE:.0%})."
        )
        LOGGER.info(meta.reason)
        return base, meta

    mask = coverage_mask & (
        (merged["last_price"] >= MIN_PRICE) & (merged["adv_usd"] >= MIN_ADV)
    )
    dropped = merged.loc[~mask & coverage_mask, "symbol"].dropna().tolist()
    if dropped:
        LOGGER.info(
            "Filtered %d symbols below thresholds (price >= %.2f, ADV >= %.0f): %s",
            len(dropped),
            MIN_PRICE,
            MIN_ADV,
            ", ".join(sorted(dropped)),
        )

    filtered = merged.loc[mask].copy()
    if filtered.empty:
        meta.reason = (
            "Liquidity filters skipped: all symbols removed by thresholds; using raw universe."
        )
        LOGGER.warning(meta.reason)
        return base, meta

    meta.filters_applied = True
    meta.filtered_count = len(filtered)
    selected_columns = list(base.columns)
    for extra in ("last_price", "adv_usd"):
        if extra not in selected_columns:
            selected_columns.append(extra)
    filtered = filtered[selected_columns]
    return filtered.reset_index(drop=True), meta


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


def _load_and_filter_universe(
    mode: str, apply_filters: bool
) -> Tuple[pd.DataFrame, FilterMetadata, pd.DataFrame]:
    base = _load_base(mode)
    if base.empty:
        if mode != "SP500_MINI":
            raise universe_registry.UniverseRegistryError(
                f"Universe {mode} is empty after loading"
            )
        base = base.head(5)

    if not apply_filters:
        reason = "Liquidity filters bypassed via apply_filters=False."
        return base, FilterMetadata(
            raw_count=len(base),
            filtered_count=len(base),
            reason=reason,
            filters_applied=False,
        ), base

    filtered, meta = _apply_filters(base, mode)
    return filtered, meta, base


def load_universe_with_meta(
    mode: str, apply_filters: bool = True
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return the requested universe along with filter metadata."""

    mode = (mode or "SP500").upper()
    filtered, meta, base = _load_and_filter_universe(mode, apply_filters)
    _ensure_mini_cache(base)
    if not meta.filters_applied and filtered is not None and filtered.empty:
        filtered = base

    result = filtered.copy()
    if meta.filters_applied and result.empty:
        LOGGER.warning(
            "Liquidity filters produced an empty universe for mode=%s; using raw symbols.",
            mode,
        )
        result = base
        meta = FilterMetadata(
            raw_count=len(base),
            filtered_count=len(base),
            reason="Liquidity filters skipped: empty result; using raw universe.",
            filters_applied=False,
        )

    return result, meta.to_dict()


def load_universe(mode: str, apply_filters: bool = True) -> pd.DataFrame:
    """Load the desired universe and attach filter metadata in ``.attrs``."""

    normalized_mode = (mode or "SP500").upper()
    result, meta = load_universe_with_meta(normalized_mode, apply_filters=apply_filters)
    result.attrs["universe_filter_meta"] = meta
    symbols = result.get("symbol", [])
    _log_universe(normalized_mode, symbols)
    return result
