#!/usr/bin/env python3
"""End-to-end bootstrap for data, caches, and learning artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src import dataops, universe_registry
from src.config import RUNS_DIR


SPEC_PATH = Path("spec/current_spec.json")


def _load_spec(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def ensure_feature_weights(spec: Dict[str, object]) -> Dict[str, object]:
    factors = spec.get("factor_features", []) if isinstance(spec, dict) else []
    out_path = RUNS_DIR / "feature_weights.json"
    if out_path.exists():
        return {"status": "existing", "path": str(out_path)}

    weights = {}
    if factors:
        weight = 1.0 / max(1, len(factors))
        weights = {f: weight for f in factors}

    payload = {
        "updated_at": None,
        "ema_lambda": 0.9,
        "rolling_weeks": 12,
        "ridge_alpha": 10.0,
        "weights": weights,
        "history": {},
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {"status": "created", "path": str(out_path), "factors": len(weights)}


def main() -> None:
    spec = _load_spec(SPEC_PATH)
    universes = spec.get("universes") if isinstance(spec, dict) else []
    summary: Dict[str, object] = {"spec_version": spec.get("version") if isinstance(spec, dict) else None}

    # Refresh registries
    if universes:
        summary["universes"] = universe_registry.refresh_all(force=True)

    # Ensure OHLCV snapshot and warm cache
    fallback = (spec.get("universe") or {}).get("fallback", "SP500_MINI") if isinstance(spec, dict) else "SP500_MINI"
    snapshot_meta = dataops.ensure_latest_ohlcv_snapshot(universe=fallback)
    snapshot = dataops.load_latest_ohlcv_snapshot()
    cache_meta = dataops.warm_price_cache_from_snapshot(snapshot, name="latest_prices")
    summary["ohlcv"] = snapshot_meta
    summary["cache"] = cache_meta

    # Seed feature weights file for iterative learning loops
    summary["feature_weights"] = ensure_feature_weights(spec)

    # Healthcheck payload for quick visibility
    summary["healthchecks"] = dataops.data_healthchecks(spec)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
