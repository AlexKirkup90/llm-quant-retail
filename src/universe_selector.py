from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from .universe_registry import expected_min_constituents as _expected_min_constituents
except Exception:  # pragma: no cover - fallback if registry import fails

    def _expected_min_constituents(name: str) -> int:
        return 1


_METRIC_COLUMNS = [
    "alpha",
    "sortino",
    "mdd",
    "hit_rate",
    "val_alpha",
    "val_sortino",
    "coverage",
    "turnover_cost",
]


@dataclass
class UniverseDecision:
    winner: str
    scores: Dict[str, float]
    probabilities: Dict[str, float]
    rationale: str
    parameters: Dict[str, float]
    metrics: pd.DataFrame
    as_of: str
    spec_version: str
    lookback_weeks: int
    min_weeks: int

    def to_log_record(self) -> Dict[str, object]:
        return {
            "as_of": self.as_of,
            "spec": self.spec_version,
            "winner": self.winner,
            "scores": self.scores,
            "probabilities": self.probabilities,
            "rationale": self.rationale,
            "lookback_weeks": self.lookback_weeks,
            "min_weeks": self.min_weeks,
        }


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def compute_universe_metrics(
    history_df: pd.DataFrame, lookback_weeks: int, min_weeks: int
) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        empty = pd.DataFrame(columns=_METRIC_COLUMNS + ["n_weeks"])
        return empty.astype({c: float for c in _METRIC_COLUMNS}, errors="ignore")

    df = history_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["universe", "date"])
    else:
        df = df.sort_values("universe")

    grouped: Dict[str, pd.DataFrame] = {
        universe: group.tail(lookback_weeks if lookback_weeks > 0 else len(group))
        for universe, group in df.groupby("universe")
    }

    records: List[pd.Series] = []
    for universe, group in grouped.items():
        stats = {}
        for col in _METRIC_COLUMNS:
            if col in group.columns:
                stats[col] = group[col].dropna().mean()
            else:
                stats[col] = np.nan
        stats["n_weeks"] = float(len(group))
        records.append(pd.Series(stats, name=universe))

    metrics_df = pd.DataFrame(records)
    if not metrics_df.empty:
        metrics_df.index.name = "universe"
        metrics_df = metrics_df.astype({"n_weeks": float})
    return metrics_df


def score_universes(
    metrics_df: pd.DataFrame, weights: Dict[str, float], temperature: float
) -> Tuple[pd.Series, pd.Series]:
    if metrics_df is None or metrics_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    weights = weights or {}
    df = _ensure_columns(metrics_df.copy(), _METRIC_COLUMNS)

    alpha = df["alpha"].fillna(0.0)
    sortino = df["sortino"].fillna(0.0)
    mdd = df["mdd"].fillna(0.0)
    coverage = df["coverage"].fillna(0.0)
    turnover_cost = df["turnover_cost"].fillna(0.0)

    scores = (
        weights.get("alpha", 0.0) * alpha
        + weights.get("sortino", 0.0) * sortino
        - weights.get("mdd", 0.0) * mdd
        + weights.get("coverage", 0.0) * coverage
        - weights.get("turnover", 0.0) * turnover_cost
    )

    if temperature is None or temperature <= 0:
        temperature = 1.0

    arr = scores.to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        probs = np.full_like(arr, 1.0 / len(arr), dtype=float)
    else:
        arr = np.where(np.isfinite(arr), arr, 0.0)
        scaled = arr / temperature
        max_val = np.max(scaled) if scaled.size else 0.0
        exps = np.exp(scaled - max_val)
        denom = exps.sum()
        if denom <= 0:
            probs = np.full_like(exps, 1.0 / len(exps))
        else:
            probs = exps / denom
    probabilities = pd.Series(probs, index=scores.index, dtype=float)
    return scores.astype(float), probabilities


def _load_history(metrics_history_path: Path) -> pd.DataFrame:
    if not metrics_history_path.exists():
        return pd.DataFrame()
    try:
        data = json.loads(metrics_history_path.read_text())
    except json.JSONDecodeError:
        return pd.DataFrame()
    if not isinstance(data, list):
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def _append_decision_log(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            log = json.loads(path.read_text())
            if not isinstance(log, list):
                log = []
        except json.JSONDecodeError:
            log = []
    else:
        log = []
    log.append(record)
    path.write_text(json.dumps(log, indent=2, sort_keys=True))


def choose_universe(
    candidates: List[str],
    constraints: Dict[str, Dict[str, float]],
    registry_fn: Callable[[str], pd.DataFrame],
    metrics_history_path: Path,
    spec: Dict[str, object],
    as_of: date,
) -> Dict[str, object]:
    selection_cfg = (spec or {}).get("universe_selection", {})
    lookback_weeks = int(selection_cfg.get("lookback_weeks", 8))
    min_weeks = int(selection_cfg.get("min_weeks", 4))
    weights = selection_cfg.get("weights", {}) or {}
    temperature = float(selection_cfg.get("temperature", 1.0))

    history_df = _load_history(metrics_history_path)
    if not history_df.empty:
        history_df = history_df[history_df.get("universe").isin(candidates)]

    metrics_df = compute_universe_metrics(history_df, lookback_weeks, min_weeks)
    metrics_df = metrics_df.reindex(candidates)
    metrics_df = _ensure_columns(metrics_df, _METRIC_COLUMNS + ["n_weeks"])
    metrics_df["n_weeks"] = metrics_df["n_weeks"].fillna(0).astype(float)

    min_cons_map = (constraints or {}).get("min_constituents", {})
    coverage_now: Dict[str, float] = {}
    for name in candidates:
        min_cons_val = min_cons_map.get(name)
        if not min_cons_val:
            min_cons_val = min_cons_map.get("default")
        if not min_cons_val:
            try:
                min_cons_val = _expected_min_constituents(name)
            except Exception:
                min_cons_val = 1
        min_cons = float(min_cons_val or 1)
        try:
            df = registry_fn(name)
            current_count = len(df) if df is not None else 0
        except Exception:
            current_count = 0
        coverage_now[name] = float(min(1.0, current_count / min_cons)) if min_cons else 0.0

    metrics_df["coverage"] = metrics_df.apply(
        lambda row: coverage_now.get(row.name, 0.0)
        if pd.isna(row.get("coverage"))
        else row.get("coverage"),
        axis=1,
    )
    metrics_df["turnover_cost"] = metrics_df["turnover_cost"].fillna(0.0)

    scores, probabilities = score_universes(metrics_df, weights, temperature)
    if scores.empty:
        scores = pd.Series(0.0, index=pd.Index(candidates, name="universe"))
        probabilities = pd.Series(
            [1.0 / len(candidates)] * len(candidates),
            index=pd.Index(candidates, name="universe"),
        )

    winner = str(probabilities.idxmax()) if not probabilities.empty else candidates[0]

    contributions = {}
    row = metrics_df.loc[winner]
    contributions["alpha"] = weights.get("alpha", 0.0) * float(row.get("alpha", 0.0))
    contributions["sortino"] = weights.get("sortino", 0.0) * float(
        row.get("sortino", 0.0)
    )
    contributions["mdd"] = -weights.get("mdd", 0.0) * float(row.get("mdd", 0.0))
    contributions["coverage"] = weights.get("coverage", 0.0) * float(
        row.get("coverage", 0.0)
    )
    contributions["turnover"] = -weights.get("turnover", 0.0) * float(
        row.get("turnover_cost", 0.0)
    )

    driver_labels = {
        "alpha": "alpha",
        "sortino": "risk-adjusted returns",
        "mdd": "drawdown control",
        "coverage": "coverage",
        "turnover": "turnover discipline",
    }
    sorted_drivers = sorted(
        contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    driver_bits: List[str] = []
    for key, value in sorted_drivers[:2]:
        if value == 0:
            continue
        direction = "positive" if value > 0 else "negative"
        driver_bits.append(f"{driver_labels.get(key, key)} ({direction} impact {value:.3f})")

    rationale_parts = [f"Selected {winner}"]
    if driver_bits:
        rationale_parts.append("; ".join(driver_bits))
    n_weeks = row.get("n_weeks", 0.0)
    if n_weeks < min_weeks:
        rationale_parts.append(
            f"history limited to {int(n_weeks)} of required {min_weeks} weeks"
        )
    rationale = " — ".join(rationale_parts)

    decision = UniverseDecision(
        winner=winner,
        scores=scores.fillna(0.0).round(6).to_dict(),
        probabilities=probabilities.fillna(0.0).round(6).to_dict(),
        rationale=rationale,
        parameters={
            "temperature": temperature,
            **{f"w_{k}": float(v) for k, v in weights.items()},
        },
        metrics=metrics_df,
        as_of=str(as_of),
        spec_version=str(spec.get("version", "")),
        lookback_weeks=lookback_weeks,
        min_weeks=min_weeks,
    )

    decisions_path = metrics_history_path.parent / "runs" / "universe_decisions.json"
    _append_decision_log(decisions_path, decision.to_log_record())

    return {
        "winner": decision.winner,
        "scores": decision.scores,
        "probabilities": decision.probabilities,
        "rationale": decision.rationale,
        "metrics": decision.metrics,
        "log_path": str(decisions_path),
        "parameters": decision.parameters,
        "lookback_weeks": decision.lookback_weeks,
        "min_weeks": decision.min_weeks,
    }
