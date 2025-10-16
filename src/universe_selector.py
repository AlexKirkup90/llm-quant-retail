from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - defensive import for optional dependency graph
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
    """Structured container for a selection outcome."""

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
    candidates: List[str]
    coverage_now: Dict[str, float]

    def to_log_record(self, fields: Iterable[str] | None = None) -> Dict[str, object]:
        base: Dict[str, object] = {
            "as_of": self.as_of,
            "spec": self.spec_version,
            "winner": self.winner,
            "candidates": self.candidates,
            "scores": self.scores,
            "probabilities": self.probabilities,
            "rationale": self.rationale,
            "lookback_weeks": self.lookback_weeks,
            "min_weeks": self.min_weeks,
            "parameters": self.parameters,
            "coverage_now": self.coverage_now,
        }
        if fields is None:
            return base
        selected: Dict[str, object] = {}
        for field in fields:
            if field in base:
                selected[field] = base[field]
        return selected


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure all requested columns exist, padding with NaNs where required."""

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def compute_universe_metrics(
    history_df: pd.DataFrame, lookback_weeks: int, min_weeks: int
) -> pd.DataFrame:
    """Aggregate trailing realised metrics for each universe.

    Parameters
    ----------
    history_df:
        Raw metrics history with at least ``date`` and ``universe`` columns.
    lookback_weeks:
        Number of most recent rows per-universe to aggregate.
    min_weeks:
        Minimum required rows for informational purposes; sub-minimum universes
        are still returned but labelled via ``n_weeks``.
    """

    if history_df is None or history_df.empty:
        empty = pd.DataFrame(columns=_METRIC_COLUMNS + ["n_weeks"])
        empty.index.name = "universe"
        return empty

    df = history_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["universe", "date"])
    else:
        df = df.sort_values("universe")

    window = max(int(lookback_weeks or 0), 0)
    grouped = []
    for name, group in df.groupby("universe"):
        if window > 0:
            group = group.tail(window)
        grouped.append((name, group))

    records: List[pd.Series] = []
    for universe_name, group in grouped:
        stats: Dict[str, float] = {}
        for col in _METRIC_COLUMNS:
            series = group[col] if col in group.columns else pd.Series(dtype=float)
            series = series.dropna()
            stats[col] = float(series.mean()) if not series.empty else float("nan")
        stats["n_weeks"] = float(len(group))
        records.append(pd.Series(stats, name=universe_name))

    metrics_df = pd.DataFrame(records)
    metrics_df.index.name = "universe"
    metrics_df = metrics_df.astype({"n_weeks": float})
    return metrics_df


def score_universes(
    metrics_df: pd.DataFrame, weights: Mapping[str, float], temperature: float
) -> Tuple[pd.Series, pd.Series]:
    """Score universes and convert to a probability distribution."""

    if metrics_df is None or metrics_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df = _ensure_columns(metrics_df.copy(), _METRIC_COLUMNS)

    def _w(key: str) -> float:
        return float(weights.get(key, 0.0)) if isinstance(weights, Mapping) else 0.0

    scores = (
        _w("alpha") * df["alpha"].fillna(0.0)
        + _w("sortino") * df["sortino"].fillna(0.0)
        - _w("mdd") * df["mdd"].fillna(0.0)
        + _w("coverage") * df["coverage"].fillna(0.0)
        - _w("turnover") * df["turnover_cost"].fillna(0.0)
    )

    temp = float(temperature or 0.0)
    if temp <= 0:
        temp = 1.0

    values = scores.to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    if values.size == 0:
        probabilities = np.array([], dtype=float)
    else:
        shifted = values / temp
        shifted -= np.max(shifted) if shifted.size else 0.0
        exps = np.exp(shifted)
        total = float(exps.sum())
        if total <= 0:
            probabilities = np.full_like(exps, 1.0 / len(exps))
        else:
            probabilities = exps / total

    probs_series = pd.Series(probabilities, index=scores.index, dtype=float)
    return scores.astype(float), probs_series


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
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df


def _append_decision_log(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            existing = json.loads(path.read_text())
            if isinstance(existing, list):
                log: List[Dict[str, object]] = existing
            else:
                log = []
        else:
            log = []
    except json.JSONDecodeError:
        log = []

    log.append(record)
    path.write_text(json.dumps(log, indent=2, sort_keys=True))


def _resolve_min_constituents(
    name: str,
    constraints: MutableMapping[str, object] | Mapping[str, object] | None,
) -> int:
    min_map = {}
    if constraints and isinstance(constraints, Mapping):
        raw = constraints.get("min_constituents")
        if isinstance(raw, Mapping):
            min_map = {k: int(v) for k, v in raw.items() if v is not None}

    for key in (name, "default"):
        if key in min_map and int(min_map[key]) > 0:
            return max(1, int(min_map[key]))

    return max(1, int(_expected_min_constituents(name)))


def _compute_coverage(
    candidates: Iterable[str],
    constraints: Mapping[str, object] | None,
    registry_fn: Callable[[str], pd.DataFrame],
) -> Dict[str, float]:
    coverage: Dict[str, float] = {}
    for name in candidates:
        min_required = float(_resolve_min_constituents(name, constraints))
        try:
            frame = registry_fn(name)
            current = float(len(frame)) if frame is not None else 0.0
        except Exception:  # pragma: no cover - defensive, exercised in tests via mocks
            current = 0.0
        ratio = 0.0 if min_required <= 0 else current / min_required
        coverage[name] = float(min(1.0, max(0.0, ratio)))
    return coverage


def choose_universe(
    candidates: List[str],
    constraints: Dict[str, object],
    registry_fn: Callable[[str], pd.DataFrame],
    metrics_history_path: Path,
    spec: Dict[str, object],
    as_of,
) -> Dict[str, object]:
    selection_cfg = (spec or {}).get("universe_selection", {}) or {}
    lookback_weeks = int(selection_cfg.get("lookback_weeks", selection_cfg.get("lookback", 8)))
    min_weeks = int(
        selection_cfg.get("min_weeks", selection_cfg.get("min_weeks_required", 4))
    )
    weights = selection_cfg.get("weights", {}) or {}
    temperature = float(selection_cfg.get("temperature", selection_cfg.get("softmax_temperature", 1.0)))

    history_df = _load_history(metrics_history_path)
    if not history_df.empty:
        history_df = history_df[history_df.get("universe").isin(candidates)]

    metrics_df = compute_universe_metrics(history_df, lookback_weeks, min_weeks)
    metrics_df = metrics_df.reindex(candidates)
    metrics_df.index.name = "universe"
    metrics_df = _ensure_columns(metrics_df, _METRIC_COLUMNS + ["n_weeks"])
    metrics_df["n_weeks"] = metrics_df["n_weeks"].fillna(0).astype(float)

    coverage_now = _compute_coverage(candidates, constraints, registry_fn)
    for name in metrics_df.index:
        if pd.isna(metrics_df.at[name, "coverage"]):
            metrics_df.at[name, "coverage"] = coverage_now.get(name, 0.0)
    metrics_df["turnover_cost"] = metrics_df["turnover_cost"].fillna(0.0)

    scores, probabilities = score_universes(metrics_df, weights, temperature)
    if scores.empty:
        scores = pd.Series(0.0, index=pd.Index(candidates, name="universe"))
        probabilities = pd.Series(
            [1.0 / len(candidates)] * len(candidates),
            index=pd.Index(candidates, name="universe"),
        )

    winner = str(probabilities.idxmax()) if not probabilities.empty else candidates[0]

    row = metrics_df.loc[winner]
    contributions = {
        "alpha": float(weights.get("alpha", 0.0)) * float(row.get("alpha", 0.0)),
        "sortino": float(weights.get("sortino", 0.0)) * float(row.get("sortino", 0.0)),
        "mdd": -float(weights.get("mdd", 0.0)) * float(row.get("mdd", 0.0)),
        "coverage": float(weights.get("coverage", 0.0)) * float(row.get("coverage", 0.0)),
        "turnover": -float(weights.get("turnover", 0.0))
        * float(row.get("turnover_cost", 0.0)),
    }

    driver_labels = {
        "alpha": "alpha",
        "sortino": "risk-adjusted returns",
        "mdd": "drawdown control",
        "coverage": "coverage",
        "turnover": "turnover discipline",
    }
    driver_bits: List[str] = []
    for key, value in sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[
        :2
    ]:
        if value == 0:
            continue
        direction = "positive" if value > 0 else "negative"
        driver_bits.append(f"{driver_labels.get(key, key)} ({direction} impact {value:.3f})")

    rationale_parts = [f"Selected {winner}"]
    if driver_bits:
        rationale_parts.append("; ".join(driver_bits))
    if float(row.get("n_weeks", 0.0)) < float(min_weeks):
        rationale_parts.append(
            f"history limited to {int(row.get('n_weeks', 0))} of required {min_weeks} weeks"
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
        candidates=list(candidates),
        coverage_now=coverage_now,
    )

    logging_cfg = selection_cfg.get("logging", {}) if isinstance(selection_cfg, Mapping) else {}
    fields = None
    if isinstance(logging_cfg, Mapping):
        requested = logging_cfg.get("fields")
        if isinstance(requested, list):
            fields = requested

    decisions_path = metrics_history_path.parent / "runs" / "universe_decisions.json"
    _append_decision_log(decisions_path, decision.to_log_record(fields))

    return {
        "winner": decision.winner,
        "scores": decision.scores,
        "probabilities": decision.probabilities,
        "rationale": decision.rationale,
        "metrics": decision.metrics,
        "candidates": decision.candidates,
        "coverage_now": decision.coverage_now,
        "log_path": str(decisions_path),
        "parameters": decision.parameters,
        "lookback_weeks": decision.lookback_weeks,
        "min_weeks": decision.min_weeks,
    }
