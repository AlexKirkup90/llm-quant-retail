from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RUNS_DIR
from . import memory

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


def adaptive_top_k(universe_size: int) -> int:
    """Return the adaptive Top-K cut-off for candidate universes."""

    if universe_size <= 0:
        return 0
    base = max(25, min(60, int(round(0.10 * universe_size))))
    return max(1, min(int(universe_size), base))


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
    """Aggregate trailing realised metrics for each universe."""

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

    k = adaptive_top_k(len(scores))
    if 0 < k < len(scores):
        filtered_scores = scores.nlargest(k)
    else:
        filtered_scores = scores

    values = filtered_scores.to_numpy(dtype=float)
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

    probs_series = pd.Series(0.0, index=scores.index, dtype=float)
    if probabilities.size:
        probs_series.loc[filtered_scores.index] = probabilities
    elif len(probs_series) > 0:
        probs_series[:] = 1.0 / len(probs_series)

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


def _bandit_settings(
    selection_cfg: Mapping[str, object] | None, override_enabled: Optional[bool]
) -> Tuple[bool, float, float, int, Dict[str, Tuple[float, float]]]:
    cfg = selection_cfg.get("bandit", {}) if isinstance(selection_cfg, Mapping) else {}
    enabled = bool(cfg.get("enabled", False))
    if override_enabled is not None:
        enabled = bool(override_enabled)
    alpha_prior = float(cfg.get("alpha_prior", 1.0))
    beta_prior = float(cfg.get("beta_prior", 1.0))
    min_obs = int(cfg.get("min_observations", cfg.get("min_weeks", 3)))
    warm_priors = memory.load_bandit_posteriors()
    return enabled, alpha_prior, beta_prior, max(1, min_obs), warm_priors


def _compute_bandit_posteriors(
    history_df: pd.DataFrame,
    candidates: Iterable[str],
    alpha_prior: float,
    beta_prior: float,
    warm_priors: Mapping[str, Tuple[float, float]] | None,
) -> Tuple[Dict[str, Dict[str, float]], pd.Series, pd.Series]:
    if history_df is None or history_df.empty or "net_alpha" not in history_df.columns:
        empty = pd.Series(dtype=float)
        return {}, empty, empty

    posteriors: Dict[str, Dict[str, float]] = {}
    means: Dict[str, float] = {}
    counts: Dict[str, float] = {}

    net_alpha = pd.to_numeric(history_df.get("net_alpha"), errors="coerce")
    history_df = history_df.assign(net_alpha=net_alpha)

    for name in candidates:
        group = history_df.loc[history_df["universe"] == name, "net_alpha"].dropna()
        observations = int(len(group))
        successes = int((group > 0).sum())
        failures = int(observations - successes)
        prior_alpha, prior_beta = (warm_priors or {}).get(name, (alpha_prior, beta_prior))
        prior_alpha = float(prior_alpha)
        prior_beta = float(prior_beta)
        if prior_alpha <= 0:
            prior_alpha = alpha_prior
        if prior_beta <= 0:
            prior_beta = beta_prior
        alpha_post = float(prior_alpha + successes)
        beta_post = float(prior_beta + failures)
        total = alpha_post + beta_post
        mean = float(alpha_post / total) if total > 0 else 0.0
        posteriors[name] = {
            "alpha": alpha_post,
            "beta": beta_post,
            "observations": observations,
            "successes": successes,
        }
        means[name] = mean
        counts[name] = observations

    return posteriors, pd.Series(means), pd.Series(counts)


def _persist_bandit_state(posteriors: Dict[str, Dict[str, float]], active: bool) -> None:
    path = RUNS_DIR / "universe_bandit.json"
    payload = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "active": bool(active),
        "posteriors": {
            name: {key: float(value) for key, value in stats.items()} for name, stats in posteriors.items()
        },
    }
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except Exception:  # pragma: no cover - disk issues should not break selection
        pass


def _resolve_bandit_reward_mode(spec: Mapping[str, object] | None) -> str:
    if not isinstance(spec, Mapping):
        return "alpha"
    cfg = spec.get("bandit") if isinstance(spec.get("bandit"), Mapping) else {}
    if isinstance(cfg, Mapping):
        mode = str(cfg.get("reward", "alpha")).lower()
        if mode in {"alpha", "alpha_sortino"}:
            return mode
    return "alpha"


def _sortino_z_score(universe: str, current_sortino: float | None, window: int = 12) -> float:
    if current_sortino is None or not np.isfinite(current_sortino):
        return 0.0
    history = memory.load_bandit_trace(limit=200)
    values: List[float] = []
    for record in history:
        if record.get("choice") != universe:
            continue
        rewards = record.get("rewards") if isinstance(record.get("rewards"), Mapping) else {}
        if not isinstance(rewards, Mapping):
            continue
        value = rewards.get("sortino")
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        values.append(value_f)
    if not values:
        return 0.0
    tail = values[-window:]
    arr = np.array(tail, dtype=float)
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=0))
    if std <= 1e-8 or not np.isfinite(std):
        return 0.0
    mean = float(arr.mean())
    return float((float(current_sortino) - mean) / std)


def update_bandit(
    choice: str,
    reward: Mapping[str, float] | None,
    date: str | pd.Timestamp,
    *,
    spec: Mapping[str, object] | None = None,
    universes: Iterable[str] | None = None,
) -> Dict[str, Tuple[float, float]]:
    """Update the persistent bandit trace with a new observation."""

    reward = reward or {}
    try:
        alpha_val = float(reward.get("alpha", 0.0))
    except (TypeError, ValueError):
        alpha_val = 0.0
    sortino_raw = reward.get("sortino")
    try:
        sortino_val = float(sortino_raw)
    except (TypeError, ValueError):
        sortino_val = float("nan")
    mode = _resolve_bandit_reward_mode(spec)
    sortino_z = 0.0
    shaped = alpha_val
    if mode == "alpha_sortino":
        sortino_z = _sortino_z_score(str(choice), sortino_val)
        shaped = 0.7 * alpha_val + 0.3 * sortino_z

    success = shaped >= 0
    current_posteriors = memory.load_bandit_posteriors()
    base_names = set(current_posteriors.keys())
    base_names.add(str(choice))
    if universes is not None:
        base_names.update(str(u) for u in universes)

    updated: Dict[str, Tuple[float, float]] = {}
    for name in sorted(base_names):
        prior_alpha, prior_beta = current_posteriors.get(name, (1.0, 1.0))
        alpha_post = float(prior_alpha)
        beta_post = float(prior_beta)
        if name == str(choice):
            if success:
                alpha_post += 1.0
            else:
                beta_post += 1.0
        updated[name] = (alpha_post, beta_post)

    record = {
        "as_of": str(date),
        "choice": str(choice),
        "reward_mode": mode,
        "rewards": {
            "alpha": alpha_val,
            "sortino": sortino_val if np.isfinite(sortino_val) else None,
        },
        "reward_value": shaped,
        "sortino_z": sortino_z,
        "posteriors": {name: [a, b] for name, (a, b) in updated.items()},
    }
    memory.append_bandit_trace(record)
    return updated


def choose_universe(
    candidates: List[str],
    constraints: Dict[str, object],
    registry_fn: Callable[[str], pd.DataFrame],
    metrics_history_path: Path,
    spec: Dict[str, object],
    as_of,
    *,
    bandit_enabled: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
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

    (
        enabled,
        alpha_prior,
        beta_prior,
        min_obs,
        persistent_priors,
    ) = _bandit_settings(selection_cfg, bandit_enabled)
    bandit_info = {"active": False, "probabilities": {}, "posteriors": {}}
    if enabled:
        if persistent_priors:
            posteriors = {}
            means_dict: Dict[str, float] = {}
            counts_dict: Dict[str, float] = {}
            for name in candidates:
                alpha_post, beta_post = persistent_priors.get(name, (alpha_prior, beta_prior))
                alpha_post = float(alpha_post)
                beta_post = float(beta_post)
                if alpha_post <= 0:
                    alpha_post = alpha_prior
                if beta_post <= 0:
                    beta_post = beta_prior
                total = alpha_post + beta_post
                mean_val = float(alpha_post / total) if total > 0 else 0.0
                observations = max(0.0, float(alpha_post + beta_post - 2.0))
                successes = max(0.0, float(alpha_post - 1.0))
                posteriors[name] = {
                    "alpha": alpha_post,
                    "beta": beta_post,
                    "observations": int(round(observations)),
                    "successes": int(round(successes)),
                }
                means_dict[name] = mean_val
                counts_dict[name] = observations
            means = pd.Series(means_dict)
            counts = pd.Series(counts_dict)
        else:
            posteriors, means, counts = _compute_bandit_posteriors(
                history_df, candidates, alpha_prior, beta_prior, persistent_priors
            )
        bandit_series = pd.Series(1.0, index=pd.Index(candidates, name="universe"), dtype=float)
        if posteriors:
            bandit_info["posteriors"] = {
                name: {
                    "alpha": float(stats["alpha"]),
                    "beta": float(stats["beta"]),
                    "observations": int(stats.get("observations", 0)),
                    "successes": int(stats.get("successes", 0)),
                }
                for name, stats in posteriors.items()
            }
            bandit_active = len(counts) > 0 and counts.min() >= min_obs
            bandit_series = means.reindex(probabilities.index).fillna(0.0)
            if bandit_series.sum() <= 0:
                bandit_series = pd.Series(1.0, index=bandit_series.index)
            bandit_series = bandit_series / bandit_series.sum()
            if bandit_active:
                bandit_info["active"] = True
                combined = probabilities.fillna(0.0) * bandit_series
                total = combined.sum()
                if total <= 0:
                    combined = bandit_series
                    total = combined.sum()
                probabilities = combined / total if total > 0 else bandit_series
        bandit_series = bandit_series.reindex(probabilities.index).fillna(0.0)
        total_bandit = bandit_series.sum()
        if total_bandit > 0:
            bandit_series = bandit_series / total_bandit
        bandit_info["probabilities"] = bandit_series.to_dict()
        if posteriors:
            _persist_bandit_state(posteriors, bandit_info["active"])

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
        "bandit": bandit_info,
    }
