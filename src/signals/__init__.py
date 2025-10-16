from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..config import RUNS_DIR

# ---------------------------------------------------------------------
# v0.3 CONFIG BRIDGE
# Expose module attributes expected by the app (ROLLING_WEEKS, EMA_LAMBDA, RIDGE_ALPHA)
# and allow overrides from spec/current_spec.json. Also allow regime scalers from spec.
# ---------------------------------------------------------------------

# Hard defaults (used if spec missing/invalid)
_DEFAULT_EMA_LAMBDA = 0.9
_DEFAULT_ROLLING_WEEKS = 12
_DEFAULT_RIDGE_ALPHA = 10.0

# Default regime scalers (used unless overridden by spec)
_DEFAULT_VOL_REGIME_SCALERS: Dict[str, Dict[str, float]] = {
    "high": {"mom_6m": 0.7, "risk_beta": 0.7},
    "med": {"mom_6m": 1.0, "risk_beta": 1.0},
    "low": {"mom_6m": 1.05, "risk_beta": 1.05},
}
_DEFAULT_MARKET_SENTIMENT_SCALERS: Dict[str, Dict[str, float]] = {
    "neg": {"quality_roic": 1.2, "def_stability": 1.2},
    "pos": {"mom_6m": 1.05, "eps_rev_3m": 1.05},
    "neu": {},
}


def _load_spec_overrides():
    """Read spec/current_spec.json and return config overrides if present."""
    spec_path = Path("spec/current_spec.json")
    if not spec_path.exists():
        return {
            "ema_lambda": _DEFAULT_EMA_LAMBDA,
            "rolling_weeks": _DEFAULT_ROLLING_WEEKS,
            "ridge_alpha": _DEFAULT_RIDGE_ALPHA,
            "vol_scalers": _DEFAULT_VOL_REGIME_SCALERS,
            "sent_scalers": _DEFAULT_MARKET_SENTIMENT_SCALERS,
        }

    try:
        spec = json.loads(spec_path.read_text())
        learning = spec.get("learning", {})
        regime_scalers = spec.get("regime_scalers", {})

        ema_lambda = float(learning.get("ema_lambda", _DEFAULT_EMA_LAMBDA))
        rolling_weeks = int(learning.get("rolling_weeks", _DEFAULT_ROLLING_WEEKS))
        ridge_alpha = float(learning.get("ridge_alpha", _DEFAULT_RIDGE_ALPHA))

        # Try to read scaler maps; fallback to defaults if keys missing
        vol_scalers = regime_scalers.get("vol_regime", _DEFAULT_VOL_REGIME_SCALERS) or _DEFAULT_VOL_REGIME_SCALERS
        sent_scalers = regime_scalers.get("market_sentiment", _DEFAULT_MARKET_SENTIMENT_SCALERS) or _DEFAULT_MARKET_SENTIMENT_SCALERS

        return {
            "ema_lambda": ema_lambda,
            "rolling_weeks": rolling_weeks,
            "ridge_alpha": ridge_alpha,
            "vol_scalers": vol_scalers,
            "sent_scalers": sent_scalers,
        }
    except Exception:
        # On any parsing/type error, return safe defaults
        return {
            "ema_lambda": _DEFAULT_EMA_LAMBDA,
            "rolling_weeks": _DEFAULT_ROLLING_WEEKS,
            "ridge_alpha": _DEFAULT_RIDGE_ALPHA,
            "vol_scalers": _DEFAULT_VOL_REGIME_SCALERS,
            "sent_scalers": _DEFAULT_MARKET_SENTIMENT_SCALERS,
        }


# Load overrides at import time
_cfg = _load_spec_overrides()

# Public module-level attributes (what the app expects)
EMA_LAMBDA: float = _cfg["ema_lambda"]
ROLLING_WEEKS: int = _cfg["rolling_weeks"]
RIDGE_ALPHA: float = _cfg["ridge_alpha"]

# Internal scaler maps (may be overridden by spec)
_VOL_REGIME_SCALERS: Dict[str, Dict[str, float]] = _cfg["vol_scalers"]
_MARKET_SENTIMENT_SCALERS: Dict[str, Dict[str, float]] = _cfg["sent_scalers"]

_FEATURE_IC_EMA_PATH = RUNS_DIR / "feature_ic_ema.json"


def get_learning_config() -> Dict[str, float | int]:
    """Public helper so callers can fetch current learning config."""
    return {
        "ema_lambda": EMA_LAMBDA,
        "rolling_weeks": ROLLING_WEEKS,
        "ridge_alpha": RIDGE_ALPHA,
    }


# ---------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------


def fit_ridge(features_hist: pd.DataFrame, fwd_returns: pd.Series, alpha: float = None) -> pd.Series:
    """
    Fit a single ridge regression on the provided cross-section.

    Parameters
    ----------
    features_hist : DataFrame
        Cross-sectional feature snapshot (rows = tickers, cols = features).
    fwd_returns : Series
        Forward returns aligned by ticker.
    alpha : float, optional
        Ridge alpha. Defaults to RIDGE_ALPHA (from spec or defaults).
    """
    if alpha is None:
        alpha = RIDGE_ALPHA

    df = features_hist.dropna()
    common = df.index.intersection(fwd_returns.dropna().index)
    if common.empty:
        return pd.Series(1.0 / max(1, df.shape[1]), index=df.columns, name="weight")
    X = df.loc[common].values
    y = fwd_returns.loc[common].values
    if len(common) < 20 or df.shape[1] == 0:
        return pd.Series(1.0 / max(1, df.shape[1]), index=df.columns, name="weight")
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    w = pd.Series(model.coef_, index=df.columns, name="weight")
    return w


def _normalize_weights(weights: pd.Series) -> pd.Series:
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = weights.abs().sum()
    if denom and not np.isclose(denom, 0.0):
        return weights / denom
    if len(weights) == 0:
        return weights
    return pd.Series(1.0 / len(weights), index=weights.index)


def _sector_zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    std = values.std(ddof=0)
    if std and not np.isclose(std, 0.0):
        return (values - values.mean()) / std
    return values * 0.0


def sector_neutralize_features(
    features: pd.DataFrame,
    sectors: pd.Series | Mapping[str, str] | None,
) -> pd.DataFrame:
    """Apply within-sector z-scoring across all feature columns."""

    if features is None or features.empty:
        return features
    if sectors is None:
        return features

    if not isinstance(sectors, pd.Series):
        sectors = pd.Series(sectors)

    aligned = sectors.reindex(features.index)
    if aligned.dropna().empty:
        return features

    neutralised = features.copy()
    for column in neutralised.columns:
        col_values = neutralised[column]
        neutralised[column] = col_values.groupby(aligned).transform(_sector_zscore)
    return neutralised.fillna(0.0)


def _infer_vol_regime(rets: pd.DataFrame) -> str:
    if isinstance(rets, pd.Series):
        rets = rets.to_frame()
    if rets.empty:
        return "med"
    vol_series = rets.mean(axis=1).rolling(20, min_periods=5).std()
    vol = vol_series.dropna().iloc[-1] if not vol_series.dropna().empty else 0.0
    if vol > 0.04:
        return "high"
    if vol < 0.015:
        return "low"
    return "med"


def _infer_market_sentiment(scores: Sequence[float] | None) -> str:
    if not scores:
        return "neu"
    arr = np.array(list(scores), dtype=float)
    if len(arr) == 0 or np.isnan(arr).all():
        return "neu"
    mean = np.nanmean(arr)
    if mean > 0.25:
        return "pos"
    if mean < -0.25:
        return "neg"
    return "neu"


def _apply_regime_scalers(weights: pd.Series, vol_regime: str, market_sentiment: str) -> pd.Series:
    adjusted = weights.copy()

    # Vol regime scalers
    vol_map = _VOL_REGIME_SCALERS.get(vol_regime.lower(), {})
    for feature, mult in vol_map.items():
        if feature in adjusted.index:
            adjusted.loc[feature] *= float(mult)

    # Market sentiment scalers
    sent_map = _MARKET_SENTIMENT_SCALERS.get(market_sentiment.lower(), {})
    for feature, mult in sent_map.items():
        if feature in adjusted.index:
            adjusted.loc[feature] *= float(mult)

    return _normalize_weights(adjusted)


def _serialise_history(history: Mapping[pd.Timestamp, pd.Series]) -> Dict[str, Dict[str, float]]:
    return {
        ts.isoformat(): {str(k): float(v) for k, v in series.dropna().items()}
        for ts, series in history.items()
    }


def _extract_feature_panel(features: object) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Accepts various feature containers and returns a dict of {timestamp: DataFrame}.
    Supports:
      - dict[timestamp -> DataFrame]
      - DataFrame with MultiIndex (level 0 = timestamp)
      - DataFrame with 'date' column
      - Plain DataFrame (single snapshot); uses features.attrs['as_of'] or now()
    """
    if isinstance(features, dict):
        return {pd.Timestamp(k): v for k, v in features.items()}

    if isinstance(features, pd.DataFrame) and isinstance(features.index, pd.MultiIndex):
        panels: Dict[pd.Timestamp, pd.DataFrame] = {}
        for ts in features.index.get_level_values(0).unique():
            frame = features.xs(ts, level=0)
            panels[pd.Timestamp(ts)] = frame
        return panels

    if isinstance(features, pd.DataFrame) and "date" in features.columns:
        panels = {}
        for ts, frame in features.groupby("date"):
            frame = frame.drop(columns=["date"])
            panels[pd.Timestamp(ts)] = frame
        return panels

    if isinstance(features, pd.DataFrame):
        panels = {}
        ts = getattr(features, "attrs", {}).get("as_of")
        timestamp = pd.Timestamp(ts) if ts else pd.Timestamp.utcnow()
        panels[timestamp] = features
        return panels

    raise TypeError("Unsupported features container")


def fit_rolling_ridge(
    rets: pd.DataFrame | pd.Series,
    features: Mapping | pd.DataFrame,
    weeks: int | None = None,
    alpha: float | None = None,
) -> pd.Series:
    """
    Fit rolling ridge regressions and return smoothed feature weights.

    Parameters
    ----------
    rets : DataFrame or Series
        Forward returns panel indexed by date (rows) and tickers (columns),
        or a Series aligned to a single cross-section.
    features : Mapping or DataFrame
        See _extract_feature_panel for accepted formats.
    weeks : int, optional
        Number of *weeks* worth of history to include. Defaults to spec/default ROLLING_WEEKS.
    alpha : float, optional
        Ridge alpha. Defaults to spec/default RIDGE_ALPHA.
    """
    if weeks is None:
        weeks = ROLLING_WEEKS
    if alpha is None:
        alpha = RIDGE_ALPHA

    # Ensure DataFrame with datetime index
    rets_df = rets.to_frame().T if isinstance(rets, pd.Series) else rets.copy()
    rets_df.index = pd.to_datetime(rets_df.index)
    rets_df = rets_df.sort_index()

    feature_panels = _extract_feature_panel(features)
    if not feature_panels:
        return pd.Series(dtype=float)

    # Convert weeks to "trading days" approx for EW smoothing window
    window = max(1, int(weeks * 5))
    history: Dict[pd.Timestamp, pd.Series] = {}

    for ts in sorted(feature_panels.keys()):
        if ts not in rets_df.index:
            continue
        feature_snapshot = feature_panels[ts]
        if feature_snapshot.empty:
            continue
        fwd = rets_df.loc[ts].dropna()
        common = feature_snapshot.index.intersection(fwd.index)
        if len(common) < 3:
            continue
        weights = fit_ridge(feature_snapshot.loc[common], fwd.loc[common], alpha=alpha)
        history[ts] = weights

    if not history:
        latest_panel = feature_panels[sorted(feature_panels.keys())[-1]]
        return pd.Series(1.0 / max(1, latest_panel.shape[1]), index=latest_panel.columns)

    hist_df = pd.DataFrame(history).T.sort_index()
    effective_window = min(window, len(hist_df))

    # EWM alpha derived from EMA_LAMBDA; guard tiny alpha
    ema_alpha = max(1e-3, 1.0 - EMA_LAMBDA)
    smoothed = hist_df.tail(effective_window).ewm(alpha=ema_alpha, adjust=False).mean()
    latest_weights = smoothed.iloc[-1]

    # Infer regimes if not explicitly provided via attrs
    vol_regime = getattr(rets, "attrs", {}).get("vol_regime") or _infer_vol_regime(rets_df)
    analyst_scores = None
    if hasattr(features, "attrs"):
        analyst_scores = features.attrs.get("analyst_scores")
    else:
        for snapshot in feature_panels.values():
            if hasattr(snapshot, "attrs") and "analyst_scores" in snapshot.attrs:
                analyst_scores = snapshot.attrs.get("analyst_scores")
                break
    market_sentiment = getattr(rets, "attrs", {}).get("market_sentiment") or _infer_market_sentiment(analyst_scores)

    scaled = _apply_regime_scalers(latest_weights, vol_regime, market_sentiment)

    # Persist feature weight snapshot and history
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "vol_regime": vol_regime,
        "market_sentiment": market_sentiment,
        "ema_lambda": EMA_LAMBDA,
        "rolling_weeks": weeks,
        "ridge_alpha": alpha,
        "weights": {k: float(v) for k, v in scaled.items()},
        "history": _serialise_history(history),
    }
    out_path = RUNS_DIR / "feature_weights.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2))
    except Exception:
        # never fail the run on serialization issues
        pass

    return scaled


def score_current(
    features_now: pd.DataFrame,
    weights: pd.Series,
    *,
    sector_map: pd.Series | Mapping[str, str] | None = None,
    sector_neutral: bool = False,
) -> pd.Series:
    cols = [c for c in weights.index if c in features_now.columns]
    if not cols:
        return pd.Series(dtype=float)
    working = features_now[cols]
    if sector_neutral:
        working = sector_neutralize_features(working, sector_map)
    X = working.fillna(0.0)
    s = X.dot(weights.loc[cols])
    return s.sort_values(ascending=False).rename("score")


def load_feature_ic_ema() -> pd.Series:
    """Load the persisted feature IC EMA series."""

    if not _FEATURE_IC_EMA_PATH.exists():
        return pd.Series(dtype=float)
    try:
        data = json.loads(_FEATURE_IC_EMA_PATH.read_text())
    except Exception:
        return pd.Series(dtype=float)
    if isinstance(data, dict):
        return pd.Series(data, dtype=float).sort_index()
    return pd.Series(dtype=float)


def save_feature_ic_ema(series: pd.Series) -> None:
    """Persist feature IC EMA values to disk."""

    if series is None or series.empty:
        try:
            if _FEATURE_IC_EMA_PATH.exists():
                _FEATURE_IC_EMA_PATH.unlink()
        except Exception:
            pass
        return
    payload = {str(k): float(v) for k, v in series.items() if pd.notna(v)}
    try:
        _FEATURE_IC_EMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FEATURE_IC_EMA_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def update_feature_ic_ema(
    existing: pd.Series | None,
    snapshot: pd.Series,
    *,
    ema_lambda: float = 0.9,
) -> pd.Series:
    """Blend new IC snapshot into an EMA."""

    new_snapshot = pd.Series(snapshot).astype(float)
    if existing is None or existing.empty:
        return new_snapshot.fillna(0.0)
    current = pd.Series(existing).astype(float)
    union_index = current.index.union(new_snapshot.index)
    current = current.reindex(union_index, fill_value=0.0)
    new_snapshot = new_snapshot.reindex(union_index, fill_value=0.0)
    lam = float(ema_lambda)
    lam = max(0.0, min(0.999, lam))
    blended = lam * current + (1 - lam) * new_snapshot
    return blended


def apply_ic_weighting(
    weights: pd.Series,
    ic_ema: pd.Series | None,
    *,
    alpha_ic: float = 0.2,
    clip: float = 0.5,
) -> pd.Series:
    """Adjust weights based on IC EMA signal."""

    if weights is None or weights.empty or ic_ema is None or ic_ema.empty:
        return weights
    w = pd.Series(weights).astype(float)
    ic_series = pd.Series(ic_ema).astype(float)
    ic_series = ic_series.reindex(w.index).fillna(0.0)
    mult = 1.0 + float(alpha_ic) * ic_series
    clip_val = abs(float(clip))
    if clip_val > 0:
        mult = mult.clip(lower=1.0 - clip_val, upper=1.0 + clip_val)
    adjusted = w * mult
    return _normalize_weights(adjusted)
