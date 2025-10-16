from __future__ import annotations

import json
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import RUNS_DIR

EMA_LAMBDA = 0.9
ROLLING_WEEKS = 12
RIDGE_ALPHA = 10.0

_VOL_REGIME_SCALERS: Dict[str, Dict[str, float]] = {
    "high": {"mom_6m": 0.7, "risk_beta": 0.7},
    "med": {"mom_6m": 1.0, "risk_beta": 1.0},
    "low": {"mom_6m": 1.05, "risk_beta": 1.05},
}

_MARKET_SENTIMENT_SCALERS: Dict[str, Dict[str, float]] = {
    "neg": {"quality_roic": 1.2, "def_stability": 1.2},
    "pos": {"mom_6m": 1.05, "eps_rev_3m": 1.05},
    "neu": {},
}


def fit_ridge(features_hist: pd.DataFrame, fwd_returns: pd.Series, alpha: float = RIDGE_ALPHA) -> pd.Series:
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


def _infer_vol_regime(rets: pd.DataFrame) -> str:
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
    vol_map = _VOL_REGIME_SCALERS.get(vol_regime.lower(), {})
    for feature, mult in vol_map.items():
        if feature in adjusted.index:
            adjusted.loc[feature] *= mult
    sent_map = _MARKET_SENTIMENT_SCALERS.get(market_sentiment.lower(), {})
    for feature, mult in sent_map.items():
        if feature in adjusted.index:
            adjusted.loc[feature] *= mult
    return _normalize_weights(adjusted)


def _serialise_history(history: Mapping[pd.Timestamp, pd.Series]) -> Dict[str, Dict[str, float]]:
    return {
        ts.isoformat(): {str(k): float(v) for k, v in series.dropna().items()}
        for ts, series in history.items()
    }


def _extract_feature_panel(features: object) -> Dict[pd.Timestamp, pd.DataFrame]:
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
    weeks: int = ROLLING_WEEKS,
    alpha: float = RIDGE_ALPHA,
) -> pd.Series:
    """Fit rolling ridge regressions and return smoothed feature weights."""

    rets_df = rets.to_frame().T if isinstance(rets, pd.Series) else rets.copy()
    rets_df.index = pd.to_datetime(rets_df.index)
    rets_df = rets_df.sort_index()
    feature_panels = _extract_feature_panel(features)
    if not feature_panels:
        return pd.Series(dtype=float)

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
    ema_alpha = max(1e-3, 1.0 - EMA_LAMBDA)
    smoothed = hist_df.tail(effective_window).ewm(alpha=ema_alpha, adjust=False).mean()
    latest_weights = smoothed.iloc[-1]

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

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "vol_regime": vol_regime,
        "market_sentiment": market_sentiment,
        "ema_lambda": EMA_LAMBDA,
        "weights": {k: float(v) for k, v in scaled.items()},
        "history": _serialise_history(history),
    }
    out_path = RUNS_DIR / "feature_weights.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

    return scaled


def score_current(features_now: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols = [c for c in weights.index if c in features_now.columns]
    if not cols:
        return pd.Series(dtype=float)
    X = features_now[cols].fillna(0.0)
    s = X.dot(weights.loc[cols])
    return s.sort_values(ascending=False).rename("score")
