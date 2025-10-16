from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import RUNS_DIR


def equity_curve(weights_hist: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    aligned = weights_hist.shift(1).fillna(0)  # next-day open proxy
    port_ret = (aligned * returns).sum(axis=1)
    return (1 + port_ret).cumprod()


def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = (curve / peak - 1.0).min()
    return float(abs(dd))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    dr = returns - rf
    downside = dr[dr < 0]
    denom = downside.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return dr.mean() / denom


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return float("nan")
    excess = returns - rf / periods_per_year
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def alpha_vs_bench(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    diff = port_ret.align(bench_ret, join="inner", fill_value=0)
    return float(diff[0].sub(diff[1]).mean())


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return float("nan")
    total = (1 + returns).prod()
    if total <= 0:
        return float("nan")
    years = len(returns) / periods_per_year if periods_per_year else 1.0
    if years <= 0:
        return float("nan")
    return float(total ** (1 / years) - 1.0)


def performance_summary(
    net_returns: pd.Series,
    bench_returns: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    *,
    gross_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    net = pd.Series(net_returns).dropna()
    gross = pd.Series(gross_returns).dropna() if gross_returns is not None else net
    bench = pd.Series(bench_returns).reindex(net.index).fillna(0.0)
    curve = (1 + net).cumprod()

    summary = {
        "net_cagr": annual_return(net, periods_per_year=periods_per_year),
        "gross_cagr": annual_return(gross, periods_per_year=periods_per_year),
        "volatility": float(net.std(ddof=0) * np.sqrt(periods_per_year)) if not net.empty else float("nan"),
        "sortino": sortino(net),
        "sharpe": sharpe(net, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(curve) if not curve.empty else float("nan"),
        "alpha": alpha_vs_bench(net, bench),
        "avg_turnover": float(pd.Series(turnover).fillna(0.0).mean()),
        "total_cost": float(pd.Series(costs).fillna(0.0).sum()),
    }
    spread = max(0.0, summary["gross_cagr"] - summary["net_cagr"])
    summary["gross_vs_net_spread"] = min(spread, summary["total_cost"])
    summary["cost_bps_weekly"] = (
        float(pd.Series(costs).fillna(0.0).mean() * 10000.0)
        if costs is not None
        else 0.0
    )
    summary["turnover"] = summary["avg_turnover"]
    summary["last_equity"] = float(curve.iloc[-1]) if not curve.empty else 1.0
    return summary


def val_split_rolling_80_20(indexed: pd.Series | pd.DataFrame, window: int = 60) -> List[Tuple[pd.Index, pd.Index]]:
    if window <= 0:
        raise ValueError("window must be positive")
    if isinstance(indexed, pd.DataFrame):
        idx = indexed.index
    else:
        idx = indexed.index
    if len(idx) < window:
        return []
    splits: List[Tuple[pd.Index, pd.Index]] = []
    for end in range(window, len(idx) + 1):
        window_idx = idx[end - window : end]
        split_point = int(len(window_idx) * 0.8)
        train_idx = window_idx[:split_point]
        val_idx = window_idx[split_point:]
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def val_metrics(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    window: int = 60,
) -> Dict[str, float]:
    port_ret = port_ret.dropna()
    bench_ret = bench_ret.dropna()
    splits = val_split_rolling_80_20(port_ret, window=window)
    if not splits:
        return {"val_sortino": float("nan"), "val_alpha": float("nan")}

    sortinos = []
    alphas = []
    for _, val_idx in splits:
        port_slice = port_ret.loc[val_idx]
        bench_slice = bench_ret.reindex(val_idx).fillna(0.0)
        if port_slice.empty:
            continue
        sortinos.append(sortino(port_slice))
        alphas.append(alpha_vs_bench(port_slice, bench_slice))

    if not sortinos:
        return {"val_sortino": float("nan"), "val_alpha": float("nan")}

    return {
        "val_sortino": float(np.nanmean(sortinos)),
        "val_alpha": float(np.nanmean(alphas)) if alphas else float("nan"),
    }


def spearman_ic(predicted: pd.Series, future: pd.Series) -> float:
    """Compute Spearman rank IC between predictions and future returns."""

    if predicted is None or future is None:
        return float("nan")
    pred = pd.Series(predicted).astype(float)
    fut = pd.Series(future).astype(float)
    aligned_pred, aligned_fut = pred.align(fut, join="inner")
    aligned_pred = aligned_pred.replace([np.inf, -np.inf], np.nan)
    aligned_fut = aligned_fut.replace([np.inf, -np.inf], np.nan)
    mask = aligned_pred.notna() & aligned_fut.notna()
    if mask.sum() < 3:
        return float("nan")
    ranks_pred = aligned_pred[mask].rank(method="average")
    ranks_fut = aligned_fut[mask].rank(method="average")
    corr = ranks_pred.corr(ranks_fut)
    return float(corr) if pd.notna(corr) else float("nan")


def hit_rate(predicted: pd.Series, future: pd.Series, top_frac: float = 0.1) -> float:
    """Return hit-rate of top predictions beating the median future return."""

    if predicted is None or future is None:
        return float("nan")
    pred = pd.Series(predicted).astype(float)
    fut = pd.Series(future).astype(float)
    aligned_pred, aligned_fut = pred.align(fut, join="inner")
    aligned_pred = aligned_pred.replace([np.inf, -np.inf], np.nan)
    aligned_fut = aligned_fut.replace([np.inf, -np.inf], np.nan)
    mask = aligned_pred.notna() & aligned_fut.notna()
    if mask.sum() == 0:
        return float("nan")
    working_scores = aligned_pred[mask]
    working_returns = aligned_fut[mask]
    if working_scores.empty:
        return float("nan")
    frac = max(0.0, min(1.0, float(top_frac)))
    top_n = max(1, int(round(len(working_scores) * frac)))
    top_idx = working_scores.sort_values(ascending=False).head(top_n).index
    median_ret = working_returns.median()
    hits = (working_returns.loc[top_idx] > median_ret).mean()
    return float(hits)


def feature_ic_snapshot(features: pd.DataFrame, future_returns: pd.Series) -> pd.Series:
    """Compute per-feature Spearman IC vs. future returns."""

    if features is None or features.empty or future_returns is None:
        return pd.Series(dtype=float)
    fut = pd.Series(future_returns).astype(float)
    ic_values: Dict[str, float] = {}
    for column in features.columns:
        values = features[column]
        if not isinstance(values, pd.Series):
            ic_values[column] = float("nan")
            continue
        aligned_vals, aligned_fut = values.align(fut, join="inner")
        aligned_vals = aligned_vals.replace([np.inf, -np.inf], np.nan)
        aligned_fut = aligned_fut.replace([np.inf, -np.inf], np.nan)
        mask = aligned_vals.notna() & aligned_fut.notna()
        if mask.sum() < 3:
            ic_values[column] = float("nan")
            continue
        corr = aligned_vals[mask].rank(method="average").corr(
            aligned_fut[mask].rank(method="average")
        )
        ic_values[column] = float(corr) if pd.notna(corr) else float("nan")
    return pd.Series(ic_values).sort_index()


def append_ic_metric(record: Dict[str, object], path: Path | None = None) -> None:
    """Append IC metrics to runs/ic_history.json."""

    if record is None:
        return
    target = Path(path) if path else RUNS_DIR / "ic_history.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    data: List[Dict[str, object]]
    if target.exists():
        try:
            parsed = json.loads(target.read_text())
            data = parsed if isinstance(parsed, list) else []
        except Exception:
            data = []
    else:
        data = []
    data.append(record)
    try:
        target.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def load_ic_history(path: Path | None = None) -> pd.DataFrame:
    """Load IC history as DataFrame."""

    target = Path(path) if path else RUNS_DIR / "ic_history.json"
    if not target.exists():
        return pd.DataFrame(columns=["date", "ic", "hit_rate", "universe", "benchmark"])
    try:
        parsed = json.loads(target.read_text())
    except Exception:
        return pd.DataFrame(columns=["date", "ic", "hit_rate", "universe", "benchmark"])
    if not isinstance(parsed, list) or not parsed:
        return pd.DataFrame(columns=["date", "ic", "hit_rate", "universe", "benchmark"])
    df = pd.DataFrame(parsed)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df.sort_values("date")
