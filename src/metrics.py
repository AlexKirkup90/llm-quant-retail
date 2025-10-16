from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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

def alpha_vs_bench(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    diff = port_ret.align(bench_ret, join="inner", fill_value=0)
    return float(diff[0].sub(diff[1]).mean())


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
