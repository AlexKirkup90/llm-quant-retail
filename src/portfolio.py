import pandas as pd
import numpy as np

def inverse_vol_weights(returns_252: pd.DataFrame, tickers: list, cap_single: float = 0.10, k: int = 15) -> pd.Series:
    vol = returns_252.std().replace(0, np.nan).dropna()
    vol = vol.loc[vol.index.intersection(tickers)]
    top = vol.nsmallest(k).index
    inv = 1.0 / vol.loc[top]
    w = inv / inv.sum()
    w = w.clip(upper=cap_single)
    w = w / w.sum()
    return w

def apply_sector_caps(weights: pd.Series, sector_map: pd.Series, cap: float = 0.35) -> pd.Series:
    w = weights.copy()
    sectors = sector_map.loc[w.index]
    for sec, sw in w.groupby(sectors).sum().items():
        if sw > cap:
            scale = cap / sw
            w.loc[sectors == sec] *= scale
    return w / w.sum()

def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    if prev_w is None or prev_w.empty:
        return new_w.abs().sum()
    idx = prev_w.index.union(new_w.index)
    return (prev_w.reindex(idx, fill_value=0) - new_w.reindex(idx, fill_value=0)).abs().sum()

def enforce_turnover(prev_w: pd.Series, cand_w: pd.Series, t_cap: float = 0.30) -> pd.Series:
    if prev_w is None or prev_w.empty:
        return cand_w
    t = turnover(prev_w, cand_w)
    if t <= t_cap:
        return cand_w
    # Linearly interpolate toward prev_w to meet turnover cap
    lam = t_cap / t
    blended = lam * cand_w + (1 - lam) * prev_w.reindex(cand_w.index, fill_value=0)
    return (blended / blended.sum()).clip(lower=0)
