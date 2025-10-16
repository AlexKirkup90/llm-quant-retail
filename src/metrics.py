import pandas as pd
import numpy as np

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
