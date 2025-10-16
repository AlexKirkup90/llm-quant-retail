import pandas as pd
import numpy as np

def pct_change_n(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    return prices.pct_change(n)

def momentum_6m(prices: pd.DataFrame) -> pd.Series:
    return prices.pct_change(126).iloc[-1].rename("mom_6m")

def beta_252d(prices: pd.DataFrame, bench_col: str = "SPY") -> pd.Series:
    # Simple market beta via covariance of daily returns
    rets = prices.pct_change().dropna()
    if bench_col not in rets.columns:
        return pd.Series(0.0, index=rets.columns, name="risk_beta")
    m = rets[bench_col]
    cov = rets.covwith(m) if hasattr(rets, "covwith") else rets.apply(lambda c: np.cov(c, m)[0,1])
    var_m = m.var()
    beta = cov / var_m if var_m != 0 else cov*0
    return pd.Series(beta, name="risk_beta")

def combine_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature set (can be extended by Codex from spec)."""
    feats = pd.concat([
        momentum_6m(prices),
        beta_252d(prices, bench_col="SPY"),
    ], axis=1)
    return feats.replace([np.inf, -np.inf], np.nan)
