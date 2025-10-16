from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


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
    cov = rets.covwith(m) if hasattr(rets, "covwith") else rets.apply(lambda c: np.cov(c, m)[0, 1])
    var_m = m.var()
    beta = cov / var_m if var_m != 0 else cov * 0
    return pd.Series(beta, name="risk_beta")


def _standardize(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    std = series.std(ddof=0)
    if std and not np.isclose(std, 0.0):
        return (series - series.mean()) / std
    return series * 0.0


def fundamental_signals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Translate deterministic fundamental data into value/quality signals."""
    cols = fundamentals.copy()
    features = pd.DataFrame(index=fundamentals.index)
    if "pe_ratio" in cols:
        inv_pe = 1.0 / cols["pe_ratio"].replace(0, np.nan)
        features["value_score"] = _standardize(inv_pe.fillna(0.0))
    if "roe" in cols:
        features["quality_score"] = _standardize(cols["roe"].fillna(0.0))
    if "dividend_yield" in cols:
        features["dividend_yield"] = cols["dividend_yield"].fillna(0.0)
    if "debt_to_equity" in cols:
        inv_lev = -cols["debt_to_equity"].fillna(0.0)
        features["leverage_score"] = _standardize(inv_lev)
    return features


def news_sentiment_signal(sentiment: pd.Series) -> pd.Series:
    """Create a z-scored news sentiment factor."""
    signal = _standardize(sentiment.fillna(0.0))
    return signal.rename("news_sentiment")


def combine_features(
    prices: pd.DataFrame,
    fundamentals: Optional[pd.DataFrame] = None,
    sentiment: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Aggregate price, fundamental, and sentiment features."""
    blocks = [
        momentum_6m(prices),
        beta_252d(prices, bench_col="SPY"),
    ]
    if fundamentals is not None and not fundamentals.empty:
        blocks.append(fundamental_signals(fundamentals))
    if sentiment is not None and not sentiment.empty:
        blocks.append(news_sentiment_signal(sentiment))
    feats = pd.concat(blocks, axis=1)
    return feats.replace([np.inf, -np.inf], np.nan)
