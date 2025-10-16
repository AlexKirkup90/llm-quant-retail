from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import fundamentals_stub, sentiment_stub


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


def eps_rev_3m(symbols: Sequence[str], stub: Optional[pd.DataFrame] = None) -> pd.Series:
    """Sector-neutral 3M EPS revision using deterministic fundamentals stub."""

    data = fundamentals_stub.load(symbols) if stub is None else stub
    if data.empty:
        return pd.Series(dtype=float, name="eps_rev_3m")

    eps = data["eps_rev_3m"].astype(float)
    sectors = data.get("sector")

    if sectors is not None:
        z = eps.groupby(sectors).transform(lambda x: _standardize(x))
    else:
        z = _standardize(eps)
    return z.fillna(0.0).rename("eps_rev_3m")


def liq_adv(symbols: Sequence[str], stub: Optional[pd.DataFrame] = None) -> pd.Series:
    """Log ADV z-score from deterministic stub fundamentals."""

    data = fundamentals_stub.load(symbols) if stub is None else stub
    if data.empty or "avg_dollar_vol_30d" not in data:
        return pd.Series(dtype=float, name="liq_adv")

    adv = np.log(pd.Series(data["avg_dollar_vol_30d"], dtype=float).clip(lower=1.0))
    return _standardize(adv).rename("liq_adv")


def def_stability(symbols: Sequence[str], stub: Optional[pd.DataFrame] = None) -> pd.Series:
    """Inverse 90-day earnings volatility (higher = more stable)."""

    data = fundamentals_stub.load(symbols) if stub is None else stub
    if data.empty or "earnings_vol_90d" not in data:
        return pd.Series(dtype=float, name="def_stability")

    vol = pd.Series(data["earnings_vol_90d"], dtype=float).replace(0, np.nan)
    inv_vol = 1.0 / vol
    return _standardize(inv_vol.fillna(0.0)).rename("def_stability")


def news_sent(symbols: Sequence[str], stub: Optional[pd.Series] = None, window: int = 7) -> pd.Series:
    """7-day trailing mean sentiment score normalized to z-score."""

    data = sentiment_stub.load(symbols, window=window) if stub is None else stub
    if data is None or data.empty:
        return pd.Series(dtype=float, name="news_sent")

    bounded = pd.Series(data, dtype=float).clip(-1.0, 1.0)
    return _standardize(bounded).rename("news_sent")


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

    tickers = feats.index.tolist()
    if tickers:
        stub_data = fundamentals_stub.load(tickers)
        sentiment_stub_series = sentiment_stub.load(tickers)
        extra_blocks = [
            eps_rev_3m(tickers, stub=stub_data),
            liq_adv(tickers, stub=stub_data),
            def_stability(tickers, stub=stub_data),
            news_sent(tickers, stub=sentiment_stub_series),
        ]
        feats = pd.concat([feats] + extra_blocks, axis=1)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats
