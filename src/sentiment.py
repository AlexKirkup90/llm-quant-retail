from typing import Dict, List

import pandas as pd

from . import dataops, universe_registry


def _get_universe_price_history(universe_name: str, years: int = 1) -> pd.DataFrame:
    try:
        symbols = universe_registry.load_universe(universe_name)["symbol"].tolist()
    except Exception:
        return pd.DataFrame()

    if not symbols:
        return pd.DataFrame()

    return dataops.fetch_prices(symbols, years=years)


def _calculate_momentum(prices: pd.DataFrame, lookback_days: int) -> float:
    if prices.empty or len(prices) < lookback_days:
        return 0.0

    returns = prices.pct_change().mean(axis=1).dropna()
    cumulative_returns = (1 + returns).cumprod() - 1

    return cumulative_returns.iloc[-1] - cumulative_returns.iloc[-lookback_days]


def compute_universe_momentum_scores(
    universe_names: List[str],
) -> Dict[str, float]:
    scores = {}
    for name in universe_names:
        prices = _get_universe_price_history(name)
        if prices.empty:
            scores[name] = 0.0
            continue

        momentum_1m = _calculate_momentum(prices, lookback_days=21)
        momentum_3m = _calculate_momentum(prices, lookback_days=63)

        scores[name] = (0.4 * momentum_1m) + (0.6 * momentum_3m)

    return scores
