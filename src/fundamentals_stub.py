"""Deterministic fundamental feature stubs for offline testing.

This module fabricates stable, repeatable fundamental style inputs for
feature engineering.  Values are keyed off ticker symbols so they remain
consistent between runs without requiring external data access.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd

_SECTORS = [
    "TECH",
    "HEALTH",
    "INDUSTRIALS",
    "FINANCIALS",
    "CONSUMER",
    "ENERGY",
]


def _stable_int(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load(symbols: Iterable[str]) -> pd.DataFrame:
    """Return deterministic pseudo-fundamental metrics for *symbols*.

    The returned frame contains the columns required by the v0.2 feature
    set: a sector label, an EPS revision figure, a 30-day average dollar
    volume proxy, and a 90-day earnings volatility proxy.  All values are
    fabricated but stable so downstream signals are repeatable in tests.
    """

    rows = []
    for sym in symbols:
        if sym is None:
            continue
        base = _stable_int(sym)
        sector = _SECTORS[base % len(_SECTORS)]

        eps_seed = _stable_int(f"{sym}:eps")
        eps_raw = ((eps_seed % 2000) / 1000.0) - 1.0  # [-1, 1)
        eps_change = round(eps_raw * 0.06, 6)

        adv_seed = _stable_int(f"{sym}:adv")
        avg_dollar_vol = 2_000_000 + (adv_seed % 8_000_000)

        vol_seed = _stable_int(f"{sym}:vol")
        earnings_vol = 0.03 + ((vol_seed % 700) / 10_000.0)  # 0.03 – 0.099

        rows.append(
            {
                "symbol": sym,
                "sector": sector,
                "eps_rev_3m": eps_change,
                "avg_dollar_vol_30d": float(avg_dollar_vol),
                "earnings_vol_90d": round(earnings_vol, 6),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["sector", "eps_rev_3m", "avg_dollar_vol_30d", "earnings_vol_90d"]).set_index(
            pd.Index([], name="symbol")
        )

    return pd.DataFrame(rows).set_index("symbol")

