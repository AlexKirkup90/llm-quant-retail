"""Deterministic sentiment stubs used for testing feature engineering."""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


def _stable_int(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def load(symbols: Iterable[str], window: int = 7) -> pd.Series:
    """Return a deterministic [-1, 1] news sentiment score for each symbol."""

    damp = max(0.2, 1.0 - min(window, 30) / 40.0)
    scores = {}
    for sym in symbols:
        if sym is None:
            continue
        token = _stable_int(f"{sym}:{window}")
        raw = ((token % 2001) / 1000.0) - 1.0
        score = max(-1.0, min(1.0, round(raw * damp, 6)))
        scores[sym] = score

    return pd.Series(scores, name="news_sentiment")

