import hashlib
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
import yfinance as yf

from .config import CACHE_DIR
from .universe import load_universe as _load_universe


def fetch_prices(symbols, years=5, interval="1d") -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    tickers = " ".join(symbols)
    df = yf.download(tickers=tickers, start=start, interval=interval, auto_adjust=True, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all").sort_index()
    df.columns = [c.replace(" ", "") for c in df.columns]
    return df

def load_universe(mode: str = "SP500") -> pd.DataFrame:
    return _load_universe(mode)


def cache_parquet(df: pd.DataFrame, name: str) -> str:
    path = CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return str(path)


def _stable_int(symbol: str) -> int:
    """Create a deterministic integer from a ticker symbol."""
    digest = hashlib.sha256(symbol.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def fetch_fundamentals(symbols: Sequence[str]) -> pd.DataFrame:
    """Return deterministic fundamental metrics for each symbol."""
    rows = []
    for sym in symbols:
        token = _stable_int(sym)
        pe_ratio = 8.0 + (token % 2500) / 150.0
        dividend_yield = ((token // 17) % 600) / 10000.0
        roe = 0.04 + ((token // 131) % 400) / 1000.0
        debt_to_equity = 0.15 + ((token // 19) % 220) / 120.0
        rows.append({
            "symbol": sym,
            "pe_ratio": round(pe_ratio, 4),
            "dividend_yield": round(dividend_yield, 4),
            "roe": round(roe, 4),
            "debt_to_equity": round(debt_to_equity, 4),
        })
    df = pd.DataFrame(rows).set_index("symbol")
    return df


def fetch_news_sentiment(symbols: Sequence[str], window: int = 7) -> pd.Series:
    """Return a smoothed sentiment score in [-1, 1] for each symbol."""
    scores = {}
    damp = max(0.2, 1.0 - min(window, 30) / 40.0)
    for sym in symbols:
        token = _stable_int(f"{sym}:{window}")
        raw = ((token % 2001) / 1000.0) - 1.0
        score = max(-1.0, min(1.0, round(raw * damp, 4)))
        scores[sym] = score
    return pd.Series(scores, name="news_sentiment")
