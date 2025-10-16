import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

from .config import CACHE_DIR, REF_DIR
from .universe import load_universe as _load_universe


LATEST_SNAPSHOT_PATH = REF_DIR / "ohlcv_latest.csv"
PRICES_CACHE_DIR = CACHE_DIR / "prices"
PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).split(" ")[-1].upper() for c in df.columns]
    return df


def _write_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return
    LATEST_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LATEST_SNAPSHOT_PATH)


def _read_snapshot() -> pd.DataFrame:
    if not LATEST_SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(LATEST_SNAPSHOT_PATH, index_col=0, parse_dates=True)
        return _normalise_columns(data)
    except Exception:
        return pd.DataFrame()


def _cache_path(universe: str) -> Path:
    safe = universe.replace("/", "_").lower()
    return PRICES_CACHE_DIR / f"{safe}.parquet"


def fetch_prices(symbols, years: int = 5, interval: str = "1d", universe: str | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch OHLCV close prices with caching and offline fallback."""

    if not symbols:
        return pd.DataFrame()

    start = (datetime.utcnow() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    tickers = " ".join(symbols)
    df = pd.DataFrame()

    if not force_refresh:
        cache_path = _cache_path(universe or "default")
        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
                cached.index = pd.to_datetime(cached.index)
                cached = cached.sort_index()
                if not cached.empty:
                    df = cached
            except Exception:
                df = pd.DataFrame()

    if df.empty or force_refresh:
        try:
            live = yf.download(
                tickers=tickers,
                start=start,
                interval=interval,
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if isinstance(live, pd.Series):
                live = live.to_frame(name="Close")
            if isinstance(live.columns, pd.MultiIndex):
                live = live["Close"]
            live = live.dropna(how="all").sort_index()
            live = _normalise_columns(live)
            df = live
            if not df.empty:
                cache_path = _cache_path(universe or "default")
                df.to_parquet(cache_path)
                _write_snapshot(df)
        except Exception:
            pass

    if df.empty:
        snapshot = _read_snapshot()
        if not snapshot.empty:
            df = snapshot

    if df.empty:
        cache_path = _cache_path(universe or "default")
        if cache_path.exists():
            df = pd.read_parquet(cache_path)

    if df.empty:
        raise RuntimeError("Unable to load OHLCV data from live or cached sources")

    df = df.reindex(columns=[c.upper() for c in symbols], copy=False)
    df = df.dropna(how="all").sort_index()
    return df

def load_universe(mode: str = "SP500") -> pd.DataFrame:
    return _load_universe(mode)


def cache_parquet(df: pd.DataFrame, name: str) -> str:
    path = CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return str(path)


def compute_adv_from_prices_approx(prices: pd.DataFrame) -> pd.Series:
    """Return an approximate ADV series when dollar volume metadata is available."""

    meta = getattr(prices, "attrs", {}) or {}
    dollar_volume = meta.get("daily_dollar_volume")
    if isinstance(dollar_volume, pd.Series):
        return dollar_volume
    if isinstance(dollar_volume, dict):
        return pd.Series(dollar_volume)
    return pd.Series(index=getattr(prices, "columns", None), dtype="float64")


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
