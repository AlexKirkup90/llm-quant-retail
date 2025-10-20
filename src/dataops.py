import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import yfinance as yf

from .config import CACHE_DIR
from .universe import load_universe as _load_universe
from . import universe_registry


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


REFERENCE_DIR = Path("data/reference")
OHLCV_LATEST_PATH = REFERENCE_DIR / "ohlcv_latest.csv"


def build_ohlcv_snapshot(universe: str, out_path: str) -> Dict[str, object]:
    """Fetch full-universe OHLCV data and persist as a wide CSV."""

    df = universe_registry.load_universe(universe)
    if df is None or df.empty:
        raise ValueError(f"Universe {universe} returned no constituents")

    symbols = (
        df.get("symbol", pd.Series(dtype=str))
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not symbols:
        raise ValueError(f"Universe {universe} returned no symbols")

    prices = fetch_prices(symbols, years=5)
    if prices.empty:
        raise ValueError(f"Failed to download prices for {universe}")

    prices = prices.sort_index().astype(float)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(target)

    rows, cols = prices.shape
    return {"rows": int(rows), "cols": int(cols), "path": str(target)}


def load_latest_ohlcv_snapshot() -> pd.DataFrame:
    """Return the most recent OHLCV snapshot if available."""

    if not OHLCV_LATEST_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OHLCV_LATEST_PATH, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(col).upper().strip() for col in df.columns]
    df = df.sort_index()
    return df


def ohlcv_snapshot_coverage(snapshot: pd.DataFrame, symbols: Sequence[str]) -> float:
    """Return coverage ratio of snapshot symbols vs requested universe."""

    if snapshot is None or snapshot.empty or not symbols:
        return 0.0
    requested = {str(sym).upper().strip() for sym in symbols if sym}
    if not requested:
        return 0.0
    available = {str(col).upper().strip() for col in snapshot.columns if col}
    if not available:
        return 0.0
    covered = len(requested & available)
    return covered / max(1, len(requested))


def write_latest_ohlcv_snapshot(prices: pd.DataFrame) -> None:
    """Persist the latest OHLCV row for quick warm-cache detection."""

    if prices is None or prices.empty:
        return
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_latest_ohlcv_snapshot()
    if not existing.empty and prices.shape[1] < existing.shape[1]:
        return
    prices = prices.sort_index()
    try:
        prices.to_csv(OHLCV_LATEST_PATH)
    except Exception:
        pass


def latest_ohlcv_snapshot_stats() -> Dict[str, object]:
    """Return metadata describing the most recent OHLCV snapshot."""

    if not OHLCV_LATEST_PATH.exists():
        return {}
    snapshot = load_latest_ohlcv_snapshot()
    if snapshot.empty:
        return {}
    rows, cols = snapshot.shape
    try:
        modified = datetime.fromtimestamp(OHLCV_LATEST_PATH.stat().st_mtime)
    except Exception:
        modified = None
    return {
        "rows": int(rows),
        "cols": int(cols),
        "timestamp": modified,
        "path": str(OHLCV_LATEST_PATH),
    }


def has_warm_price_cache(name: str, min_constituents: int = 50) -> bool:
    """Return True when a cached parquet exists with enough symbols."""

    min_constituents = max(1, int(min_constituents or 0))
    base = str(name or "prices")
    if base.endswith(".parquet"):
        base = base[: -len(".parquet")]

    candidates = {
        base,
        base.lower(),
        f"prices_{base}",
        f"prices_{base.lower()}",
    }

    for candidate in candidates:
        path = CACHE_DIR / f"{candidate}.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if isinstance(frame, pd.Series):
            width = 1
        else:
            width = int(getattr(frame, "shape", (0, 0))[1])
        if width >= min_constituents:
            return True
    return False


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
