import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from .utils import load_sp500_symbols
from .config import CACHE_DIR

def fetch_prices(symbols, years=5, interval="1d") -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=365*years)).strftime("%Y-%m-%d")
    tickers = " ".join(symbols)
    df = yf.download(tickers=tickers, start=start, interval=interval, auto_adjust=True, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all").sort_index()
    df.columns = [c.replace(" ", "") for c in df.columns]
    return df

def load_universe() -> pd.DataFrame:
    return load_sp500_symbols()

def cache_parquet(df: pd.DataFrame, name: str) -> str:
    path = CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return str(path)
