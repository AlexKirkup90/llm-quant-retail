import pandas as pd
from .config import SP500_CSV

def load_sp500_symbols() -> pd.DataFrame:
    df = pd.read_csv(SP500_CSV)
    df["symbol"] = df["symbol"].str.upper().str.strip()
    return df

def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)
