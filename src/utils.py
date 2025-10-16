import pandas as pd
from pathlib import Path
from .config import SP500_CSV, REF_DIR

_MINIMAL_SP500_CSV = """symbol,name,sector
AAPL,Apple Inc.,Information Technology
MSFT,Microsoft Corporation,Information Technology
AMZN,Amazon.com Inc.,Consumer Discretionary
GOOGL,Alphabet Inc. Class A,Communication Services
NVDA,NVIDIA Corporation,Information Technology
SPY,SPDR S&P 500 ETF Trust,ETF
"""

def _ensure_sp500_csv():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    if not SP500_CSV.exists() or SP500_CSV.stat().st_size == 0:
        # Auto-create a minimal CSV so first run never crashes
        SP500_CSV.write_text(_MINIMAL_SP500_CSV)

def load_sp500_symbols() -> pd.DataFrame:
    _ensure_sp500_csv()
    try:
        df = pd.read_csv(SP500_CSV)
    except Exception:
        # If somehow unreadable, rewrite minimal file and try again
        SP500_CSV.write_text(_MINIMAL_SP500_CSV)
        df = pd.read_csv(SP500_CSV)
    df["symbol"] = df["symbol"].str.upper().str.strip()
    return df
