from pathlib import Path

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
REF_DIR = DATA_DIR / "reference"
SP500_CSV = REF_DIR / "sp500.csv"

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
