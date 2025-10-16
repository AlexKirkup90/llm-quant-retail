from pathlib import Path

# Resolve project root from this file's location (robust under pytest/streamlit/CI)
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REF_DIR = DATA_DIR / "reference"
SP500_CSV = REF_DIR / "sp500.csv"

RUNS_DIR = BASE_DIR / "runs"

# Ensure dirs exist at import time (safe if they already exist)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
