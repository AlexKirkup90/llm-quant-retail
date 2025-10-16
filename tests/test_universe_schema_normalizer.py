import pandas as pd

from src.universe import NORMALIZED_COLS, ensure_universe_schema


def test_index_symbols_are_promoted():
    df = pd.DataFrame({"Name": ["Alpha", "Beta"], "Sector": ["Tech", "Health"]}, index=[" aapl ", "msft"])
    df.index.name = "Ticker"

    normalized = ensure_universe_schema(df, "TEST_INDEX")

    assert normalized.columns.tolist() == NORMALIZED_COLS
    assert normalized["symbol"].tolist() == ["AAPL", "MSFT"]
    assert normalized["name"].tolist() == ["Alpha", "Beta"]
    assert normalized["sector"].tolist() == ["Tech", "Health"]


def test_mixed_case_headers_and_code_column():
    df = pd.DataFrame({
        " Code ": [" nflx ", " googl"],
        "company": ["Netflix", "Alphabet"],
        "GICS Sector": ["Communication Services", "Communication Services"],
    })

    normalized = ensure_universe_schema(df, "TEST_HEADERS")

    assert normalized["symbol"].tolist() == ["NFLX", "GOOGL"]
    assert normalized["name"].tolist() == ["Netflix", "Alphabet"]
    assert all(val == "Communication Services" for val in normalized["sector"])


def test_missing_sector_defaults_to_blank():
    df = pd.DataFrame({"Ticker": ["TSLA", "AMD"], "Security": ["Tesla", "Advanced Micro Devices"]})

    normalized = ensure_universe_schema(df, "TEST_MISSING_SECTOR")

    assert normalized["sector"].tolist() == ["", ""]
    assert normalized["name"].tolist() == ["Tesla", "Advanced Micro Devices"]
