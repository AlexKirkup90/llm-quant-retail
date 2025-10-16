import pandas as pd
import pytest

from src.universe import load_universe, MIN_PRICE, MIN_ADV

@pytest.mark.parametrize("mode", ["SP500", "SP500_FULL", "R1000"])
def test_load_universe_filters_and_not_empty(mode):
    df = load_universe(mode)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "symbol" in df.columns
    if {"close", "adv_usd"}.issubset(df.columns):
        assert (df["close"] >= MIN_PRICE).all()
        assert (df["adv_usd"] >= MIN_ADV).all()


def test_low_liquidity_names_removed():
    df = load_universe("R1000")
    symbols = df["symbol"].tolist()
    assert "GE" not in symbols  # price < MIN_PRICE
    assert "BB" not in symbols  # ADV < MIN_ADV
    assert "MMM" not in symbols  # ADV < MIN_ADV
