import pandas as pd
import pytest

from src.portfolio import apply_single_name_cap
from src.universe_selector import adaptive_top_k


def test_single_name_cap_enforces_bounds():
    weights = pd.Series({f"T{i:02d}": 0.05 + 0.02 * i for i in range(1, 13)})

    capped = apply_single_name_cap(weights, cap=0.10)

    assert capped.sum() == pytest.approx(1.0)
    assert (capped >= -1e-9).all()
    assert (capped <= 0.10 + 1e-9).all()
    assert capped.index.tolist() == capped.sort_values(ascending=False).index.tolist()


def test_adaptive_top_k_scaling():
    assert adaptive_top_k(0) == 0
    assert adaptive_top_k(10) == 10  # cannot exceed universe size
    assert adaptive_top_k(100) == 25  # floor of 25
    assert adaptive_top_k(800) == 60  # capped at 60
