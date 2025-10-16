import numpy as np
import pandas as pd
import pytest

from src import risk


def test_transaction_cost_scales_with_turnover():
    base = risk.transaction_cost(0.5, adv=10_000_000, base_bps=5)
    higher = risk.transaction_cost(1.0, adv=10_000_000, base_bps=5)
    assert higher > base > 0


def test_vol_target_and_beta_limit():
    returns = pd.Series(np.linspace(-0.01, 0.02, 10))
    weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
    asset_betas = pd.Series({"AAA": 1.4, "BBB": 0.8})

    adj_w, scale = risk.apply_vol_target(weights, returns, target_vol=0.10)
    assert scale <= 3.0
    assert pytest.approx(adj_w.sum(), rel=1e-6) == weights.sum() * scale

    capped_w, beta_scale = risk.enforce_beta_limit(adj_w, asset_betas, limit=1.2)
    port_beta = float((capped_w * asset_betas).sum())
    assert abs(port_beta) <= 1.2 + 1e-6
    assert beta_scale <= 1.0


def test_drawdown_scale_and_overlays():
    curve = pd.Series([1.0, 1.05, 0.9, 0.92])
    scale = risk.drawdown_scale(curve, threshold=0.08)
    assert 0.35 <= scale <= 1.0

    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    returns = pd.Series([0.01, -0.005, 0.002])
    betas = pd.Series({"AAA": 1.1, "BBB": 0.9})
    adjusted, state = risk.apply_overlays(
        weights,
        returns,
        betas,
        curve,
        target_vol=0.15,
        beta_limit=1.2,
        drawdown_limit=0.05,
    )
    overlay_dict = state.as_dict()
    assert isinstance(overlay_dict, dict)
    assert all(0.0 <= value <= 3.0 for value in overlay_dict.values())
    assert adjusted.sum() <= weights.sum() * 1.5
