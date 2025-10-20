from src import portfolio


def test_tune_rebalance_band_increases_when_turnover_high():
    history = [0.5, 0.48, 0.52, 0.46]
    tuned = portfolio.tune_rebalance_band(history, current_band=0.25)
    assert tuned > 0.25


def test_tune_rebalance_band_decreases_when_turnover_low():
    history = [0.1, 0.15, 0.2, 0.18]
    tuned = portfolio.tune_rebalance_band(history, current_band=0.3)
    assert tuned < 0.3
