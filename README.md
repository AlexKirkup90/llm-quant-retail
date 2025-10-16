# llm-quant-retail

## Universe Registry

The platform ships with a registry-backed loader for the following universes:

- `SP500_MINI` (smoke-test subset)
- `SP500_FULL`
- `R1000`
- `NASDAQ_100`
- `FTSE_350`

Universe constituents are sourced from Wikipedia and cached under
`data/reference/{universe}.csv`. Cached files are automatically refreshed when
older than 90 days (configurable per universe). FTSE tickers are persisted with
an ``.L`` suffix to match LSE identifiers.

Liquidity and price filters are only applied when a recent OHLCV snapshot is
available under `data/reference/ohlcv_latest.csv`. When the snapshot is missing
or stale the app falls back to the raw constituent lists and surfaces an
informational warning.

To force-refresh everything locally run:

```bash
python scripts/refresh_universes.py
```

The script prints a JSON blob describing whether each universe was refreshed
from the live site or loaded from the existing cache.

## Universe Selection Engine (v0.4)

Version 0.4 introduces an automated universe selection engine driven by
historical telemetry stored in `metrics_history.json`. For each candidate
universe the app aggregates the most recent lookback window (default: 8 weeks)
and computes the following metrics:

- Alpha, Sortino, max drawdown (`mdd`), and hit rate
- Validation metrics (`val_alpha`, `val_sortino`)
- Observed coverage (fraction of required constituents available)
- Estimated turnover cost (currently `0.0005 * turnover_fraction`)

Scores are calculated using a weighted linear model:

```
score = w_alpha * alpha + w_sortino * sortino - w_mdd * mdd
        + w_coverage * coverage - w_turnover * turnover_cost
```

The weights, lookback window, minimum-history requirement, softmax temperature,
and per-universe minimum constituent constraints are defined in
`spec/current_spec.json`. Coverage is computed as
`min(1.0, len(current_universe) / min_constituents)` so that sparse universes are
penalised, and the softmax temperature keeps a controlled level of exploration.
Every decision (candidates, scores, probabilities, rationale) is logged to
`runs/universe_decisions.json` for auditability.

### Auto vs Manual mode

The Streamlit UI exposes a **Universe Mode** selector:

- **Auto** (default) runs the selection engine, displays a table of candidates
  with their metrics, scores, and probabilities (highlighting the winner), and
  proceeds with the chosen universe for the weekly cycle.
- **Manual** preserves the previous behaviour, letting you pick a universe from
  the dropdown.

Regardless of the mode, each run logs metrics enriched with the chosen universe
name, realised coverage, and turnover cost under `metrics_history.json`.
