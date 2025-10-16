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

To force-refresh everything locally run:

```bash
python scripts/refresh_universes.py
```

The script prints a JSON blob describing whether each universe was refreshed
from the live site or loaded from the existing cache.
