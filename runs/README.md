# Runs Artifacts

This directory captures artifacts generated during weekly cycles, backtests, and
experimentation:

- `backtest_results/`: JSON summaries emitted by the backtester for each run.
- `universe_bandit.json`: Posterior state for the net-alpha bandit selector.
- Other telemetry files (e.g., metrics history) may be written here by the
  Streamlit app during operation.

The test suite removes ephemeral artifacts, but when running the app manually
these files provide a convenient audit trail of the system state.
