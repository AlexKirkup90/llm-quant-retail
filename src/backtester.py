from __future__ import annotations

import json
import logging
import signal
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import metrics, portfolio, risk
from .universe_registry import load_universe_normalized
from .config import RUNS_DIR


logger = logging.getLogger(__name__)


def _run_with_timeout(
    func: Callable[[], object],
    timeout_s: float,
) -> Tuple[Optional[object], bool]:
    """Execute ``func`` with a hard timeout.

    Returns a tuple of (result, timed_out).
    """

    timeout_s = float(timeout_s)
    if timeout_s <= 0:
        return func(), False

    if hasattr(signal, "SIGALRM"):
        timed_out = False

        def _handler(signum, frame):  # type: ignore[override]
            raise TimeoutError()

        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
            else:  # pragma: no cover - fallback for environments without setitimer
                signal.alarm(int(np.ceil(timeout_s)))
            try:
                return func(), False
            except TimeoutError:
                timed_out = True
                return None, timed_out
            finally:
                if hasattr(signal, "setitimer"):
                    signal.setitimer(signal.ITIMER_REAL, 0)
                else:  # pragma: no cover
                    signal.alarm(0)
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    result: Dict[str, object] = {}
    error: Dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["value"] = func()
        except BaseException as exc:  # pragma: no cover - re-raised in main thread
            error["exc"] = exc

    thread = threading.Thread(target=_target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return None, True
    if "exc" in error:
        raise error["exc"]
    return result.get("value"), False


@dataclass
class BacktestConfig:
    universe: str
    spec_version: str
    rebalance_frequency: str = "W-FRI"
    lookback_days: int = 252
    target_vol: Optional[float] = 0.15
    beta_limit: Optional[float] = 1.2
    drawdown_limit: Optional[float] = 0.20
    base_bps: float = 5.0
    benchmark: str = "SPY"


class WalkForwardBacktester:
    """Simple walk-forward engine for weekly portfolio simulation."""

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], pd.Series],
        config: BacktestConfig,
        adv: Optional[pd.Series] = None,
        residual_fn: Optional[Callable[[pd.DataFrame, Optional[pd.Series]], pd.Series]] = None,
        regime_blend_fn: Optional[
            Callable[[pd.Series, pd.Timestamp, pd.DataFrame], Tuple[pd.Series, Dict[str, float]]]
        ] = None,
        timeout_seconds: float = 8.0,
        progress_interval: int = 5,
    ) -> None:
        self.prices = prices.sort_index()
        self.strategy_fn = strategy_fn
        self.config = config
        self.adv = adv
        self.residual_fn = residual_fn
        self.regime_blend_fn = regime_blend_fn
        self.timeout_seconds = float(timeout_seconds)
        self.progress_interval = max(1, int(progress_interval))

    def _ensure_returns(self) -> pd.DataFrame:
        closes = self.prices.ffill().dropna(how="all")
        returns = closes.pct_change().dropna(how="all")
        return returns

    def _rebalance_schedule(self, returns: pd.DataFrame) -> List[pd.Timestamp]:
        freq = self.config.rebalance_frequency
        if not freq:
            freq = "W-FRI"
        sampled = returns.resample(freq).last().dropna(how="all")
        schedule = sampled.index.tolist()
        if not schedule:
            schedule = [returns.index[-1]]
        if schedule[-1] != returns.index[-1]:
            schedule.append(returns.index[-1])
        return schedule

    def _limit_residual_universe(self, returns: pd.DataFrame) -> pd.DataFrame:
        benchmark = self.config.benchmark
        columns = pd.Index(returns.columns)
        candidates = columns.drop(benchmark) if benchmark in columns else columns

        if len(candidates) <= 200:
            selected = candidates
        elif self.adv is not None and not self.adv.empty:
            adv_slice = self.adv.reindex(candidates).fillna(0.0)
            selected = adv_slice.sort_values(ascending=False).head(200).index
        else:
            proxy = returns.reindex(columns=candidates).abs().mean().fillna(0.0)
            selected = proxy.sort_values(ascending=False).head(200).index

        if benchmark in columns:
            selected = pd.Index(selected).union(pd.Index([benchmark]))

        return returns.reindex(columns=selected).dropna(how="all", axis=1)

    def run(
        self,
        log: bool = True,
        should_abort: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, object]:
        returns = self._ensure_returns()
        schedule = self._rebalance_schedule(returns)
        if len(schedule) < 2:
            raise ValueError("Not enough data to perform walk-forward backtest")

        weights_history: Dict[pd.Timestamp, pd.Series] = {}
        turnover_series: Dict[pd.Timestamp, float] = {}
        cost_series: Dict[pd.Timestamp, float] = {}
        overlay_trace: Dict[str, Dict[str, float]] = {}
        overlay_scalers: Dict[pd.Timestamp, float] = {}
        beta_history: Dict[pd.Timestamp, float] = {}
        ic_trace: Dict[pd.Timestamp, float] = {}
        hit_trace: Dict[pd.Timestamp, float] = {}
        regime_trace: Dict[pd.Timestamp, Dict[str, float]] = {}
        residual_trace: Dict[pd.Timestamp, pd.Series] = {}

        gross_returns: List[pd.Series] = []
        net_returns: List[pd.Series] = []
        benchmark_returns = returns.get(self.config.benchmark, pd.Series(0.0, index=returns.index))

        prev_weights: Optional[pd.Series] = None
        equity_curve = pd.Series(1.0, index=[returns.index[0]])

        n_steps = len(schedule) - 1
        aborted = False

        for i, (start, end) in enumerate(zip(schedule[:-1], schedule[1:])):
            if should_abort and should_abort():
                aborted = True
                break

            history_window = self.prices.loc[:start].tail(self.config.lookback_days)
            if history_window.empty:
                continue

            try:
                weights = self.strategy_fn(history_window)
                if not isinstance(weights, pd.Series) or weights.empty:
                    continue
                weights = weights.reindex(history_window.columns).fillna(0.0)
                weights = weights.clip(lower=0.0)
                if weights.sum() == 0:
                    continue

                weights = weights / weights.sum()
                weights = portfolio.apply_single_name_cap(weights)

                hist_returns = history_window.pct_change().dropna()
                asset_betas = risk.estimate_asset_betas(
                    hist_returns, benchmark_col=self.config.benchmark
                )
                port_hist = hist_returns.reindex(columns=weights.index, fill_value=0.0)
                port_hist = port_hist.mul(weights, axis=1).sum(axis=1)

                weights, overlay_state = risk.apply_overlays(
                    weights,
                    port_hist,
                    asset_betas,
                    equity_curve,
                    target_vol=self.config.target_vol,
                    beta_limit=self.config.beta_limit,
                    drawdown_limit=self.config.drawdown_limit,
                )
                overlay_trace[start.isoformat()] = overlay_state.as_dict()
                overlay_scalers[start] = float(overlay_state.vol_scale)

                if self.residual_fn is not None:
                    window_returns = history_window.pct_change().dropna(how="all")
                    if not window_returns.empty:
                        capped_returns = self._limit_residual_universe(window_returns)
                        bench_series = window_returns.get(self.config.benchmark)

                        def _residual_job() -> Optional[pd.Series]:
                            return self.residual_fn(capped_returns, bench_series)

                        residual_output, timed_out = _run_with_timeout(
                            _residual_job, self.timeout_seconds
                        )
                        if timed_out:
                            logger.warning(
                                "Residual regression timed out at %s; skipping window.",
                                start.isoformat(),
                            )
                        elif isinstance(residual_output, pd.Series):
                            residual_trace[start] = residual_output

                regime_state: Dict[str, float] = {"trend": 1.0, "mean_reversion": 0.0}
                if self.regime_blend_fn is not None:

                    def _blend_job() -> Tuple[pd.Series, Dict[str, float]]:
                        return self.regime_blend_fn(weights.copy(), start, history_window)

                    blend_result, timed_out = _run_with_timeout(
                        _blend_job, self.timeout_seconds
                    )
                    if timed_out:
                        logger.warning(
                            "Regime blend timed out at %s; skipping window.",
                            start.isoformat(),
                        )
                    elif isinstance(blend_result, tuple) and len(blend_result) == 2:
                        new_weights, regime_payload = blend_result
                        if isinstance(new_weights, pd.Series) and new_weights.sum() > 0:
                            weights = new_weights.clip(lower=0.0)
                            if weights.sum() > 0:
                                weights = weights / weights.sum()
                            weights = portfolio.apply_single_name_cap(weights)
                        if isinstance(regime_payload, dict):
                            regime_state = regime_payload

                period_mask = (returns.index > start) & (returns.index <= end)
                period_returns = returns.loc[period_mask]
                if period_returns.empty:
                    continue

                weights_history[start] = weights

                beta_vector = asset_betas.reindex(weights.index).fillna(0.0)
                beta_history[start] = float((weights * beta_vector).sum())

                turnover_val = portfolio.turnover(prev_weights, weights)
                turnover_series[start] = float(turnover_val)

                if self.adv is not None:
                    adv_values = self.adv.reindex(weights.index).fillna(self.adv.median())
                else:
                    adv_values = pd.Series(1.0, index=weights.index)
                avg_adv = float(np.maximum(adv_values.mean(), 1e-6))
                cost = risk.transaction_cost(turnover_val, avg_adv, base_bps=self.config.base_bps)
                cost_series[start] = float(cost)

                gross = period_returns.reindex(columns=weights.index, fill_value=0.0).mul(
                    weights, axis=1
                ).sum(axis=1)
                net = gross.copy()
                if not net.empty:
                    net.iloc[0] -= cost

                gross_returns.append(gross)
                net_returns.append(net)

                cumulative = (1 + net).cumprod()
                equity_curve = equity_curve.reindex(equity_curve.index.union(cumulative.index))
                equity_curve.update(cumulative * equity_curve.iloc[-1])

                last_period = period_returns.iloc[-1]
                ic_val = metrics.spearman_ic(weights, last_period)
                hit_val = metrics.hit_rate(weights, last_period)
                ic_trace[start] = float(ic_val) if pd.notna(ic_val) else float("nan")
                hit_trace[start] = float(hit_val) if pd.notna(hit_val) else float("nan")
                regime_trace[start] = regime_state
                if pd.notna(ic_val) or pd.notna(hit_val):
                    metrics.append_ic_metric(
                        {
                            "date": start.isoformat(),
                            "ic": float(ic_val) if pd.notna(ic_val) else float("nan"),
                            "hit_rate": float(hit_val) if pd.notna(hit_val) else float("nan"),
                            "universe": self.config.universe,
                            "benchmark": self.config.benchmark,
                        }
                    )

                prev_weights = weights
            finally:
                if progress_callback:
                    progress_callback(i, n_steps)
                if (i % self.progress_interval) == 0 or i == n_steps - 1:
                    print(
                        f"[backtester] window {i + 1}/{n_steps} done",
                        flush=True,
                    )

        if not gross_returns:
            raise ValueError("Strategy did not produce any valid rebalances")

        gross_series = pd.concat(gross_returns).sort_index()
        net_series = pd.concat(net_returns).sort_index()
        equity_curve = (1 + net_series).cumprod()
        turnover_series = pd.Series(turnover_series).sort_index()
        cost_series = pd.Series(cost_series).sort_index()

        bench_series = benchmark_returns.reindex(net_series.index).fillna(0.0)

        summary = metrics.performance_summary(
            net_series,
            bench_series,
            turnover_series,
            cost_series,
            gross_returns=gross_series,
        )

        summary["beta_vs_benchmark"] = metrics.beta_vs_bench(
            net_series, self.prices, self.config.benchmark
        )

        avg_overlay = float(np.mean(list(overlay_scalers.values()))) if overlay_scalers else 1.0
        avg_beta = float(np.mean(list(beta_history.values()))) if beta_history else 0.0
        summary["overlay_scaler_mean"] = avg_overlay
        summary["portfolio_beta_mean"] = avg_beta
        summary["vol_realized"] = summary.get("volatility")
        if self.config.base_bps > 0 and summary.get("total_cost", 0.0) <= 0 and not cost_series.empty:
            summary["total_cost"] = float(cost_series.sum())

        result = {
            "gross_returns": gross_series,
            "net_returns": net_series,
            "equity_curve": equity_curve,
            "turnover": turnover_series,
            "costs": cost_series,
            "summary": summary,
            "overlays": overlay_trace,
            "overlay_scalers": pd.Series(overlay_scalers).sort_index(),
            "portfolio_beta": pd.Series(beta_history).sort_index(),
            "ic": pd.Series(ic_trace).sort_index(),
            "hit_rate": pd.Series(hit_trace).sort_index(),
            "regime_blend": regime_trace,
            "residual_alpha": {ts: residual for ts, residual in residual_trace.items()},
            "aborted": aborted,
        }

        if log:
            log_dir = RUNS_DIR / "backtest_results"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            name = f"{self.config.universe}_{self.config.spec_version}_{timestamp}.json"
            payload = {
                "config": asdict(self.config),
                "summary": summary,
                "overlays": overlay_trace,
                "stats": {
                    "gross": gross_series.describe().to_dict(),
                    "net": net_series.describe().to_dict(),
                },
                "overlay_scalers": {
                    ts.isoformat(): float(val) for ts, val in overlay_scalers.items()
                },
                "portfolio_beta": {
                    ts.isoformat(): float(val) for ts, val in beta_history.items()
                },
            }
            (log_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True))
            result["log_path"] = str(log_dir / name)

        return result
def prepare_backtest_universe(
    universe_name: str,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Load and normalise a backtest universe with benchmark appended."""

    universe_df = load_universe_normalized(universe_name, apply_filters=True).copy()
    symbols = (
        universe_df["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not symbols:
        raise ValueError(f"No symbols resolved for universe {universe_name}")

    bench = metrics.default_benchmark(universe_name)
    if bench not in symbols:
        symbols.append(bench)

    return universe_df, symbols, bench
