from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import metrics, portfolio, risk
from .config import RUNS_DIR


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
    ) -> None:
        self.prices = prices.sort_index()
        self.strategy_fn = strategy_fn
        self.config = config
        self.adv = adv

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

    def run(self, log: bool = True) -> Dict[str, object]:
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

        gross_returns: List[pd.Series] = []
        net_returns: List[pd.Series] = []
        benchmark_returns = returns.get(self.config.benchmark, pd.Series(0.0, index=returns.index))

        prev_weights: Optional[pd.Series] = None
        equity_curve = pd.Series(1.0, index=[returns.index[0]])

        for start, end in zip(schedule[:-1], schedule[1:]):
            history_window = self.prices.loc[:start].tail(self.config.lookback_days)
            if history_window.empty:
                continue

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
            asset_betas = risk.estimate_asset_betas(hist_returns, benchmark_col=self.config.benchmark)
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

            gross = period_returns.reindex(columns=weights.index, fill_value=0.0).mul(weights, axis=1).sum(axis=1)
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
            regime_trace[start] = {"trend": 1.0, "mean_reversion": 0.0}
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
