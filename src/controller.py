import json
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
from src import (
    dataops,
    features,
    signals,
    portfolio,
    metrics,
    report,
    memory,
    universe_registry,
    universe_selector,
    explain,
    risk,
    regime,
    ui_utils,
)
from src.features import velocity as feature_velocity
from src.signals import residuals as residual_signals
from src.universe_registry import (
    UniverseRegistryError,
    load_universe_normalized,
    refresh_universe,
)
from src.ui_utils import (
    get_universe_choices,
    _resolve_symbols_from_universe,
    _apply_runtime_cap,
    _format_cap_status,
    _symbol_cache_key,
    LABELS,
    BENCHMARK_LABELS,
    default_benchmark,
)

class Controller:
    def __init__(self):
        if not hasattr(st, "session_state"):
            st.session_state = {}

        self.spec_path = Path("spec/current_spec.json")
        self.spec_data = json.loads(self.spec_path.read_text()) if self.spec_path.exists() else {}
        self.spec_version = str(self.spec_data.get("version", "0.6"))

    def k(self, *parts: str) -> str:
        return "__".join(str(p) for p in parts)

    def run_weekly_cycle(self, as_of, universe_mode, manual_universe, apply_filters, runtime_cap, bypass_cap_if_cache_warm,
                         turnover_cap, rebalance_band, sector_neutral, bandit_enabled, enable_velocity,
                         enable_residual, enable_adaptive_ic, enable_regime_blend):
        with st.spinner("Running weekly workflow..."):
            try:
                metrics_history_path = Path("metrics_history.json")
                decision_info = None
                universe_choices = get_universe_choices()
                default_index = universe_choices.index("SP500_FULL") if "SP500_FULL" in universe_choices else 0
                selected_universe_name = manual_universe or universe_choices[default_index]
                if universe_mode == "auto":
                    selection_cfg = self.spec_data.get("universe_selection", {}) or {}
                    candidates = selection_cfg.get("candidates") or universe_choices
                    constraints = selection_cfg.get("constraints", {}) or {}
                    decision_info = universe_selector.choose_universe(
                        list(candidates),
                        constraints,
                        lambda name: load_universe_normalized(name, apply_filters=True),
                        metrics_history_path,
                        self.spec_data,
                        str(as_of),
                        bandit_enabled=bandit_enabled,
                    )
                    selected_universe_name = decision_info.get("winner", candidates[0])

                st.subheader("Universe selection")
                st.write(f"Selected universe: **{selected_universe_name}**")
                if decision_info:
                    self.display_decision_info(decision_info)

                uni, sector_lookup = self.load_universe_data(selected_universe_name, apply_filters)
                if uni is None:
                    return

                symbols = _resolve_symbols_from_universe(uni, selected_universe_name)
                if not symbols:
                    st.error(f"No symbols resolved for universe {selected_universe_name}.")
                    st.stop()

                self.display_universe_size(uni, apply_filters)

                symbols_effective, benchmark_symbol, benchmark_label = self.prepare_symbols_and_benchmark(symbols, selected_universe_name, runtime_cap, bypass_cap_if_cache_warm)

                prices = self.fetch_price_data(symbols_effective, as_of)

                st.write("Downloaded prices:", prices.shape)

                feats, feature_history = self.calculate_features(prices, enable_velocity)

                w_ridge = self.calculate_signal_weights(prices, feats, feature_history, sector_lookup, enable_residual, enable_adaptive_ic, as_of, selected_universe_name, benchmark_symbol)

                scores = signals.score_current(
                    feats,
                    w_ridge,
                    sector_map=sector_lookup.reindex(feats.index) if sector_lookup is not None else None,
                    sector_neutral=sector_neutral,
                )
                if scores.empty:
                    scores = pd.Series(np.random.randn(len(feats.index)), index=feats.index)

                self.update_explainability_payload(as_of, selected_universe_name, scores, w_ridge, feats, prices, benchmark_symbol)

                w_final = self.construct_portfolio(prices, scores, sector_lookup, enable_regime_blend, turnover_cap, rebalance_band)

                port = self.save_portfolio(as_of, w_final)

                self.display_portfolio_metrics(prices, w_final, benchmark_symbol, benchmark_label, as_of, sector_lookup)

                self.provide_downloads(port, prices, w_final, as_of, benchmark_symbol, selected_universe_name, sector_neutral, bandit_enabled, decision_info)

                st.success("Weekly portfolio created and metrics logged.")

            except Exception as exc:
                st.error(f"Run failed: {exc}")
                st.stop()

    def display_decision_info(self, decision_info):
        metrics_table = decision_info["metrics"].copy()
        metrics_table = metrics_table.reindex(decision_info.get("candidates", metrics_table.index))
        display_cols = ["alpha", "sortino", "mdd", "coverage", "turnover_cost", "n_weeks"]
        for col in display_cols:
            if col not in metrics_table.columns:
                metrics_table[col] = np.nan
        summary_df = metrics_table[display_cols].copy()
        scores = pd.Series(decision_info.get("scores", {}))
        probs = pd.Series(decision_info.get("probabilities", {}))
        summary_df["score"] = scores.reindex(summary_df.index)
        summary_df["probability"] = probs.reindex(summary_df.index)
        st.dataframe(
            summary_df.fillna(0.0).style.format({
                "alpha": "{:.4f}", "sortino": "{:.2f}", "mdd": "{:.2%}",
                "coverage": "{:.1%}", "turnover_cost": "{:.4%}",
                "probability": "{:.1%}", "score": "{:.4f}",
            })
        )
        st.caption(decision_info.get("rationale", ""))
        bandit_info = decision_info.get("bandit", {})
        if bandit_info:
            st.write("Bandit status:")
            st.json(bandit_info)

    def load_universe_data(self, universe_name, apply_filters):
        try:
            uni = load_universe_normalized(universe_name, apply_filters=bool(apply_filters))
        except UniverseRegistryError:
            st.warning(f"Universe cache looked invalid; refreshing and retrying {universe_name}.")
            try:
                refresh_universe(universe_name, force=True)
                uni = load_universe_normalized(universe_name, apply_filters=bool(apply_filters))
            except Exception as exc:
                st.error(f"Failed to load {universe_name}: {exc}")
                return None, None
        except Exception as exc:
            st.error(f"Failed to load {universe_name}: {exc}")
            return None, None

        attrs = dict(getattr(uni, "attrs", {}))
        uni.attrs.update(attrs)
        if not apply_filters:
            uni.attrs["universe_filter_meta"] = {
                "raw_count": len(uni),
                "filtered_count": len(uni),
                "reason": "Liquidity filters bypassed via apply_filters=False.",
                "filters_applied": False,
            }

        sector_lookup = None
        if "sector" in uni.columns:
            with pd.option_context("mode.use_inf_as_na", True):
                sector_lookup = (
                    uni.set_index("symbol")["sector"].astype(str).str.upper().replace("", pd.NA)
                )
        return uni, sector_lookup

    def display_universe_size(self, uni, apply_filters):
        filter_meta = uni.attrs.get("universe_filter_meta", {})
        raw_count = int(filter_meta.get("raw_count", len(uni)))
        filtered_count = int(filter_meta.get("filtered_count", len(uni)))
        st.write(f"Universe size — raw: **{raw_count}**, post-filter: **{filtered_count}**")
        if filter_meta.get("reason"):
            st.info(filter_meta.get("reason"))

    def prepare_symbols_and_benchmark(self, symbols, universe_name, runtime_cap, bypass_cap_if_cache_warm):
        min_expected = universe_registry.expected_min_constituents(universe_name)
        cache_key = _symbol_cache_key(universe_name)
        warm_price_cache = dataops.has_warm_price_cache(cache_key, min_expected)

        cache_warm_key = self.k("weekly", "cache_warm")
        snapshot_df = dataops.load_latest_ohlcv_snapshot()
        snapshot_coverage = dataops.ohlcv_snapshot_coverage(snapshot_df, symbols)
        if snapshot_coverage >= 0.80:
            st.session_state[cache_warm_key] = True
        cache_warm_flag = bool(st.session_state.get(cache_warm_key, False))

        effective_warm = warm_price_cache or cache_warm_flag
        symbols_effective, _ = _apply_runtime_cap(
            symbols, runtime_cap, effective_warm, bypass_cap_if_cache_warm
        )
        bypass_active = effective_warm and bypass_cap_if_cache_warm
        display_count, cap_label = _format_cap_status(len(symbols), runtime_cap, bypass_active)
        st.write(f"Effective universe for data fetch: **{display_count}** (cap applied: {cap_label})")

        if bypass_active and cache_warm_flag:
            snapshot_rows = int(getattr(snapshot_df, "shape", (0, 0))[0])
            snapshot_cols = int(getattr(snapshot_df, "shape", (0, 0))[1])
            st.success(f"Warm price cache detected: {snapshot_rows} rows × {snapshot_cols} symbols")
        elif warm_price_cache:
            st.success("Warm price cache detected.")

        if not symbols_effective:
            st.error(f"No symbols resolved for universe {universe_name}.")
            st.stop()

        bench = default_benchmark(universe_name)
        if bench not in symbols_effective:
            symbols_effective.append(bench)

        benchmark_symbol = bench
        benchmark_label = BENCHMARK_LABELS.get(benchmark_symbol, benchmark_symbol)
        return symbols_effective, benchmark_symbol, benchmark_label

    def fetch_price_data(self, symbols, as_of):
        years = self.spec_data.get("data", {}).get("price_years", 5)
        try:
            prices = dataops.fetch_prices(symbols, years=years)
            cache_key = _symbol_cache_key(str(as_of.year))
            dataops.cache_parquet(prices, cache_key)
            dataops.write_latest_ohlcv_snapshot(prices)
        except Exception as exc:
            st.warning(f"Price fetch failed ({exc}); generating fallback series.")
            idx = pd.date_range(end=pd.Timestamp(as_of), periods=252 * 5, freq="B")
            random_walk = np.cumprod(1 + np.random.randn(len(idx), len(symbols)) * 0.001, axis=0)
            prices = pd.DataFrame(random_walk, index=idx, columns=symbols)
            dataops.write_latest_ohlcv_snapshot(prices)

        if prices.empty:
            st.error("No price data available after fetch.")
            st.stop()
        return prices

    def calculate_features(self, prices, enable_velocity):
        try:
            feats = features.combine_features(prices)
        except Exception:
            feats = pd.DataFrame(index=prices.columns)
        feats = feats.fillna(0.0)

        feature_history = {}
        history_dates = prices.index[-signals.ROLLING_WEEKS:]
        for ts in history_dates:
            price_slice = prices.loc[:ts]
            if not price_slice.empty:
                try:
                    feature_snapshot = features.combine_features(price_slice)
                    feature_history[ts] = feature_snapshot
                except Exception:
                    continue

        if enable_velocity and feature_history:
            velocity_cfg = self.spec_data.get("signals", {}).get("velocity", {})
            history_panel = pd.concat({ts: df for ts, df in feature_history.items()}, names=["date", "symbol"])
            try:
                velocity_panel = feature_velocity.build_velocity_features(history_panel, velocity_cfg.get("windows", {}))
                if not velocity_panel.empty:
                    for ts in list(feature_history.keys()):
                        try:
                            vel_snapshot = velocity_panel.xs(ts, level=0)
                            feature_history[ts] = feature_history[ts].join(vel_snapshot, how="left")
                        except Exception:
                            continue
                    try:
                        latest_ts = sorted(feature_history.keys())[-1]
                        feats = feats.join(velocity_panel.xs(latest_ts, level=0), how="left")
                    except Exception:
                        pass
            except Exception:
                pass
        feats = feats.fillna(0.0)
        return feats, feature_history

    def calculate_signal_weights(self, prices, feats, feature_history, sector_lookup, enable_residual, enable_adaptive_ic, as_of, universe_name, benchmark_symbol):
        returns = prices.pct_change().dropna(how="all")
        bench_returns = returns.get(benchmark_symbol, pd.Series(0.0, index=returns.index)).fillna(0.0)
        try:
            fwd5 = (1 + returns).rolling(5).apply(lambda x: x.prod() - 1, raw=True).shift(-5)
            fwd5 = fwd5.iloc[:-5] if len(fwd5) >= 5 else fwd5.iloc[0:0]
        except Exception:
            fwd5 = pd.DataFrame(index=returns.index, columns=returns.columns)

        residual_target = None
        if enable_residual:
            try:
                residual_target = residual_signals.compute_residual_returns(returns, sector_lookup, bench_returns)
            except Exception:
                pass

        try:
            w_ridge = signals.fit_rolling_ridge(fwd5, feature_history)
            if w_ridge.empty:
                target = fwd5.iloc[-1].fillna(0.0) if not fwd5.empty else pd.Series(0.0, index=feats.index)
                w_ridge = signals.fit_ridge(feats, target)
        except Exception:
            w_ridge = pd.Series(dtype=float)

        if enable_residual and residual_target is not None and not residual_target.empty:
            aligned_target = residual_target.reindex(feats.index).fillna(0.0)
            if aligned_target.abs().sum() > 0:
                try:
                    w_ridge = signals.fit_ridge(feats, aligned_target)
                except Exception:
                    pass

        ic_ema_series = None
        adaptive_cfg = self.spec_data.get("signals", {}).get("adaptive_ic", {})
        if enable_adaptive_ic and not fwd5.empty:
            try:
                latest_target = fwd5.iloc[-1]
                feature_ic_snapshot = metrics.feature_ic_snapshot(feats, latest_target)
                existing_ic = signals.load_feature_ic_ema()
                ic_ema_series = signals.update_feature_ic_ema(existing_ic, feature_ic_snapshot, ema_lambda=float(adaptive_cfg.get("ema_lambda", 0.9)))
                signals.save_feature_ic_ema(ic_ema_series)
                w_ridge = signals.apply_ic_weighting(w_ridge, ic_ema_series, alpha_ic=float(adaptive_cfg.get("alpha_ic", 0.2)), clip=float(adaptive_cfg.get("clip", 0.5)))
            except Exception:
                ic_ema_series = signals.load_feature_ic_ema()

        return w_ridge

    def update_explainability_payload(self, as_of, universe_name, scores, w_ridge, feats, prices, benchmark_symbol):
        returns = prices.pct_change().dropna(how="all")
        fwd5 = (1 + returns).rolling(5).apply(lambda x: x.prod() - 1, raw=True).shift(-5)
        fwd5 = fwd5.iloc[:-5] if len(fwd5) >= 5 else fwd5.iloc[0:0]

        ic_value, hit_value = float("nan"), float("nan")
        if not fwd5.empty:
            latest_target = fwd5.iloc[-1]
            ic_value = metrics.spearman_ic(scores, latest_target)
            hit_value = metrics.hit_rate(scores, latest_target)
            evaluation_cfg = self.spec_data.get("evaluation", {})
            if evaluation_cfg.get("track_ic", True) or evaluation_cfg.get("track_hit_rate", True):
                payload = {
                    "date": str(as_of), "ic": float(ic_value), "hit_rate": float(hit_value),
                    "universe": universe_name, "benchmark": benchmark_symbol
                }
                metrics.append_ic_metric(payload)

        top_holdings = scores.head(15).index.tolist()
        st.session_state[self.k("explain", "payload")] = {
            "as_of": str(as_of), "universe": universe_name, "scores": scores,
            "attribution": explain.coef_attribution(w_ridge, top_n=10),
            "shap": explain.shap_like_contributions(feats.reindex(index=top_holdings), w_ridge, top_features=5),
            "ic_ema": signals.load_feature_ic_ema(), "ic_value": ic_value, "hit_rate": hit_value
        }

    def construct_portfolio(self, prices, scores, sector_lookup, enable_regime_blend, turnover_cap, rebalance_band):
        returns_252 = prices.pct_change().dropna(how="all").tail(252)
        top_holdings = scores.head(15).index.tolist()

        try:
            w0 = portfolio.inverse_vol_weights(returns_252, top_holdings, cap_single=0.10, k=min(15, len(top_holdings)))
        except Exception:
            w0 = pd.Series(1 / max(1, len(top_holdings)), index=top_holdings)

        try:
            w_sector = portfolio.apply_sector_caps(w0, sector_lookup, cap=0.35) if sector_lookup is not None else w0
        except Exception:
            w_sector = w0

        if enable_regime_blend:
            try:
                trend_series = (returns_252.reindex(columns=w_sector.index, fill_value=0.0).mul(w_sector, axis=1).sum(axis=1))
                trend_perf = metrics.sharpe(trend_series, periods_per_year=252)
                mr_candidates = scores.sort_values(ascending=True).head(len(w_sector))
                mr_weights = portfolio.inverse_vol_weights(returns_252, mr_candidates.index.tolist(), cap_single=0.10, k=min(15, len(mr_candidates)))
                mr_series = (returns_252.reindex(columns=mr_weights.index, fill_value=0.0).mul(mr_weights, axis=1).sum(axis=1))
                mr_perf = metrics.sharpe(mr_series, periods_per_year=252)
                blend_meta = {"trend": float(trend_perf or 0.0), "mean_reversion": float(mr_perf or 0.0)}
                w_sector = regime.blend_weights(w_sector, mr_weights, blend_meta)
            except Exception:
                pass

        last = memory.load_last_portfolio()
        last_w = pd.Series({h["ticker"]: h["weight"] for h in last["holdings"]}) if last else None

        try:
            w_final = portfolio.apply_turnover_controls(last_w, w_sector, turnover_cap=float(turnover_cap), rebalance_band=float(rebalance_band))
        except Exception:
            w_final = w_sector

        w_final = portfolio.apply_single_name_cap(w_final, cap=0.10)
        return w_final.sort_values(ascending=False)

    def save_portfolio(self, as_of, w_final):
        port = {
            "as_of": str(as_of),
            "holdings": [{"ticker": t, "weight": w} for t, w in w_final.items()],
            "cash_weight": float(max(0.0, 1.0 - w_final.sum()))
        }
        memory.save_portfolio(port)
        st.success("Weekly portfolio created.")
        st.json(port)
        return port

    def display_portfolio_metrics(self, prices, w_final, benchmark_symbol, benchmark_label, as_of, sector_lookup):
        returns_252 = prices.pct_change().dropna(how="all").tail(252)
        port_rets = (returns_252.reindex(columns=w_final.index, fill_value=0.0).mul(w_final, axis=1).sum(axis=1))
        curve = (1 + port_rets).cumprod()
        bench = returns_252.get(benchmark_symbol, pd.Series(0.0, index=returns_252.index))

        sor = metrics.sortino(port_rets)
        mdd = metrics.max_drawdown(curve)
        alpha = metrics.alpha_vs_bench(port_rets, bench)

        st.subheader("Weekly metrics")
        st.write(f"- Sortino: **{sor:.2f}**\n- Max Drawdown: **{mdd:.2%}**\n- Alpha vs {benchmark_label} (weekly mean): **{alpha:.4%}**")

        st.subheader("Risk dashboard")
        beta_col, vol_col, dd_col, scaler_col = st.columns(4)
        portfolio_beta = metrics.beta_vs_bench(port_rets, prices, benchmark_symbol)
        vol_realised = risk.annualised_volatility(port_rets)
        beta_col.metric(f"Beta vs {benchmark_label}", f"{portfolio_beta:.2f}" if pd.notna(portfolio_beta) else "n/a")
        vol_col.metric("Realised vol", f"{vol_realised:.2%}" if pd.notna(vol_realised) else "n/a")
        dd_col.metric("Max drawdown", f"{mdd:.2%}")
        scaler_col.metric("Overlay scaler", "1.00")

        if sector_lookup is not None:
            st.subheader("Sector Attribution")
            attribution_df = metrics.calculate_sector_attribution(pd.DataFrame({str(as_of): w_final}), returns_252, sector_lookup)
            st.dataframe(attribution_df)

    def provide_downloads(self, port, prices, w_final, as_of, benchmark_symbol, universe_name, sector_neutral, bandit_enabled, decision_info):
        returns = prices.pct_change().dropna(how="all")
        port_rets = (returns.reindex(columns=w_final.index, fill_value=0.0).mul(w_final, axis=1).sum(axis=1))
        curve = (1 + port_rets).cumprod()

        holdings_csv = StringIO()
        pd.DataFrame(port["holdings"]).to_csv(holdings_csv, index=False)
        st.download_button("Download holdings.csv", holdings_csv.getvalue(), "holdings.csv", "text/csv")

        equity_csv = StringIO()
        curve.to_frame("equity").to_csv(equity_csv)
        st.download_button("Download equity_curve.csv", equity_csv.getvalue(), "equity_curve.csv", "text/csv")

        summary_payload = self.create_summary_payload(prices, w_final, as_of, benchmark_symbol, universe_name, sector_neutral, bandit_enabled, decision_info)
        summary_json = json.dumps(summary_payload, indent=2)
        st.download_button("Download summary.json", summary_json, "summary.json", "application/json")

        note = f"# Weekly AI Portfolio — {as_of}\n\n- Sortino: {summary_payload['sortino']:.2f}\n- Max Drawdown: {summary_payload['max_drawdown']:.2%}\n- Alpha (vs {benchmark_symbol}, weekly mean): {summary_payload['alpha']:.4%}\n"
        out_path = report.write_markdown(note)
        with open(out_path, "rb") as handle:
            st.download_button("Download weekly report", handle.read(), Path(out_path).name)

        out_dir = Path("runs") / "backtest_results"
        out_dir.mkdir(exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        plot_path = str(out_dir / f"equity_curve_{ts}.png")
        report.generate_equity_curve_plot(curve, plot_path, benchmark_series=returns.get(benchmark_symbol))
        st.image(plot_path, caption="Equity Curve")

    def create_summary_payload(self, prices, w_final, as_of, benchmark_symbol, universe_name, sector_neutral, bandit_enabled, decision_info):
        returns = prices.pct_change().dropna(how="all")
        port_rets = (returns.reindex(columns=w_final.index, fill_value=0.0).mul(w_final, axis=1).sum(axis=1))
        curve = (1 + port_rets).cumprod()
        bench = returns.get(benchmark_symbol, pd.Series(0.0, index=returns.index))

        sor = metrics.sortino(port_rets)
        mdd = metrics.max_drawdown(curve)
        alpha = metrics.alpha_vs_bench(port_rets, bench)

        last = memory.load_last_portfolio()
        last_w = pd.Series({h['ticker']: h['weight'] for h in last['holdings']}) if last else None
        turnover_fraction = float(portfolio.turnover(last_w, w_final)) if last_w is not None else float(w_final.abs().sum())
        cost_bps_weekly = float(0.0005 * turnover_fraction * 10000.0)

        portfolio_beta = metrics.beta_vs_bench(port_rets, prices, benchmark_symbol)
        vol_realised = risk.annualised_volatility(port_rets)

        payload = st.session_state.get(self.k("explain", "payload"), {})
        ic_value = payload.get("ic_value", float("nan"))
        hit_rate = payload.get("hit_rate", float("nan"))

        return {
            "spec": self.spec_version, "as_of": str(as_of), "alpha": float(alpha),
            "net_alpha": float(alpha - 0.0005 * turnover_fraction), "sortino": float(sor),
            "max_drawdown": float(mdd), "turnover": float(turnover_fraction),
            "cost_bps_weekly": cost_bps_weekly, "portfolio_beta": float(portfolio_beta),
            "vol_realized": float(vol_realised) if pd.notna(vol_realised) else float("nan"),
            "benchmark": benchmark_symbol, "sector_neutral": bool(sector_neutral),
            "bandit_mode": bool(bandit_enabled and bool(decision_info)), "overlay_scaler": 1.0,
            "regime_blend": {}, "ic": ic_value, "hit_rate": hit_rate
        }

    def analyse_market_conditions(self):
        st.write("Analyzing market conditions...")
        self.run_weekly_cycle(
            as_of=date.today(),
            universe_mode="auto",
            manual_universe=None,
            apply_filters=True,
            runtime_cap=150,
            bypass_cap_if_cache_warm=True,
            turnover_cap=0.40,
            rebalance_band=0.25,
            sector_neutral=False,
            bandit_enabled=True,
            enable_velocity=True,
            enable_residual=True,
            enable_adaptive_ic=True,
            enable_regime_blend=True,
        )

    def analyse_current_portfolio(self):
        st.write("Analyzing current portfolio...")
        last_portfolio = memory.load_last_portfolio()
        if not last_portfolio:
            st.warning("No portfolio found.")
            return

        st.json(last_portfolio)

        holdings = pd.DataFrame(last_portfolio["holdings"])
        if "weight" in holdings.columns:
            st.bar_chart(holdings.set_index("ticker")["weight"])

    def generate_new_portfolio(self):
        st.write("Generating new portfolio...")
        self.run_weekly_cycle(
            as_of=date.today(),
            universe_mode="auto",
            manual_universe=None,
            apply_filters=True,
            runtime_cap=150,
            bypass_cap_if_cache_warm=True,
            turnover_cap=0.40,
            rebalance_band=0.25,
            sector_neutral=False,
            bandit_enabled=True,
            enable_velocity=True,
            enable_residual=True,
            enable_adaptive_ic=True,
            enable_regime_blend=True,
        )
