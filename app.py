import json
import pathlib
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

# --- Make project root importable (works under streamlit/pytest/CI) ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.universe import ensure_universe_schema
from src.universe_registry import registry_list
from src.metrics import default_benchmark

PREFERRED = ["SP500_MINI", "SP500_FULL", "R1000", "NASDAQ_100", "FTSE_350"]
LABELS = {
    "SP500_MINI": "S&P 500 (Mini)",
    "SP500_FULL": "S&P 500",
    "R1000": "Russell 1000",
    "NASDAQ_100": "NASDAQ-100",
    "FTSE_350": "FTSE 350",
}

BENCHMARK_LABELS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "ISF.L": "ISF.L",
}


def _resolve_benchmark(universe_name: str) -> Tuple[str, str]:
    ticker = default_benchmark(universe_name)
    label = BENCHMARK_LABELS.get(ticker, ticker)
    return ticker, label


def _ensure_benchmark_symbol(symbols: Sequence[str], universe_name: str) -> Tuple[List[str], str]:
    bench = default_benchmark(universe_name)
    updated = list(symbols)
    if bench not in updated:
        updated.append(bench)
    return updated, bench


def get_universe_choices() -> List[str]:
    """Return preferred ordering combining registry and spec-defined universes."""

    choices = sorted(set(registry_list()))
    spec_path = pathlib.Path("spec/current_spec.json")
    try:
        spec = json.loads(spec_path.read_text())
        choices = sorted(set(choices) | set(spec.get("universes", [])))
    except Exception:
        pass
    ordered = [u for u in PREFERRED if u in choices]
    ordered.extend(u for u in choices if u not in PREFERRED)
    return ordered


def _clean_symbol_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
    )


def _resolve_symbols_from_universe(df: pd.DataFrame, universe_name: str = "UNKNOWN") -> List[str]:
    """Return deduplicated symbols regardless of column/index placement."""

    if df is None or df.empty:
        return []
    normalized = ensure_universe_schema(df, universe_name)
    cleaned = _clean_symbol_series(normalized["symbol"]).dropna()
    return pd.Index(cleaned).drop_duplicates().tolist()


def _apply_runtime_cap(
    symbols: Sequence[str],
    cap: int,
    cache_warm: bool,
    bypass_cap_if_warm: bool,
) -> Tuple[List[str], int]:
    """Return symbols subject to runtime cap plus the effective limit applied."""

    symbol_list = list(symbols)
    try:
        cap_value = int(cap)
    except Exception:
        cap_value = 0
    if cache_warm and bypass_cap_if_warm:
        return symbol_list, 0
    if cap_value <= 0:
        return symbol_list, 0
    effective = min(cap_value, len(symbol_list))
    return symbol_list[:effective], effective


def _symbol_cache_key(universe_name: str) -> str:
    return f"prices_{str(universe_name).lower()}".replace("/", "_")


def main():
    import streamlit as st
    import numpy as np
    from datetime import date

    from src import (
        dataops,
        features,
        signals,
        portfolio,
        metrics,
        report,
        memory,
        universe,
        universe_registry,
        universe_selector,
        explain,
        risk,
        regime,
    )
    from src.features import velocity as feature_velocity
    from src.signals import residuals as residual_signals

    if hasattr(st, "set_page_config"):
        st.set_page_config(page_title="LLM-Codex Quant — Weekly", layout="wide")
    st.title("LLM-Codex Quant — Weekly Cycle")

    if not hasattr(st, "session_state"):
        st.session_state = {}

    spec_path = Path("spec/current_spec.json")
    spec_data = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    spec_version = str(spec_data.get("version", "0.6"))
    st.caption(f"Spec version: {spec_version}")

    def k(*parts: str) -> str:
        return "__".join(str(p) for p in parts)

    signals_cfg: Dict[str, Dict[str, object]] = spec_data.get("signals", {}) or {}
    velocity_cfg = signals_cfg.get("velocity", {}) or {}
    residual_cfg = signals_cfg.get("residual_alpha", {}) or {}
    adaptive_cfg = signals_cfg.get("adaptive_ic", {}) or {}
    regime_cfg = signals_cfg.get("regime_blend", {}) or {}
    evaluation_cfg = spec_data.get("evaluation", {}) or {}

    def render_weekly_tab():
        as_of = st.date_input("As-of date", value=date.today(), key=k("weekly", "as_of"))

        selection_cfg = spec_data.get("universe_selection", {}) or {}
        universe_choices = get_universe_choices()
        default_index = universe_choices.index("SP500_FULL") if "SP500_FULL" in universe_choices else 0

        universe_mode = st.selectbox(
            "Universe Mode",
            ["auto", "manual"],
            index=0,
            key=k("weekly", "mode"),
        )
        manual_universe = None
        if universe_mode == "manual":
            manual_universe = st.selectbox(
                "Universe",
                universe_choices,
                index=default_index,
                key=k("weekly", "manual_universe"),
                format_func=lambda u: LABELS.get(u, u),
            )

        apply_filters = st.checkbox(
            "Apply liquidity/price filters",
            value=True,
            key=k("weekly", "apply_filters"),
        )

        cap_default = 150
        runtime_cap = st.slider(
            "Runtime cap (0 = no cap)",
            min_value=0,
            max_value=1000,
            value=cap_default,
            key=k("weekly", "runtime_cap_slider"),
        )
        bypass_cap_if_cache_warm = st.checkbox(
            "Bypass cap when cache is warm",
            value=True,
            key=k("weekly", "bypass_cap"),
        )

        turnover_cap_key = k("weekly", "turnover_cap")
        rebalance_band_key = k("weekly", "rebalance_band")
        turnover_cap = st.slider(
            "Turnover cap (weekly)",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.01,
            key=turnover_cap_key,
        )
        rebalance_band = st.slider(
            "Rebalance band (fraction of target weight)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            key=rebalance_band_key,
        )

        sector_neutral = st.checkbox(
            "Sector-neutral scoring",
            value=False,
            key=k("weekly", "sector_neutral"),
        )

        bandit_default = bool(selection_cfg.get("bandit", {}).get("enabled", False))
        bandit_enabled = st.checkbox(
            "Bandit selection (net alpha)",
            value=bandit_default,
            key=k("weekly", "bandit"),
        )

        enable_velocity = st.checkbox(
            "Enable velocity features",
            value=bool(velocity_cfg.get("enable", False)),
            key=k("weekly", "velocity"),
        )
        enable_residual = st.checkbox(
            "Use residual alpha",
            value=bool(residual_cfg.get("enable", False)),
            key=k("weekly", "residual_alpha"),
        )
        enable_adaptive_ic = st.checkbox(
            "Adaptive weighting by IC",
            value=bool(adaptive_cfg.get("enable", False)),
            key=k("weekly", "adaptive_ic"),
        )
        enable_regime_blend = st.checkbox(
            "Regime blend",
            value=bool(regime_cfg.get("enable", False)),
            key=k("weekly", "regime_blend"),
        )

        st.markdown(
            "Run the automated weekly rebalance to refresh features, signals, and telemetry."
        )

        if st.button("Refresh universe lists now", key=k("weekly", "refresh")):
            results = universe_registry.refresh_all(force=True)
            st.success("Universe registry refresh complete.")
            st.json(results)

        if not st.button("Run Weekly Cycle", type="primary", key=k("weekly", "run")):
            return

        with st.spinner("Running weekly workflow..."):
            try:
                metrics_history_path = Path("metrics_history.json")
                decision_info = None
                selected_universe_name = manual_universe or universe_choices[default_index]
                if universe_mode == "auto":
                    candidates = selection_cfg.get("candidates") or universe_choices
                    constraints = selection_cfg.get("constraints", {}) or {}
                    decision_info = universe_selector.choose_universe(
                        list(candidates),
                        constraints,
                        universe_registry.load_universe,
                        metrics_history_path,
                        spec_data,
                        str(as_of),
                        bandit_enabled=bandit_enabled,
                    )
                    selected_universe_name = decision_info.get("winner", candidates[0])

                st.subheader("Universe selection")
                st.write(f"Selected universe: **{selected_universe_name}**")
                if decision_info:
                    metrics_table = decision_info["metrics"].copy()
                    metrics_table = metrics_table.reindex(decision_info.get("candidates", metrics_table.index))
                    display_cols = [
                        "alpha",
                        "sortino",
                        "mdd",
                        "coverage",
                        "turnover_cost",
                        "n_weeks",
                    ]
                    for col in display_cols:
                        if col not in metrics_table.columns:
                            metrics_table[col] = np.nan
                    summary_df = metrics_table[display_cols].copy()
                    scores = pd.Series(decision_info.get("scores", {}))
                    probs = pd.Series(decision_info.get("probabilities", {}))
                    summary_df["score"] = scores.reindex(summary_df.index)
                    summary_df["probability"] = probs.reindex(summary_df.index)
                    st.dataframe(
                        summary_df.fillna(0.0).style.format(
                            {
                                "alpha": "{:.4f}",
                                "sortino": "{:.2f}",
                                "mdd": "{:.2%}",
                                "coverage": "{:.1%}",
                                "turnover_cost": "{:.4%}",
                                "probability": "{:.1%}",
                                "score": "{:.4f}",
                            }
                        )
                    )
                    st.caption(decision_info.get("rationale", ""))
                    bandit_info = decision_info.get("bandit", {})
                    if bandit_info:
                        st.write("Bandit status:")
                        st.json(bandit_info)

                try:
                    uni = universe.load_universe(selected_universe_name, apply_filters=apply_filters)
                except universe_registry.UniverseRegistryError as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"Failed to load {selected_universe_name}: {exc}")
                    st.stop()

                attrs = dict(getattr(uni, "attrs", {}))
                uni = ensure_universe_schema(uni, selected_universe_name)
                uni.attrs.update(attrs)

                symbols = (
                    uni["symbol"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )
                symbols, benchmark_symbol = _ensure_benchmark_symbol(
                    symbols, selected_universe_name
                )
                benchmark_label = BENCHMARK_LABELS.get(benchmark_symbol, benchmark_symbol)
                sector_lookup = None
                if "sector" in uni.columns:
                    with pd.option_context("mode.use_inf_as_na", True):
                        sector_lookup = (
                            uni.set_index("symbol")["sector"].astype(str).str.upper().replace("", pd.NA)
                        )
                if not symbols:
                    st.error(f"No symbols resolved for universe {selected_universe_name}.")
                    st.stop()

                filter_meta = uni.attrs.get("universe_filter_meta", {})
                raw_count = int(filter_meta.get("raw_count", len(uni)))
                filtered_count = int(filter_meta.get("filtered_count", len(uni)))
                st.write(
                    "Universe size — "
                    f"raw: **{raw_count}**, post-filter: **{filtered_count}**"
                )
                if filter_meta.get("reason"):
                    st.info(filter_meta.get("reason"))

                min_expected = universe_registry.expected_min_constituents(selected_universe_name)
                cache_key = _symbol_cache_key(selected_universe_name)
                warm_price_cache = dataops.has_warm_price_cache(cache_key, min_expected)

                snapshot_df = dataops.load_latest_ohlcv_snapshot()
                snapshot_coverage = dataops.ohlcv_snapshot_coverage(snapshot_df, symbols)
                cache_warm_key = k("weekly", "cache_warm")
                if snapshot_coverage >= 0.80:
                    st.session_state[cache_warm_key] = True
                    st.caption(
                        f"OHLCV snapshot warm — coverage: {snapshot_coverage:.1%}"
                    )
                cache_warm_flag = bool(st.session_state.get(cache_warm_key, False))

                effective_warm = warm_price_cache or cache_warm_flag
                capped_symbols, effective_cap = _apply_runtime_cap(
                    symbols,
                    runtime_cap,
                    effective_warm,
                    bypass_cap_if_cache_warm,
                )
                if cache_warm_flag and bypass_cap_if_cache_warm:
                    effective_cap = 0
                st.write(
                    f"Effective universe for data fetch: **{len(capped_symbols)}** "
                    f"(cap applied: {effective_cap if effective_cap else 'none'})"
                )
                if cache_warm_flag and bypass_cap_if_cache_warm:
                    st.success(
                        f"Full universe enabled (coverage: {snapshot_coverage:.1%})"
                    )
                elif warm_price_cache:
                    st.success("Warm price cache detected.")
                if not capped_symbols:
                    st.error(f"No symbols resolved for universe {selected_universe_name}.")
                    st.stop()

                years = spec_data.get("data", {}).get("price_years", 5)
                try:
                    prices = dataops.fetch_prices(capped_symbols, years=years)
                    dataops.cache_parquet(prices, cache_key)
                    dataops.write_latest_ohlcv_snapshot(prices)
                except Exception as exc:
                    st.warning(f"Price fetch failed ({exc}); generating fallback series.")
                    idx = pd.date_range(end=pd.Timestamp(as_of), periods=252 * 5, freq="B")
                    random_walk = np.cumprod(1 + np.random.randn(len(idx), len(capped_symbols)) * 0.001, axis=0)
                    prices = pd.DataFrame(random_walk, index=idx, columns=capped_symbols)
                    dataops.write_latest_ohlcv_snapshot(prices)

                if prices.empty:
                    st.error("No price data available after fetch.")
                    st.stop()

                st.write("Downloaded prices:", prices.shape)

                try:
                    feats = features.combine_features(prices)
                except Exception:
                    feats = pd.DataFrame(index=prices.columns)
                feats = feats.fillna(0.0)

                returns = prices.pct_change().dropna(how="all")
                bench_returns = returns.get(
                    benchmark_symbol, pd.Series(0.0, index=returns.index)
                ).fillna(0.0)
                try:
                    fwd5 = (1 + returns).rolling(5, min_periods=5).apply(lambda x: x.prod() - 1).shift(-5)
                    fwd5 = fwd5.iloc[:-5] if len(fwd5) >= 5 else fwd5.iloc[0:0]
                except Exception:
                    fwd5 = pd.DataFrame(index=returns.index, columns=returns.columns)

                feature_history = {}
                history_dates = returns.index[-signals.ROLLING_WEEKS :]
                for ts in history_dates:
                    price_slice = prices.loc[:ts]
                    if price_slice.empty:
                        continue
                    try:
                        feature_snapshot = features.combine_features(price_slice)
                        feature_history[ts] = feature_snapshot
                    except Exception:
                        continue

                velocity_panel = pd.DataFrame()
                if enable_velocity and feature_history:
                    history_panel = pd.concat(
                        {ts: df for ts, df in feature_history.items()},
                        names=["date", "symbol"],
                    )
                    try:
                        velocity_panel = feature_velocity.build_velocity_features(
                            history_panel,
                            velocity_cfg.get("windows", {}),
                        )
                    except Exception:
                        velocity_panel = pd.DataFrame()
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
                feats = feats.fillna(0.0)

                residual_target = None
                if enable_residual:
                    try:
                        residual_target = residual_signals.compute_residual_returns(
                            returns,
                            sector_lookup if sector_lookup is not None else None,
                            bench_returns,
                        )
                    except Exception:
                        residual_target = None

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

                feature_ic_snapshot = pd.Series(dtype=float)
                if not fwd5.empty:
                    try:
                        latest_target = fwd5.iloc[-1]
                        feature_ic_snapshot = metrics.feature_ic_snapshot(feats, latest_target)
                    except Exception:
                        feature_ic_snapshot = pd.Series(dtype=float)

                ic_ema_series = None
                if enable_adaptive_ic and not feature_ic_snapshot.empty:
                    existing_ic = signals.load_feature_ic_ema()
                    ic_ema_series = signals.update_feature_ic_ema(
                        existing_ic,
                        feature_ic_snapshot,
                        ema_lambda=float(adaptive_cfg.get("ema_lambda", 0.9)),
                    )
                    signals.save_feature_ic_ema(ic_ema_series)
                    w_ridge = signals.apply_ic_weighting(
                        w_ridge,
                        ic_ema_series,
                        alpha_ic=float(adaptive_cfg.get("alpha_ic", 0.2)),
                        clip=float(adaptive_cfg.get("clip", 0.5)),
                    )
                else:
                    ic_ema_series = signals.load_feature_ic_ema()

                sector_series = None
                if sector_lookup is not None:
                    try:
                        sector_series = sector_lookup.reindex(feats.index)
                    except Exception:
                        sector_series = None

                scores = signals.score_current(
                    feats,
                    w_ridge,
                    sector_map=sector_series,
                    sector_neutral=sector_neutral,
                )
                if scores.empty:
                    scores = pd.Series(np.random.randn(len(feats.index)), index=feats.index)

                ic_value = float("nan")
                hit_value = float("nan")
                if not fwd5.empty:
                    latest_target = fwd5.iloc[-1]
                    ic_value = metrics.spearman_ic(scores, latest_target)
                    hit_value = metrics.hit_rate(scores, latest_target)
                    if evaluation_cfg.get("track_ic", True) or evaluation_cfg.get("track_hit_rate", True):
                        payload = {
                            "date": str(as_of),
                            "ic": float(ic_value) if pd.notna(ic_value) else float("nan"),
                            "hit_rate": float(hit_value) if pd.notna(hit_value) else float("nan"),
                            "universe": selected_universe_name,
                            "benchmark": benchmark_symbol,
                        }
                        metrics.append_ic_metric(payload)

                attribution = explain.coef_attribution(w_ridge, top_n=10)

                top_holdings = scores.head(15).index.tolist()
                shap_table = explain.shap_like_contributions(
                    feats.reindex(index=top_holdings),
                    w_ridge,
                    top_features=5,
                )
                st.session_state[k("explain", "payload")] = {
                    "as_of": str(as_of),
                    "universe": selected_universe_name,
                    "scores": scores,
                    "attribution": attribution,
                    "shap": shap_table,
                    "ic_ema": ic_ema_series,
                    "ic_value": ic_value,
                    "hit_rate": hit_value,
                }

                returns_252 = returns.tail(252)
                try:
                    w0 = portfolio.inverse_vol_weights(
                        returns_252,
                        top_holdings,
                        cap_single=0.10,
                        k=min(15, len(top_holdings)),
                    )
                except Exception:
                    w0 = pd.Series(1 / max(1, len(top_holdings)), index=top_holdings)

                try:
                    w_sector = (
                        portfolio.apply_sector_caps(w0, sector_lookup, cap=0.35)
                        if sector_lookup is not None
                        else w0
                    )
                except Exception:
                    w_sector = w0

                blend_meta = {}
                if enable_regime_blend:
                    try:
                        trend_series = (
                            returns_252.reindex(columns=w_sector.index, fill_value=0.0)
                            .mul(w_sector, axis=1)
                            .sum(axis=1)
                        )
                        trend_perf = metrics.sharpe(trend_series, periods_per_year=252)
                    except Exception:
                        trend_series = pd.Series(dtype=float)
                        trend_perf = float("nan")

                    try:
                        mr_candidates = scores.sort_values(ascending=True).head(len(w_sector))
                        mr_weights = portfolio.inverse_vol_weights(
                            returns_252,
                            mr_candidates.index.tolist(),
                            cap_single=0.10,
                            k=min(15, len(mr_candidates)),
                        )
                        mr_series = (
                            returns_252.reindex(columns=mr_weights.index, fill_value=0.0)
                            .mul(mr_weights, axis=1)
                            .sum(axis=1)
                        )
                        mr_perf = metrics.sharpe(mr_series, periods_per_year=252)
                    except Exception:
                        mr_weights = w_sector
                        mr_series = pd.Series(dtype=float)
                        mr_perf = float("nan")

                    blend_meta = {"trend": float(trend_perf or 0.0), "mean_reversion": float(mr_perf or 0.0)}
                    try:
                        w_sector = regime.blend_weights(w_sector, mr_weights, blend_meta)
                    except Exception:
                        pass

                last = memory.load_last_portfolio()
                last_w = None
                if last:
                    last_w = pd.Series({h["ticker"]: h["weight"] for h in last.get("holdings", [])})

                try:
                    w_final = portfolio.apply_turnover_controls(
                        last_w,
                        w_sector,
                        turnover_cap=float(turnover_cap),
                        rebalance_band=float(rebalance_band),
                    )
                except Exception:
                    w_final = w_sector

                w_final = portfolio.apply_single_name_cap(w_final, cap=0.10)
                w_final = w_final.sort_values(ascending=False)
                turnover_fraction = float(portfolio.turnover(last_w, w_final)) if last_w is not None else float(w_final.abs().sum())

                port = {
                    "as_of": str(as_of),
                    "holdings": [
                        {"ticker": ticker, "weight": float(weight)} for ticker, weight in w_final.items()
                    ],
                    "cash_weight": float(max(0.0, 1.0 - w_final.sum())),
                }
                memory.save_portfolio(port)
                st.success("Weekly portfolio created.")
                st.json(port)

                port_rets = (
                    returns_252.reindex(columns=w_final.index, fill_value=0.0).mul(w_final, axis=1).sum(axis=1)
                )
                curve = (1 + port_rets).cumprod()
                bench = returns_252.get(
                    benchmark_symbol, pd.Series(0.0, index=returns_252.index)
                )

                sor = metrics.sortino(port_rets) if len(port_rets) else 0.0
                mdd = metrics.max_drawdown(curve) if len(curve) else 0.0
                alpha = metrics.alpha_vs_bench(port_rets, bench) if not bench.empty else 0.0
                net_alpha = alpha - 0.0005 * turnover_fraction
                cost_bps_weekly = float(0.0005 * turnover_fraction * 10000.0)

                beta_series = risk.estimate_asset_betas(
                    returns_252, benchmark_col=benchmark_symbol
                )
                portfolio_beta = float((w_final * beta_series.reindex(w_final.index).fillna(0.0)).sum())
                vol_realised = risk.annualised_volatility(port_rets)

                st.subheader("Weekly metrics")
                st.write(
                    f"- Sortino: **{sor:.2f}**  \n"
                    f"- Max Drawdown: **{mdd:.2%}**  \n"
                    f"- Alpha vs {benchmark_label} (weekly mean): **{alpha:.4%}**"
                )

                st.subheader("Risk dashboard")
                beta_col, vol_col, dd_col, scaler_col = st.columns(4)
                beta_col.metric(f"Beta vs {benchmark_label}", f"{portfolio_beta:.2f}")
                vol_col.metric("Realised vol", f"{vol_realised:.2%}" if pd.notna(vol_realised) else "n/a")
                dd_col.metric("Max drawdown", f"{mdd:.2%}")
                scaler_col.metric("Overlay scaler", "1.00")
                st.caption(
                    f"Turnover: {turnover_fraction:.2%} — Cost drag: {cost_bps_weekly:.2f} bps/week"
                )

                holdings_csv = StringIO()
                pd.DataFrame(port["holdings"]).to_csv(holdings_csv, index=False)
                equity_csv = StringIO()
                curve.to_frame("equity").to_csv(equity_csv)
                summary_payload = {
                    "spec": spec_version,
                    "as_of": str(as_of),
                    "alpha": float(alpha),
                    "net_alpha": float(net_alpha),
                    "sortino": float(sor),
                    "max_drawdown": float(mdd),
                    "turnover": float(turnover_fraction),
                    "cost_bps_weekly": float(cost_bps_weekly),
                    "portfolio_beta": float(portfolio_beta),
                    "vol_realized": float(vol_realised) if pd.notna(vol_realised) else float("nan"),
                    "benchmark": benchmark_symbol,
                    "sector_neutral": bool(sector_neutral),
                    "bandit_mode": bool(bandit_enabled and bool(decision_info)),
                    "overlay_scaler": 1.0,
                    "regime_blend": blend_meta,
                    "ic": float(ic_value) if pd.notna(ic_value) else float("nan"),
                    "hit_rate": float(hit_value) if pd.notna(hit_value) else float("nan"),
                }
                summary_json = json.dumps(summary_payload, indent=2)

                st.download_button(
                    "Download holdings.csv",
                    data=holdings_csv.getvalue(),
                    file_name="holdings.csv",
                    mime="text/csv",
                    key=k("weekly", "dl_holdings"),
                )
                st.download_button(
                    "Download equity_curve.csv",
                    data=equity_csv.getvalue(),
                    file_name="equity_curve.csv",
                    mime="text/csv",
                    key=k("weekly", "dl_curve"),
                )
                st.download_button(
                    "Download summary.json",
                    data=summary_json,
                    file_name="summary.json",
                    mime="application/json",
                    key=k("weekly", "dl_summary"),
                )

                note = (
                    f"# Weekly AI Portfolio — {as_of}\n\n"
                    f"- Sortino: {sor:.2f}\n"
                    f"- Max Drawdown: {mdd:.2%}\n"
                    f"- Alpha (vs {benchmark_label}, weekly mean): {alpha:.4%}\n"
                )
                out_path = report.write_markdown(note)
                with open(out_path, "rb") as handle:
                    st.download_button(
                        "Download weekly report",
                        data=handle.read(),
                        file_name=Path(out_path).name,
                        key=k("weekly", "dl_report"),
                    )

                val_metrics = metrics.val_metrics(port_rets, bench)
                coverage_universe = len([sym for sym in symbols if sym != benchmark_symbol])
                coverage_ratio = float(coverage_universe / max(1, min_expected))
                metrics_record = {
                    "spec": spec_version,
                    "date": str(as_of),
                    "alpha": float(alpha),
                    "net_alpha": float(net_alpha),
                    "sortino": float(sor),
                    "mdd": float(mdd),
                    "hit_rate": float((port_rets > bench).mean()),
                    "ic": float(ic_value) if pd.notna(ic_value) else float("nan"),
                    "ic_hit_rate": float(hit_value) if pd.notna(hit_value) else float("nan"),
                    "val_sortino": float(val_metrics.get("val_sortino", float("nan"))),
                    "val_alpha": float(val_metrics.get("val_alpha", float("nan"))),
                    "universe": selected_universe_name,
                    "coverage": coverage_ratio,
                    "turnover_cost": float(0.0005 * turnover_fraction if turnover_fraction else 0.0),
                    "total_cost": float(0.0005 * turnover_fraction),
                    "cost_bps_weekly": float(cost_bps_weekly),
                    "overlay_scaler": 1.0,
                    "sector_neutral": bool(sector_neutral),
                    "bandit_mode": bool(bandit_enabled and bool(decision_info)),
                    "benchmark": benchmark_symbol,
                    "regime_blend": blend_meta,
                }
                memory.append_metrics(metrics_record)
                st.success("Metrics logged to metrics_history.json")

            except Exception as exc:
                st.error(f"Run failed: {exc}")
                st.stop()

    def render_backtest_tab():
        bt_mode = st.selectbox(
            "Universe Mode",
            ["auto", "manual"],
            index=0,
            key=k("bt", "mode"),
        )

        choices = sorted(registry_list())
        preferred = [
            "SP500_FULL",
            "R1000",
            "NASDAQ_100",
            "FTSE_350",
            "SP500_MINI",
        ]
        universe_choices = [u for u in preferred if u in choices]
        universe_choices.extend(u for u in choices if u not in universe_choices)

        default_bt_index = (
            universe_choices.index("SP500_FULL") if "SP500_FULL" in universe_choices else 0
        )
        selected_bt_universe = universe_choices[default_bt_index]
        if bt_mode == "manual":
            selected_bt_universe = st.selectbox(
                "Backtest Universe",
                universe_choices,
                index=default_bt_index,
                key=k("bt", "universe"),
            )

        years = st.slider(
            "History (years)",
            min_value=3,
            max_value=10,
            value=5,
            key=k("bt", "years"),
        )
        costs_on = st.checkbox(
            "Include transaction costs",
            value=True,
            key=k("bt", "tc"),
        )
        vol_on = st.checkbox(
            "Target volatility (10% ann.)",
            value=True,
            key=k("bt", "vol"),
        )
        sector_neutral = st.checkbox(
            "Sector-neutral scoring",
            value=True,
            key=k("bt", "sector_neutral"),
        )
        use_velocity = st.checkbox(
            "Velocity features",
            value=True,
            key=k("bt", "vel"),
        )
        use_residual = st.checkbox(
            "Residual alpha",
            value=True,
            key=k("bt", "resid"),
        )
        use_adapt_ic = st.checkbox(
            "Adaptive weighting by IC",
            value=True,
            key=k("bt", "ic"),
        )
        use_regime = st.checkbox(
            "Regime blend",
            value=True,
            key=k("bt", "regime"),
        )

        if "abort_bt" not in st.session_state:
            st.session_state["abort_bt"] = False

        turnover_cap = 0.40
        rebalance_band = 0.25

        if st.button("Run Walk-Forward Backtest", key=k("bt", "run")):
            with st.spinner("Running walk-forward backtest..."):
                from src.backtester import BacktestConfig, WalkForwardBacktester

                st.session_state["abort_bt"] = False
                abort_placeholder = st.empty()
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                progress_state: Dict[str, object] = {"bar": None}

                def _abort_backtest() -> None:
                    st.session_state["abort_bt"] = True

                abort_placeholder.button(
                    "Abort run",
                    key=k("bt", "abort_button"),
                    type="secondary",
                    on_click=_abort_backtest,
                )

                def _should_abort() -> bool:
                    return bool(st.session_state.get("abort_bt", False))

                def _progress(step: int, total: int) -> None:
                    if total <= 0:
                        return
                    ratio = min(1.0, (step + 1) / float(total))
                    if progress_state["bar"] is None:
                        progress_state["bar"] = progress_placeholder.progress(ratio)
                    else:
                        bar = progress_state["bar"]
                        try:
                            bar.progress(ratio)
                        except Exception:
                            progress_state["bar"] = progress_placeholder.progress(ratio)
                    status_placeholder.info(f"Window {step + 1}/{total}")

                def _cleanup() -> None:
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    abort_placeholder.empty()
                    st.session_state["abort_bt"] = False

                try:
                    universe_df = universe.load_universe(
                        selected_bt_universe, apply_filters=True
                    )
                except universe_registry.UniverseRegistryError as exc:
                    st.error(str(exc))
                    _cleanup()
                    return
                except Exception as exc:
                    st.error(f"Failed to load {selected_bt_universe}: {exc}")
                    _cleanup()
                    return

                attrs = dict(getattr(universe_df, "attrs", {}))
                universe_df = ensure_universe_schema(universe_df, selected_bt_universe)
                universe_df.attrs.update(attrs)

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
                symbols, benchmark_symbol = _ensure_benchmark_symbol(
                    symbols, selected_bt_universe
                )

                sector_lookup = None
                if "sector" in universe_df.columns:
                    sector_lookup = (
                        universe_df.set_index("symbol")["sector"]
                        .astype(str)
                        .str.upper()
                        .replace("", pd.NA)
                    )

                adv_series = None
                if "adv" in universe_df.columns:
                    try:
                        adv_series = (
                            universe_df.set_index("symbol")["adv"].astype(float).replace({pd.NA: 0.0})
                        )
                    except Exception:
                        adv_series = None

                lookback_days = int(years * 252)

                try:
                    prices = dataops.fetch_prices(symbols, years=years)
                except Exception as exc:
                    st.error(f"Price fetch failed: {exc}")
                    _cleanup()
                    return

                if prices.empty:
                    st.error("No price data available for backtest.")
                    _cleanup()
                    return

                prev_weights = None
                ic_state = None

                def strategy_fn(price_window: pd.DataFrame) -> pd.Series:
                    nonlocal prev_weights, ic_state

                    price_window = price_window.ffill()
                    returns = price_window.pct_change().dropna(how="all")
                    if returns.empty:
                        return pd.Series(dtype=float)

                    try:
                        feats = features.combine_features(price_window)
                    except Exception:
                        feats = pd.DataFrame(index=price_window.columns)
                    feats = feats.fillna(0.0)

                    feature_history = {}
                    history_dates = returns.index[-signals.ROLLING_WEEKS :]
                    for ts in history_dates:
                        window_slice = price_window.loc[:ts]
                        if window_slice.empty:
                            continue
                        try:
                            feature_history[ts] = features.combine_features(window_slice)
                        except Exception:
                            continue

                    if use_velocity and feature_history:
                        try:
                            history_panel = pd.concat(
                                {ts: df for ts, df in feature_history.items()},
                                names=["date", "symbol"],
                            )
                            velocity_panel = feature_velocity.build_velocity_features(
                                history_panel,
                                velocity_cfg.get("windows", {}),
                            )
                        except Exception:
                            velocity_panel = pd.DataFrame()
                        if not velocity_panel.empty:
                            for ts in list(feature_history.keys()):
                                try:
                                    feature_history[ts] = feature_history[ts].join(
                                        velocity_panel.xs(ts, level=0), how="left"
                                    )
                                except Exception:
                                    continue
                            try:
                                latest_ts = sorted(feature_history.keys())[-1]
                                feats = feats.join(
                                    velocity_panel.xs(latest_ts, level=0), how="left"
                                )
                            except Exception:
                                pass
                    feats = feats.fillna(0.0)

                    bench_returns = returns.get(
                        benchmark_symbol, pd.Series(0.0, index=returns.index)
                    ).fillna(0.0)

                    try:
                        fwd5 = (
                            (1 + returns)
                            .rolling(5, min_periods=5)
                            .apply(lambda x: x.prod() - 1)
                            .shift(-5)
                        )
                        fwd5 = fwd5.iloc[:-5] if len(fwd5) >= 5 else fwd5.iloc[0:0]
                    except Exception:
                        fwd5 = pd.DataFrame(index=returns.index, columns=returns.columns)

                    residual_target = None
                    if use_residual:
                        try:
                            residual_target = residual_signals.compute_residual_returns(
                                returns,
                                sector_lookup if sector_lookup is not None else None,
                                bench_returns,
                            )
                        except Exception:
                            residual_target = None

                    try:
                        w_ridge = signals.fit_rolling_ridge(fwd5, feature_history)
                        if w_ridge.empty:
                            target = (
                                fwd5.iloc[-1].fillna(0.0)
                                if not fwd5.empty
                                else pd.Series(0.0, index=feats.index)
                            )
                            w_ridge = signals.fit_ridge(feats, target)
                    except Exception:
                        w_ridge = pd.Series(dtype=float)

                    if use_residual and residual_target is not None and not residual_target.empty:
                        aligned_target = residual_target.reindex(feats.index).fillna(0.0)
                        if aligned_target.abs().sum() > 0:
                            try:
                                w_ridge = signals.fit_ridge(feats, aligned_target)
                            except Exception:
                                pass

                    feature_ic_snapshot = pd.Series(dtype=float)
                    if not fwd5.empty:
                        try:
                            latest_target = fwd5.iloc[-1]
                            feature_ic_snapshot = metrics.feature_ic_snapshot(
                                feats, latest_target
                            )
                        except Exception:
                            feature_ic_snapshot = pd.Series(dtype=float)

                    if use_adapt_ic and not feature_ic_snapshot.empty:
                        if ic_state is None:
                            ic_state = signals.load_feature_ic_ema()
                        ic_state = signals.update_feature_ic_ema(
                            ic_state,
                            feature_ic_snapshot,
                            ema_lambda=float(adaptive_cfg.get("ema_lambda", 0.9)),
                        )
                        signals.save_feature_ic_ema(ic_state)
                        w_ridge = signals.apply_ic_weighting(
                            w_ridge,
                            ic_state,
                            alpha_ic=float(adaptive_cfg.get("alpha_ic", 0.2)),
                            clip=float(adaptive_cfg.get("clip", 0.5)),
                        )

                    sector_series = None
                    if sector_lookup is not None:
                        try:
                            sector_series = sector_lookup.reindex(feats.index)
                        except Exception:
                            sector_series = None

                    scores = signals.score_current(
                        feats,
                        w_ridge,
                        sector_map=sector_series,
                        sector_neutral=sector_neutral,
                    )
                    if scores.empty:
                        scores = pd.Series(np.random.randn(len(feats.index)), index=feats.index)

                    top = scores.head(15)
                    returns_252 = returns.tail(252)
                    try:
                        w0 = portfolio.inverse_vol_weights(
                            returns_252,
                            top.index.tolist(),
                            cap_single=0.10,
                            k=min(15, len(top)),
                        )
                    except Exception:
                        w0 = pd.Series(1 / max(1, len(top)), index=top.index)

                    try:
                        w_sector = (
                            portfolio.apply_sector_caps(w0, sector_lookup, cap=0.35)
                            if sector_lookup is not None
                            else w0
                        )
                    except Exception:
                        w_sector = w0

                    if use_regime:
                        try:
                            trend_series = (
                                returns_252.reindex(columns=w_sector.index, fill_value=0.0)
                                .mul(w_sector, axis=1)
                                .sum(axis=1)
                            )
                            trend_perf = metrics.sharpe(
                                trend_series, periods_per_year=252
                            )
                        except Exception:
                            trend_perf = float("nan")

                        try:
                            mr_candidates = scores.sort_values(ascending=True).head(len(w_sector))
                            mr_weights = portfolio.inverse_vol_weights(
                                returns_252,
                                mr_candidates.index.tolist(),
                                cap_single=0.10,
                                k=min(15, len(mr_candidates)),
                            )
                            mr_series = (
                                returns_252.reindex(columns=mr_weights.index, fill_value=0.0)
                                .mul(mr_weights, axis=1)
                                .sum(axis=1)
                            )
                            mr_perf = metrics.sharpe(
                                mr_series, periods_per_year=252
                            )
                            blend_meta = {
                                "trend": float(trend_perf or 0.0),
                                "mean_reversion": float(mr_perf or 0.0),
                            }
                            w_sector = regime.blend_weights(w_sector, mr_weights, blend_meta)
                        except Exception:
                            pass

                    if prev_weights is not None:
                        try:
                            w_final = portfolio.apply_turnover_controls(
                                prev_weights,
                                w_sector,
                                turnover_cap=float(turnover_cap),
                                rebalance_band=float(rebalance_band),
                            )
                        except Exception:
                            w_final = w_sector
                    else:
                        w_final = w_sector

                    w_final = portfolio.apply_single_name_cap(w_final, cap=0.10)
                    if w_final.sum() <= 0:
                        return w_final
                    w_final = (w_final / w_final.sum()).sort_values(ascending=False)
                    prev_weights = w_final
                    return w_final

                config = BacktestConfig(
                    universe=selected_bt_universe,
                    spec_version=spec_version,
                    lookback_days=lookback_days,
                    target_vol=0.10 if vol_on else None,
                    beta_limit=1.2,
                    drawdown_limit=0.20,
                    base_bps=5.0 if costs_on else 0.0,
                    benchmark=benchmark_symbol,
                )

                engine = WalkForwardBacktester(
                    prices,
                    strategy_fn,
                    config,
                    adv=adv_series,
                )

                try:
                    result = engine.run(
                        log=True,
                        should_abort=_should_abort,
                        progress_callback=_progress,
                    )
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    _cleanup()
                    return

                _cleanup()

                summary = result.get("summary", {})
                if result.get("aborted"):
                    st.warning("Backtest aborted by user.")
                else:
                    st.success("Backtest completed.")
                st.json(summary)

                equity_curve = result.get("equity_curve")
                if equity_curve is not None and not equity_curve.empty:
                    st.line_chart(equity_curve)

                gross = result.get("gross_returns")
                net = result.get("net_returns")
                if gross is not None and not gross.empty and net is not None and not net.empty:
                    perf_df = pd.DataFrame({
                        "gross": gross,
                        "net": net,
                    })
                    st.area_chart(perf_df)

    def render_explain_tab():
        payload = st.session_state.get(k("explain", "payload"))
        if not payload:
            st.info("Run the Weekly Cycle to populate explainability panels.")
            return

        as_of = payload.get("as_of", "n/a")
        st.subheader(f"Top candidates — {as_of}")
        scores = payload.get("scores")
        if isinstance(scores, pd.Series) and not scores.empty:
            st.dataframe(scores.head(20).to_frame("score"))
        else:
            st.info("No candidate scores available.")

        attribution = payload.get("attribution")
        st.subheader("Top Feature Weights (latest)")
        if isinstance(attribution, pd.DataFrame) and not attribution.empty:
            st.bar_chart(attribution["importance"])
            st.dataframe(attribution)
        else:
            st.info("No feature weights available.")

        shap_table = payload.get("shap")
        st.subheader("Feature contributions (top holdings)")
        if isinstance(shap_table, pd.DataFrame) and not shap_table.empty:
            st.dataframe(shap_table)
        else:
            st.info("No contributions to display.")

        st.subheader("Rolling IC & hit-rate (26 weeks)")
        ic_history_df = metrics.load_ic_history()
        if ic_history_df.empty:
            st.info("No IC history recorded yet.")
        else:
            recent_hist = ic_history_df.tail(26).set_index("date")[["ic", "hit_rate"]]
            st.line_chart(recent_hist)

        st.subheader("Feature IC_EMA (top/bottom 5)")
        ic_ema_series = payload.get("ic_ema")
        if ic_ema_series is None or getattr(ic_ema_series, "empty", True):
            st.info("No IC_EMA values available.")
        else:
            ic_table = explain.ic_ema_table(ic_ema_series, top_n=5)
            st.dataframe(ic_table)

        ic_value = payload.get("ic_value")
        hit_rate = payload.get("hit_rate")
        if ic_value is not None or hit_rate is not None:
            st.caption(
                f"Latest IC: {ic_value if ic_value is not None else float('nan'):.4f} — "
                f"Hit-rate: {hit_rate if hit_rate is not None else float('nan'):.2%}"
            )

    tabs = st.tabs(["Weekly", "Backtest", "Explain"])
    with tabs[0]:
        render_weekly_tab()
    with tabs[1]:
        render_backtest_tab()
    with tabs[2]:
        render_explain_tab()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
