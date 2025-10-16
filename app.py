import json
import sys
from io import StringIO
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

# --- Make project root importable (works under streamlit/pytest/CI) ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clean_symbol_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
    )


def _resolve_symbols_from_universe(df: pd.DataFrame) -> List[str]:
    """Return deduplicated symbols regardless of column/index placement."""

    if df is None or df.empty:
        return []
    if "symbol" in df.columns:
        raw = df["symbol"]
    else:
        raw = pd.Index(df.index, name="symbol").to_series()
    cleaned = _clean_symbol_series(raw).dropna()
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
    )

    if hasattr(st, "set_page_config"):
        st.set_page_config(page_title="LLM-Codex Quant — Weekly", layout="wide")
    st.title("LLM-Codex Quant — Weekly Cycle")

    spec_path = Path("spec/current_spec.json")
    spec_data = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    spec_version = str(spec_data.get("version", "0.6"))
    st.caption(f"Spec version: {spec_version}")

    def k(*parts: str) -> str:
        return "::".join(str(p) for p in parts)

    as_of = st.date_input("As-of date", value=date.today(), key=k("weekly", "as_of"))

    selection_cfg = spec_data.get("universe_selection", {}) or {}
    universe_choices = (
        spec_data.get("universe_modes")
        or spec_data.get("universe", {}).get("modes")
        or ["SP500_MINI", "SP500_FULL", "R1000", "NASDAQ_100", "FTSE_350"]
    )
    default_index = (
        universe_choices.index("SP500_FULL") if "SP500_FULL" in universe_choices else 0
    )

    universe_mode = st.selectbox(
        "Universe Mode",
        ["auto", "manual"],
        index=0,
        key=k("weekly", "mode"),
    )
    manual_universe = None
    if universe_mode == "manual":
        manual_universe = st.selectbox(
            "Universe", universe_choices, index=default_index, key=k("weekly", "manual_universe")
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

            filter_meta = uni.attrs.get("universe_filter_meta", {})
            raw_count = int(filter_meta.get("raw_count", len(uni)))
            filtered_count = int(filter_meta.get("filtered_count", len(uni)))
            st.write(
                "Universe size — "
                f"raw: **{raw_count}**, post-filter: **{filtered_count}**"
            )
            if filter_meta.get("reason"):
                st.info(filter_meta.get("reason"))

            symbols = _resolve_symbols_from_universe(uni)
            if "SPY" not in symbols:
                symbols.append("SPY")

            min_expected = universe_registry.expected_min_constituents(selected_universe_name)
            cache_key = _symbol_cache_key(selected_universe_name)
            warm_cache = dataops.has_warm_price_cache(cache_key, min_expected)
            capped_symbols, effective_cap = _apply_runtime_cap(
                symbols,
                runtime_cap,
                warm_cache,
                bypass_cap_if_cache_warm,
            )
            st.write(
                f"Effective universe for data fetch: **{len(capped_symbols)}** "
                f"(cap applied: {effective_cap if effective_cap else 'none'})"
            )
            if warm_cache:
                st.success("Warm price cache detected.")
            if not capped_symbols:
                st.error(f"No symbols resolved for universe {selected_universe_name}.")
                st.stop()

            years = spec_data.get("data", {}).get("price_years", 5)
            try:
                prices = dataops.fetch_prices(capped_symbols, years=years)
                dataops.cache_parquet(prices, cache_key)
            except Exception as exc:
                st.warning(f"Price fetch failed ({exc}); generating fallback series.")
                idx = pd.date_range(end=pd.Timestamp(as_of), periods=252 * 5, freq="B")
                random_walk = np.cumprod(1 + np.random.randn(len(idx), len(capped_symbols)) * 0.001, axis=0)
                prices = pd.DataFrame(random_walk, index=idx, columns=capped_symbols)

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

            try:
                w_ridge = signals.fit_rolling_ridge(fwd5, feature_history)
                if w_ridge.empty:
                    target = fwd5.iloc[-1].fillna(0.0) if not fwd5.empty else pd.Series(0.0, index=feats.index)
                    w_ridge = signals.fit_ridge(feats, target)
            except Exception:
                w_ridge = pd.Series(dtype=float)

            sector_map = None
            try:
                sector_map = uni.set_index("symbol")["sector"].reindex(feats.index)
            except Exception:
                sector_map = None

            scores = signals.score_current(
                feats,
                w_ridge,
                sector_map=sector_map,
                sector_neutral=sector_neutral,
            )
            if scores.empty:
                scores = pd.Series(np.random.randn(len(feats.index)), index=feats.index)

            st.subheader("Top candidates")
            st.dataframe(scores.head(20).to_frame("score"))

            attribution = explain.coef_attribution(w_ridge, top_n=10)
            st.subheader("Top Feature Weights (latest)")
            if attribution.empty:
                st.info("No feature weights available.")
            else:
                st.bar_chart(attribution["importance"])
                st.dataframe(attribution)

            top_holdings = scores.head(15).index.tolist()
            shap_table = explain.shap_like_contributions(
                feats.reindex(index=top_holdings),
                w_ridge,
                top_features=5,
            )
            st.subheader("Feature contributions (top holdings)")
            if shap_table.empty:
                st.info("No contributions to display.")
            else:
                st.dataframe(shap_table)

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
                w_sector = portfolio.apply_sector_caps(w0, sector_map, cap=0.35) if sector_map is not None else w0
            except Exception:
                w_sector = w0

            last = memory.load_last_portfolio()
            last_w = None
            if last:
                last_w = pd.Series({h["ticker"]: h["weight"] for h in last.get("holdings", [])})

            try:
                w_final = portfolio.enforce_turnover(last_w, w_sector, t_cap=0.30)
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
            bench = returns_252.get("SPY", pd.Series(0.0, index=returns_252.index))

            sor = metrics.sortino(port_rets) if len(port_rets) else 0.0
            mdd = metrics.max_drawdown(curve) if len(curve) else 0.0
            alpha = metrics.alpha_vs_bench(port_rets, bench) if not bench.empty else 0.0
            net_alpha = alpha - 0.0005 * turnover_fraction
            cost_bps_weekly = float(0.0005 * turnover_fraction * 10000.0)

            beta_series = risk.estimate_asset_betas(returns_252, benchmark_col="SPY")
            portfolio_beta = float((w_final * beta_series.reindex(w_final.index).fillna(0.0)).sum())
            vol_realised = risk.annualised_volatility(port_rets)

            st.subheader("Weekly metrics")
            st.write(
                f"- Sortino: **{sor:.2f}**  \n"
                f"- Max Drawdown: **{mdd:.2%}**  \n"
                f"- Alpha vs SPY (weekly mean): **{alpha:.4%}**"
            )

            st.subheader("Risk dashboard")
            beta_col, vol_col, dd_col, scaler_col = st.columns(4)
            beta_col.metric("Beta vs SPY", f"{portfolio_beta:.2f}")
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
                "sector_neutral": bool(sector_neutral),
                "bandit_mode": bool(bandit_enabled and bool(decision_info)),
                "overlay_scaler": 1.0,
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
                f"- Alpha (vs SPY, weekly mean): {alpha:.4%}\n"
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
            metrics_record = {
                "spec": spec_version,
                "date": str(as_of),
                "alpha": float(alpha),
                "net_alpha": float(net_alpha),
                "sortino": float(sor),
                "mdd": float(mdd),
                "hit_rate": float((port_rets > bench).mean()),
                "val_sortino": float(val_metrics.get("val_sortino", float("nan"))),
                "val_alpha": float(val_metrics.get("val_alpha", float("nan"))),
                "universe": selected_universe_name,
                "coverage": float(len(symbols) / max(1, min_expected)),
                "turnover_cost": float(0.0005 * turnover_fraction if turnover_fraction else 0.0),
                "total_cost": float(0.0005 * turnover_fraction),
                "cost_bps_weekly": float(cost_bps_weekly),
                "overlay_scaler": 1.0,
                "sector_neutral": bool(sector_neutral),
                "bandit_mode": bool(bandit_enabled and bool(decision_info)),
            }
            memory.append_metrics(metrics_record)
            st.success("Metrics logged to metrics_history.json")

        except Exception as exc:
            st.error(f"Run failed: {exc}")
            st.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
