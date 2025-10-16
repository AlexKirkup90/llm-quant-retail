from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_spec() -> dict:
    spec_path = Path("spec/current_spec.json")
    if not spec_path.exists():
        return {}
    try:
        return json.loads(spec_path.read_text())
    except Exception:
        return {}


def main():
    import streamlit as st
    import numpy as np
    import pandas as pd
    from datetime import date

    from src import (
        backtester,
        dataops,
        features,
        memory,
        metrics,
        portfolio,
        report,
        signals,
        universe,
        universe_registry,
        universe_selector,
    )
    from src.backtester import BacktestConfig, WalkForwardBacktester

    st.title("LLM-Codex Quant — Portfolio Lab")

    spec_data = _load_spec()
    universe_choices = ["SP500_MINI", "SP500_FULL", "R1000", "NASDAQ_100", "FTSE_350"]

    cycle_tab, backtest_tab = st.tabs(["Weekly Cycle", "Backtest"])

    with cycle_tab:
        as_of = st.date_input("As-of date", value=date.today())

        universe_mode = st.selectbox("Universe Mode", ["auto", "manual"], index=0)
        manual_universe = None
        if universe_mode == "manual":
            manual_universe = st.selectbox("Universe", universe_choices, index=0)

        apply_filters = st.checkbox("Apply liquidity/price filters", value=True)

        if st.button("Refresh universe lists now"):
            results = universe_registry.refresh_all(force=True)
            st.success("Universe registry refresh complete.")
            st.json(results)

            universes = getattr(universe_registry, "_UNIVERSES", {})
            row_counts = {}
            symbol_samples = {}
            for name, status in results.items():
                if isinstance(status, str) and status.lower().startswith("error:"):
                    st.markdown(f"**{name}**")
                    st.error(status)
                    continue
                definition = universes.get(name)
                if not definition:
                    continue
                try:
                    df = universe_registry.load_universe(name)
                    row_counts[name] = len(df)
                    symbol_samples[name] = df["symbol"].head(5).tolist()
                except Exception as exc:
                    st.markdown(f"**{name}**")
                    st.error(f"error: {exc}")

            if row_counts:
                st.write("Latest universe row counts:")
                st.json(row_counts)
            if symbol_samples:
                st.write("Sample symbols (first 5):")
                st.json(symbol_samples)

        st.markdown(
            "Run the automated weekly cycle to assemble features, optimise weights, "
            "and review the resulting portfolio snapshot."
        )

        if st.button("Run Weekly Cycle"):
            try:
                selection_cfg = spec_data.get("universe_selection", {})
                decision_info = None
                if universe_mode == "auto":
                    candidates = selection_cfg.get("candidates") or spec_data.get("universe", {}).get("modes", [])
                    if not candidates:
                        candidates = universe_choices
                    constraints = selection_cfg.get("constraints", {}) or {}
                    decision_info = universe_selector.choose_universe(
                        list(candidates),
                        constraints,
                        universe_registry.load_universe,
                        Path("metrics_history.json"),
                        spec_data,
                        as_of,
                    )
                    selected_universe_name = decision_info.get("winner") or candidates[0]
                else:
                    selected_universe_name = manual_universe or universe_choices[0]

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
                st.write(f"Universe size (raw): {raw_count}")
                st.write(f"Universe size (post-filter): {filtered_count}")

                if decision_info:
                    metrics_table = decision_info["metrics"].copy()
                    metrics_table = metrics_table.reindex(decision_info.get("candidates", metrics_table.index))
                    display_cols = ["alpha", "sortino", "mdd", "coverage", "turnover_cost", "n_weeks"]
                    for col in display_cols:
                        if col not in metrics_table.columns:
                            metrics_table[col] = np.nan
                    summary_df = metrics_table[display_cols].copy()
                    score_series = pd.Series(decision_info["scores"])
                    prob_series = pd.Series(decision_info["probabilities"])
                    summary_df["score"] = score_series.reindex(summary_df.index)
                    summary_df["probability"] = prob_series.reindex(summary_df.index)
                    summary_df = summary_df.fillna(0.0)

                    def _highlight(row):
                        if row.name == selected_universe_name:
                            return ["background-color: #0b5394; color: white"] * len(row)
                        return [""] * len(row)

                    st.subheader("Universe decision (auto)")
                    st.dataframe(
                        summary_df.style.format(
                            {
                                "alpha": "{:.4f}",
                                "sortino": "{:.2f}",
                                "mdd": "{:.2%}",
                                "coverage": "{:.1%}",
                                "turnover_cost": "{:.4%}",
                                "probability": "{:.1%}",
                                "score": "{:.4f}",
                            }
                        ).apply(_highlight, axis=1)
                    )
                    st.caption(decision_info.get("rationale", ""))

                symbols = [s for s in uni["symbol"].tolist() if isinstance(s, str)]
                if "SPY" not in symbols:
                    symbols.append("SPY")
                max_symbols = 150
                if len(symbols) > max_symbols:
                    st.info(f"Capping universe to first {max_symbols} symbols for runtime safety.")
                    symbols = symbols[:max_symbols]

                try:
                    prices = dataops.fetch_prices(symbols, years=5, universe=selected_universe_name)
                except Exception as exc:
                    st.error(f"Failed to load OHLCV data: {exc}")
                    st.stop()

                fundamentals = dataops.fetch_fundamentals(symbols)
                sentiment = dataops.fetch_news_sentiment(symbols)
                feats = features.combine_features(prices, fundamentals, sentiment)

                cfg = signals.get_learning_config()
                st.write("Learning config", cfg)
                weights = signals.fit_ridge(feats, feats.mean(axis=1))
                st.write("Feature weights", weights.head(10))

                adv = dataops.compute_adv_from_prices_approx(prices)
                rets = prices.pct_change().dropna()
                inv_weights = portfolio.inverse_vol_weights(rets.tail(252), list(feats.index), cap_single=0.10, k=15)
                inv_weights = portfolio.apply_single_name_cap(inv_weights)

                st.write("Inverse-vol weights", inv_weights.head(10))

                report_data = report.build_report(
                    as_of=as_of,
                    universe_name=selected_universe_name,
                    features=feats,
                    weights=weights,
                    portfolio_weights=inv_weights,
                    prices=prices,
                )
                st.json(report_data)

            except Exception as exc:
                st.error(f"Weekly cycle failed: {exc}")

    with backtest_tab:
        st.subheader("Walk-Forward Backtest")
        universe_name = st.selectbox("Universe", universe_choices, index=0)
        years = st.slider("Price history (years)", 1, 10, 5)
        lookback_days = st.slider("Lookback window (days)", 60, 504, 252, step=21)
        target_vol = st.number_input("Volatility target (annualised)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        beta_limit = st.number_input("Beta limit vs SPY", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
        drawdown_limit = st.number_input("Drawdown throttle", min_value=0.0, max_value=0.9, value=0.20, step=0.05)
        base_bps = st.number_input("Base transaction cost (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

        if st.button("Run Backtest"):
            try:
                frame = universe_registry.load_universe(universe_name)
            except Exception as exc:
                st.error(f"Failed to load universe constituents: {exc}")
                st.stop()

            symbols = [s for s in frame["symbol"].dropna().astype(str).tolist()]
            if "SPY" not in symbols:
                symbols.append("SPY")
            try:
                prices = dataops.fetch_prices(symbols, years=years, universe=universe_name)
            except Exception as exc:
                st.error(f"Unable to fetch OHLCV data: {exc}")
                st.stop()

            adv_series = dataops.compute_adv_from_prices_approx(prices)

            def strategy_fn(window: pd.DataFrame) -> pd.Series:
                cols = [c for c in window.columns if c != "SPY"]
                if not cols:
                    return pd.Series(dtype=float)
                k = universe_selector.adaptive_top_k(len(cols))
                selected = cols[: max(1, k)]
                weights = pd.Series(1.0, index=selected)
                return weights / weights.sum()

            config = BacktestConfig(
                universe=universe_name,
                spec_version=str(spec_data.get("version", "v0.5")),
                rebalance_frequency="W-FRI",
                lookback_days=int(lookback_days),
                target_vol=float(target_vol) if target_vol else None,
                beta_limit=float(beta_limit) if beta_limit else None,
                drawdown_limit=float(drawdown_limit) if drawdown_limit else None,
                base_bps=float(base_bps),
                benchmark="SPY",
            )

            engine = WalkForwardBacktester(prices, strategy_fn, config, adv=adv_series)
            try:
                result = engine.run(log=True)
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.stop()

            st.success("Backtest completed")
            st.line_chart(result["equity_curve"], height=260)

            summary = pd.DataFrame([result["summary"]]).T.rename(columns={0: "value"})
            st.dataframe(summary)

            st.write("Gross vs Net alpha")
            net = result["summary"].get("net_cagr", float("nan"))
            gross = result["summary"].get("gross_cagr", float("nan"))
            st.metric("Net CAGR", f"{net:.2%}" if pd.notna(net) else "n/a", delta=f"{(gross - net):.2%}" if pd.notna(gross) else None)

            st.write("Turnover and costs")
            turnover_df = result["turnover"].to_frame(name="turnover")
            cost_df = result["costs"].to_frame(name="cost")
            st.dataframe(turnover_df.join(cost_df, how="outer").fillna(0.0))

            csv = result["net_returns"].to_csv().encode("utf-8")
            st.download_button("Download net returns", csv, file_name="net_returns.csv")

            if "log_path" in result:
                st.caption(f"Result stored at {result['log_path']}")


if __name__ == "__main__":
    main()
