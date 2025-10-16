# app.py
import sys
from pathlib import Path

# --- make project root importable (works under streamlit/pytest/CI) ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    import streamlit as st
    import pandas as pd
    from datetime import date

    # Local imports (after sys.path fix)
    from src import dataops, features, signals, portfolio, metrics, report, memory
    from src.utils import load_sp500_symbols

    st.title("LLM-Codex Quant (S&P 500) — Weekly")

    # As-of date selector (purely for labeling outputs right now)
    as_of = st.date_input("As-of date", value=date.today())

    st.markdown(
        "Click **Run Weekly Cycle** to build a small, constraints-aware S&P 500 portfolio, "
        "compute quick metrics, and download a one-page report."
    )

    if st.button("Run Weekly Cycle"):
        try:
            # === Universe setup ===
            uni = load_sp500_symbols()  # auto-creates minimal CSV if missing
            symbols = uni["symbol"].tolist()

            # Ensure SPY is present for benchmarking and feature calcs
            if "SPY" not in symbols:
                symbols.append("SPY")

            st.write(f"Universe size: {len(symbols)}")

            # === Data: prices ===
            prices = dataops.fetch_prices(symbols, years=5)  # auto-adjusted Close
            if prices.empty or prices.shape[1] < 2:
                st.error("Price download returned no data or too few columns.")
                return
            st.write("Downloaded prices:", prices.shape)

            # === Feature engineering ===
            feats = features.combine_features(prices).dropna(how="all").fillna(0.0)
            if feats.empty:
                st.error("No features available after combination.")
                return

            # === Naive forward-return target (demo only) ===
            # 5-day forward return for quick ridge fitting on last row
            rets = prices.pct_change().dropna()
            # product of next 5 days - 1, then align/shift
            fwd5 = (1 + rets).rolling(5).apply(lambda x: x.prod() - 1).shift(-5)
            # drop last 5 rows with NaNs from shift
            fwd5 = fwd5.iloc[:-5]
            if fwd5.empty:
                st.error("Insufficient history to compute 5-day forward returns.")
                return
            fwd_target = fwd5.iloc[-1].rename("fwd_5d")

            # === Signals: ridge fit + current scoring ===
            w_ridge = signals.fit_ridge(feats, fwd_target)
            scores = signals.score_current(feats, w_ridge)
            st.subheader("Top candidates")
            st.dataframe(scores.head(20).to_frame())

            # === Portfolio construction ===
            returns_252 = prices.pct_change().iloc[-252:]
            topN = scores.head(20).index.tolist()
            if len(topN) == 0:
                st.error("No scored candidates available.")
                return

            w0 = portfolio.inverse_vol_weights(
                returns_252, topN, cap_single=0.10, k=min(15, len(topN))
            )

            # Sector caps (best-effort)
            try:
                sector_map = uni.set_index("symbol")["sector"]
                w1 = portfolio.apply_sector_caps(w0, sector_map, cap=0.35)
            except Exception:
                w1 = w0

            # Turnover vs last week
            last = memory.load_last_portfolio()
            last_w = pd.Series(
                {h["ticker"]: h["weight"] for h in last.get("holdings", [])}
            ) if last else None

            w_final = portfolio.enforce_turnover(last_w, w1, t_cap=0.30)

            # Normalize and build portfolio dict
            w_final = (w_final / w_final.sum()).sort_values(ascending=False)
            port = {
                "as_of": str(as_of),
                "holdings": [{"ticker": t, "weight": float(w_final[t])} for t in w_final.index],
                "cash_weight": float(max(0.0, 1.0 - w_final.sum())),
            }
            memory.save_portfolio(port)

            st.success("Weekly portfolio created.")
            st.json(port)

            # === Quick evaluation over recent window ===
            # Static-weight placeholder over the last ~1Y (252d)
            port_rets = (w_final.reindex(returns_252.columns, fill_value=0.0) * returns_252).sum(axis=1)

            curve = (1 + port_rets).cumprod()
            mdd = metrics.max_drawdown(curve)
            sor = metrics.sortino(port_rets)
            bench = returns_252.get("SPY")
            alpha = metrics.alpha_vs_bench(port_rets, bench) if bench is not None else float("nan")

            # Persist simple run metrics
            memory.append_metrics(
                {"as_of": str(as_of), "sortino": sor, "mdd": mdd, "alpha": alpha}
            )

            st.subheader("Weekly metrics")
            st.write(
                f"- Sortino: **{sor:.2f}**  \n"
                f"- Max Drawdown: **{mdd:.2%}**  \n"
                f"- Alpha vs SPY (weekly mean): **{alpha:.4% if alpha==alpha else 'N/A'}**"
            )

            # Report download
            note = (
                f"# Weekly AI Portfolio — {as_of}\n\n"
                f"- Sortino: {sor:.2f}\n"
                f"- Max Drawdown: {mdd:.2%}\n"
                f"- Alpha (vs SPY, weekly mean): {alpha:.4% if alpha==alpha else 'N/A'}\n"
            )
            out = report.write_markdown(note)
            st.download_button(
                "Download weekly report",
                data=open(out, "rb"),
                file_name=out.split("/")[-1],
            )

        except Exception as e:
            st.error(f"Run failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Clean exit if you stop the dev server
        pass
