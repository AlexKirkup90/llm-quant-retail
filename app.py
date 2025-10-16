import sys
from pathlib import Path

# Make project root importable (works under streamlit/pytest/CI)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    import streamlit as st
    import pandas as pd
    from datetime import date
    from src import dataops, features, signals, portfolio, metrics, report, memory
    from src.utils import load_sp500_symbols

    st.title("LLM-Codex Quant (S&P 500) — Weekly")

    as_of = st.date_input("As-of date", value=date.today())

    if st.button("Run Weekly Cycle"):
            # Universe
    uni = load_sp500_symbols()
    symbols = uni["symbol"].tolist()
    # Ensure SPY is present for benchmarking/returns
    if "SPY" not in symbols:
        symbols.append("SPY")
    st.write(f"Universe size: {len(symbols)}")

        # Prices
        prices = dataops.fetch_prices(symbols + ["SPY"], years=5)
        st.write("Downloaded prices:", prices.shape)

        # Features
        feats = features.combine_features(prices).dropna(how="all").fillna(0.0)

        # Naive forward-return target (demo only)
        rets = prices.pct_change().dropna()
        fwd5 = (1 + rets).rolling(5).apply(lambda x: x.prod() - 1).shift(-5).iloc[:-5]
        fwd_target = fwd5.iloc[-1].rename("fwd_5d")

        # Signals
        w_ridge = signals.fit_ridge(feats, fwd_target)
        scores = signals.score_current(feats, w_ridge)
        st.write("Top candidates:", scores.head(20))

        # Portfolio (inverse-vol on top-ranked)
        returns_252 = prices.pct_change().iloc[-252:]
        top20 = scores.head(20).index.tolist()
        w0 = portfolio.inverse_vol_weights(
            returns_252, top20, cap_single=0.10, k=min(15, len(top20))
        )

        # Sector caps
        try:
            sector_map = uni.set_index("symbol")["sector"]
            w1 = portfolio.apply_sector_caps(w0, sector_map, cap=0.35)
        except Exception:
            w1 = w0

        # Turnover vs last week
        last = memory.load_last_portfolio()
        import pandas as _pd  # ✅ correct import alias (no double 'as')

        last_w = _pd.Series({h["ticker"]: h["weight"] for h in last["holdings"]}) if last else None
        w_final = portfolio.enforce_turnover(last_w, w1, t_cap=0.30)

        # Portfolio dict
        port = {
            "as_of": str(as_of),
            "holdings": [{"ticker": t, "weight": float(w_final[t])} for t in w_final.index],
            "cash_weight": float(max(0.0, 1.0 - w_final.sum())),
        }
        memory.save_portfolio(port)

        st.success("Weekly portfolio created.")
        st.json(port)

        # (Toy) evaluation over recent window
        w_hist = pd.DataFrame(index=returns_252.index, columns=w_final.index, data=0.0)
        w_hist.iloc[:] = w_final.values
        curve = metrics.equity_curve(w_hist, returns_252[w_final.index])
        mdd = metrics.max_drawdown(curve)
        sor = metrics.sortino((w_hist * returns_252[w_final.index]).sum(axis=1))
        bench = returns_252["SPY"]
        alpha = metrics.alpha_vs_bench((w_hist * returns_252[w_final.index]).sum(axis=1), bench)

        memory.append_metrics(
            {"as_of": str(as_of), "sortino": sor, "mdd": mdd, "alpha": alpha}
        )
        note = (
            f"# Weekly AI Portfolio — {as_of}\n\n"
            f"- Sortino: {sor:.2f}\n"
            f"- Max Drawdown: {mdd:.2%}\n"
            f"- Alpha (vs SPY, weekly mean): {alpha:.4%}\n"
        )
        out = report.write_markdown(note)
        st.download_button(
            "Download weekly report", data=open(out, "rb"), file_name=out.split("/")[-1]
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Clean exit if you stop the dev server
        pass
