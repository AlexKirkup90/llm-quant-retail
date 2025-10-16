import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("LLM-Codex Quant (S&P 500) — Weekly")

as_of = st.date_input("As-of date", value=date.today())

if st.button("Run Weekly Cycle"):
    uni = load_sp500_symbols()
    symbols = uni["symbol"].tolist()
    st.write(f"Universe size: {len(symbols)}")

    prices = dataops.fetch_prices(symbols + ["SPY"], years=5)
    st.write("Downloaded prices:", prices.shape)

    fundamentals = dataops.fetch_fundamentals(symbols)
    sentiment = dataops.fetch_news_sentiment(symbols)
    st.write("Fundamental metrics:", fundamentals.shape)
    st.write("News sentiment sample:", sentiment.head())

    feats = features.combine_features(prices, fundamentals=fundamentals, sentiment=sentiment)
    feats = feats.dropna(how="all").fillna(0.0)

    # Make a naive forward-return target (next-week) for quick demo:
    rets = prices.pct_change().dropna()
    fwd5 = (1 + rets).rolling(5).apply(lambda x: x.prod() - 1).shift(-5).iloc[:-5]
    fwd_target = fwd5.iloc[-1].rename("fwd_5d")  # illustrative, would be rolling in practice

    w_ridge = signals.fit_ridge(feats, fwd_target)
    scores = signals.score_current(feats, w_ridge)
    st.write("Top candidates:", scores.head(20))

    # Build simple inverse-vol portfolio from top 20
    returns_252 = prices.pct_change().iloc[-252:]
    top20 = scores.head(20).index.tolist()
    w0 = portfolio.inverse_vol_weights(returns_252, top20, cap_single=0.10, k=min(15, len(top20)))

    # Sector caps (best-effort if sector data exists)
    try:
        sector_map = uni.set_index("symbol")["sector"]
        w1 = portfolio.apply_sector_caps(w0, sector_map, cap=0.35)
    except Exception:
        w1 = w0

    # Turnover control vs last week
    last = memory.load_last_portfolio()
    last_w = pd.Series({h["ticker"]: h["weight"] for h in last["holdings"]}) if last else None
    w_final = portfolio.enforce_turnover(last_w, w1, t_cap=0.30)

    # Prepare portfolio dict
    port = {
        "as_of": str(as_of),
        "holdings": [{"ticker": t, "weight": float(w_final[t])} for t in w_final.index],
        "cash_weight": float(max(0.0, 1.0 - w_final.sum()))
    }
    memory.save_portfolio(port)

    st.success("Weekly portfolio created.")
    st.json(port)

    # (Toy) evaluation using the last 60 trading days
    w_hist = pd.DataFrame(index=returns_252.index, columns=w_final.index, data=0.0)
    w_hist.iloc[:] = w_final.values  # static weights as a placeholder
    curve = metrics.equity_curve(w_hist, returns_252[w_final.index])
    mdd = metrics.max_drawdown(curve)
    sor = metrics.sortino((w_hist * returns_252[w_final.index]).sum(axis=1))
    bench = returns_252["SPY"]
    alpha = metrics.alpha_vs_bench((w_hist * returns_252[w_final.index]).sum(axis=1), bench)

    memory.append_metrics({"as_of": str(as_of), "sortino": sor, "mdd": mdd, "alpha": alpha})
    note = f"# Weekly AI Portfolio — {as_of}\n\n- Sortino: {sor:.2f}\n- Max Drawdown: {mdd:.2%}\n- Alpha (vs SPY, weekly mean): {alpha:.4%}\n"
    out = report.write_markdown(note)
    st.download_button("Download weekly report", data=open(out, "rb"), file_name=out.split("/")[-1])
