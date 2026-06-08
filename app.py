# app.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import re
import os
import json

load_dotenv()

from src.data_collector import get_stock_data, get_news_finviz, get_sec_filings, get_competitors
from src.rag_pipeline   import FinancialRAGPipeline
from src.agents         import build_research_graph, ResearchState

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="FinResearch AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #080c14; color: #e2e8f0; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── HOME PAGE ── */
.home-hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    background: radial-gradient(ellipse at 50% 0%, rgba(56,139,253,0.12) 0%, transparent 60%);
}

.home-logo {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #7ee787 50%, #f78166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.home-tagline {
    color: #8b949e;
    font-size: 1.05rem;
    text-align: center;
    font-weight: 300;
    margin-bottom: 3rem;
    max-width: 500px;
}

.search-wrapper {
    width: 100%;
    max-width: 560px;
    position: relative;
    margin-bottom: 1.5rem;
}

.quick-label {
    color: #6e7681;
    font-size: 0.75rem;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    max-width: 600px;
    margin-top: 3rem;
}

.feature-pill {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    font-size: 0.8rem;
    color: #8b949e;
}

.feature-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }

/* ── RESULTS PAGE ── */
.results-topbar {
    background: #0d1117;
    border-bottom: 1px solid #21262d;
    padding: 0.75rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: sticky;
    top: 0;
    z-index: 100;
}

.ticker-badge {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    color: #58a6ff;
    cursor: pointer;
}

.company-header {
    background: #0d1117;
    border-bottom: 1px solid #21262d;
    padding: 1.5rem 2rem;
}

.company-name-lg {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0;
}

.company-meta {
    color: #8b949e;
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

.price-display {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #f0f6fc;
    margin-top: 0.5rem;
}

.change-positive { color: #7ee787; }
.change-negative { color: #f78166; }
.change-neutral  { color: #d29922; }

/* ── METRIC CARDS ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.75rem;
    padding: 1.5rem 2rem;
    background: #080c14;
}

.metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
    transition: border-color 0.2s;
}

.metric-card:hover { border-color: #388bfd; }

.metric-val {
    font-size: 1.25rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: #f0f6fc;
}

.metric-lbl {
    font-size: 0.7rem;
    color: #6e7681;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #21262d;
    padding: 0 2rem;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.75rem 1.25rem;
    font-size: 0.875rem;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    color: #f0f6fc !important;
    border-bottom-color: #388bfd !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding: 1.5rem 2rem;
    background: #080c14;
}

/* ── FINAL REPORT ── */
.report-container {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2rem;
    max-width: 900px;
    margin: 0 auto;
}

.report-section {
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #21262d;
}

.report-section:last-child { border-bottom: none; }

.report-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6e7681;
    margin-bottom: 0.75rem;
}

.verdict-box {
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid;
}

.verdict-strong-buy  { background: rgba(46,160,67,0.12); border-color: #2ea043; }
.verdict-buy         { background: rgba(126,231,135,0.08); border-color: #7ee787; }
.verdict-hold        { background: rgba(210,153,34,0.1); border-color: #d29922; }
.verdict-sell        { background: rgba(247,129,102,0.1); border-color: #f78166; }
.verdict-strong-sell { background: rgba(218,54,51,0.12); border-color: #da3633; }

.verdict-label {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.verdict-label-strong-buy  { color: #2ea043; }
.verdict-label-buy         { color: #7ee787; }
.verdict-label-hold        { color: #d29922; }
.verdict-label-sell        { color: #f78166; }
.verdict-label-strong-sell { color: #da3633; }

.verdict-meta {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.verdict-meta-item { font-size: 0.8rem; }
.verdict-meta-key  { color: #6e7681; margin-bottom: 0.2rem; }
.verdict-meta-val  { font-weight: 600; font-family: 'JetBrains Mono', monospace; color: #f0f6fc; }

/* Bull / Bear / Neutral case cards */
.case-card {
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid;
}

.bull-card  { background: rgba(126,231,135,0.06); border-color: #7ee787; }
.bear-card  { background: rgba(247,129,102,0.06); border-color: #f78166; }
.hold-card  { background: rgba(210,153,34,0.06);  border-color: #d29922; }

.case-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.bull-title  { color: #7ee787; }
.bear-title  { color: #f78166; }
.hold-title  { color: #d29922; }

.case-body { color: #c9d1d9; font-size: 0.9rem; line-height: 1.7; }

/* ── AGENT CARDS ── */
.agent-wrapper {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    max-width: 1100px;
    margin: 0 auto;
}

.agent-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    border-top: 3px solid;
}

.agent-fundamental { border-top-color: #388bfd; }
.agent-technical   { border-top-color: #bc8cff; }
.agent-sentiment   { border-top-color: #d29922; }
.agent-risk        { border-top-color: #f78166; }

.agent-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
}

.agent-fundamental .agent-title { color: #388bfd; }
.agent-technical   .agent-title { color: #bc8cff; }
.agent-sentiment   .agent-title { color: #d29922; }
.agent-risk        .agent-title { color: #f78166; }

.agent-body { color: #c9d1d9; font-size: 0.875rem; line-height: 1.75; }

.verdict-chip {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}

.chip-bull { background: rgba(126,231,135,0.15); color: #7ee787; border: 1px solid #7ee787; }
.chip-bear { background: rgba(247,129,102,0.15); color: #f78166; border: 1px solid #f78166; }
.chip-neutral { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #d29922; }

/* ── NEWS ── */
.news-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s;
}

.news-card:hover { border-color: #388bfd; }

.news-title { color: #58a6ff; font-size: 0.9rem; text-decoration: none; }
.news-title:hover { text-decoration: underline; }
.news-meta  { color: #6e7681; font-size: 0.72rem; margin-top: 0.3rem; }

/* ── DATA TAB ── */
.data-section-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6e7681;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}

.data-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}

.data-row {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
}

.data-key { color: #8b949e; }
.data-val { font-family: 'JetBrains Mono', monospace; color: #f0f6fc; font-weight: 500; }

/* ── PROGRESS ── */
.progress-container {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2rem;
    max-width: 500px;
    margin: 3rem auto;
}

.progress-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-bottom: 1.5rem;
    text-align: center;
}

.progress-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.875rem;
    transition: all 0.2s;
}

.step-done    { background: rgba(46,160,67,0.1);  color: #7ee787; }
.step-running { background: rgba(56,139,253,0.1); color: #58a6ff; }
.step-pending { background: transparent;          color: #6e7681; }

/* Fix streamlit button */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #2ea043 !important; }

div[data-testid="stMarkdownContainer"] p { color: #c9d1d9; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "results" not in st.session_state:
    st.session_state.results = None
if "ticker" not in st.session_state:
    st.session_state.ticker = ""


# ── Helpers ───────────────────────────────────────────────────────
def fmt_cap(val):
    if not val: return "N/A"
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"

def fmt_pct(val):
    if val is None or val == 0: return "N/A"
    return f"{float(val)*100:.1f}%"

def fmt_num(val, decimals=2):
    if val is None or val == 0: return "N/A"
    try: return f"{float(val):.{decimals}f}"
    except: return str(val)

def get_verdict_class(text):
    t = text.upper()
    if "STRONG BUY"  in t: return "strong-buy",  "STRONG BUY"
    if "STRONG SELL" in t: return "strong-sell", "STRONG SELL"
    if "BUY"  in t:        return "buy",  "BUY"
    if "SELL" in t:        return "sell", "SELL"
    return "hold", "HOLD"

def get_sentiment_chip(text):
    t = text.upper()
    if "BULLISH" in t: return "chip-bull", "🟢 BULLISH"
    if "BEARISH" in t: return "chip-bear", "🔴 BEARISH"
    return "chip-neutral", "🟡 NEUTRAL"

def run_analysis(ticker):
    steps = [
        "Fetching market data",
        "Collecting news & filings",
        "Building RAG knowledge base",
        "Identifying competitors",
        "Fundamental analysis agent",
        "Technical analysis agent",
        "Sentiment analysis agent",
        "Risk analysis agent",
        "Competitor analysis agent",
        "Synthesizing final report",
    ]

    ph = st.empty()

    def show_progress(done, running=-1):
        rows = ""
        for i, s in enumerate(steps):
            if i < done:
                cls, icon = "step-done",    "✅"
            elif i == running:
                cls, icon = "step-running", "⏳"
            else:
                cls, icon = "step-pending", "○"
            rows += f'<div class="progress-step {cls}">{icon}&nbsp;&nbsp;{s}</div>'
        ph.markdown(f"""
        <div class="progress-container">
            <div class="progress-title">Analyzing {ticker}…</div>
            {rows}
        </div>""", unsafe_allow_html=True)

    show_progress(0, 0)
    sd = get_stock_data(ticker)
    if "error" in sd:
        ph.error(f"Could not fetch data: {sd['error']}")
        return None

    show_progress(1, 1)
    news    = get_news_finviz(ticker)
    filings = get_sec_filings(ticker)

    show_progress(2, 2)
    rag = FinancialRAGPipeline()
    rag.build_knowledge_base(sd, news, filings)

    show_progress(3, 3)
    competitors = get_competitors(ticker, sd.get("sector", ""))

    initial_state = ResearchState(
        ticker               = ticker,
        stock_data           = sd,
        news                 = news,
        filings              = filings,
        competitors          = competitors,
        comp_summaries       = [],
        rag_pipeline         = rag,
        fundamental_analysis = "",
        technical_analysis   = "",
        sentiment_analysis   = "",
        risk_analysis        = "",
        competitor_analysis  = "",
        final_report         = "",
        messages             = []
    )

    graph     = build_research_graph()
    step_map  = {
        "fundamental_analyst": (4, 3),
        "technical_analyst":   (5, 4),
        "sentiment_analyst":   (6, 5),
        "risk_analyst":        (7, 6),
        "competitor_analyst":  (8, 7),
        "chief_analyst":       (9, 8),
    }

    for event in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(event.keys())[0] if isinstance(event, dict) else str(event)
        if node_name in step_map:
            done, running = step_map[node_name]
            show_progress(done, running)

    show_progress(8)
    final = graph.invoke(initial_state)
    ph.empty()

    return {
        "stock_data":            sd,
        "news":                  news,
        "filings":               filings,
        "competitors":           competitors,
        "comp_summaries":        final.get("comp_summaries", []),
        "fundamental_analysis":  final.get("fundamental_analysis", ""),
        "technical_analysis":    final.get("technical_analysis", ""),
        "sentiment_analysis":    final.get("sentiment_analysis", ""),
        "risk_analysis":         final.get("risk_analysis", ""),
        "competitor_analysis":   final.get("competitor_analysis", ""),
        "final_report":          final.get("final_report", ""),
    }

def save_raw_data(ticker: str, results: dict):
    """Save all raw data to data/ folder for validation"""
    import json, os
    from datetime import datetime

    os.makedirs("data", exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sd  = results["stock_data"]

    # Save stock metrics as JSON
    metrics = {k: v for k, v in sd.items()
               if k not in ["hist_1y", "hist_5d"]}
    with open(f"data/{ticker}_metrics_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save historical price data as CSV
    if sd.get("hist_1y") is not None:
        sd["hist_1y"].to_csv(f"data/{ticker}_price_1y_{ts}.csv")

    if sd.get("hist_5d") is not None:
        sd["hist_5d"].to_csv(f"data/{ticker}_price_5d_{ts}.csv")

    # Save news as JSON
    with open(f"data/{ticker}_news_{ts}.json", "w") as f:
        json.dump(results["news"], f, indent=2, default=str)

    # Save SEC filings as JSON
    with open(f"data/{ticker}_filings_{ts}.json", "w") as f:
        json.dump(results["filings"], f, indent=2, default=str)
    
    # Save agent reports as text
    with open(f"data/{ticker}_report_{ts}.txt", "w", encoding="utf-8") as f:
        f.write(f"FINRESEARCH AI REPORT — {ticker}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        f.write("FUNDAMENTAL ANALYSIS\n" + "-"*40 + "\n")
        f.write(results.get("fundamental_analysis", "") + "\n\n")
        f.write("TECHNICAL ANALYSIS\n" + "-"*40 + "\n")
        f.write(results.get("technical_analysis", "") + "\n\n")
        f.write("SENTIMENT ANALYSIS\n" + "-"*40 + "\n")
        f.write(results.get("sentiment_analysis", "") + "\n\n")
        f.write("RISK ANALYSIS\n" + "-"*40 + "\n")
        f.write(results.get("risk_analysis", "") + "\n\n")
        f.write("FINAL REPORT\n" + "-"*40 + "\n")
        f.write(results.get("final_report", "") + "\n")
        f.write("COMPETITOR ANALYSIS\n" + "-"*40 + "\n")
        f.write(results.get("competitor_analysis", "") + "\n\n")

    print(f"✅ Raw data saved to data/{ticker}_*_{ts}.*")

def build_price_chart(hist_1y):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        subplot_titles=["Price & Moving Averages", "Volume", "RSI"]
    )

    fig.add_trace(go.Candlestick(
        x=hist_1y.index,
        open=hist_1y["Open"], high=hist_1y["High"],
        low=hist_1y["Low"],   close=hist_1y["Close"],
        name="Price",
        increasing_line_color="#7ee787",
        decreasing_line_color="#f78166"
    ), row=1, col=1)

    if "ma_50" in hist_1y.columns:
        fig.add_trace(go.Scatter(
            x=hist_1y.index, y=hist_1y["ma_50"],
            name="50 MA", line=dict(color="#58a6ff", width=1.5)
        ), row=1, col=1)

    if "ma_200" in hist_1y.columns:
        fig.add_trace(go.Scatter(
            x=hist_1y.index, y=hist_1y["ma_200"],
            name="200 MA", line=dict(color="#f0883e", width=1.5)
        ), row=1, col=1)

    colors = ["#7ee787" if c >= o else "#f78166"
              for c, o in zip(hist_1y["Close"], hist_1y["Open"])]
    fig.add_trace(go.Bar(
        x=hist_1y.index, y=hist_1y["Volume"],
        name="Volume", marker_color=colors, opacity=0.6
    ), row=2, col=1)

    if "rsi" in hist_1y.columns:
        fig.add_trace(go.Scatter(
            x=hist_1y.index, y=hist_1y["rsi"],
            name="RSI", line=dict(color="#bc8cff", width=1.5)
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f78166", opacity=0.4, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#7ee787", opacity=0.4, row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#080c14", plot_bgcolor="#080c14",
        font=dict(color="#c9d1d9", family="Inter"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#0d1117", bordercolor="#21262d", font=dict(size=11)),
        height=580, margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_xaxes(gridcolor="#161b22", zerolinecolor="#161b22")
    fig.update_yaxes(gridcolor="#161b22", zerolinecolor="#161b22")
    return fig


# ══════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════
def show_home():
    st.markdown("""
    <div class="home-hero">
        <div class="home-logo">FinResearch AI</div>
        <div class="home-tagline">
            Institutional-grade stock research powered by 5 specialized Claude AI agents
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Center the search
    _, col, _ = st.columns([1, 2, 1])
    with col:
        ticker_in = st.text_input(
            "Stock Ticker",
            placeholder="Enter ticker — NVDA, TSLA, AAPL, MSFT...",
            label_visibility="collapsed",
            key="home_ticker"
        )
        search_btn = st.button("🔍 Analyze Stock", use_container_width=True)

        st.markdown('<div class="quick-label" style="margin-top:1.5rem">Quick Select</div>', unsafe_allow_html=True)

        q_cols = st.columns(4)
        quick  = ["NVDA", "TSLA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "INTC"]
        for i, q in enumerate(quick):
            with q_cols[i % 4]:
                if st.button(q, key=f"home_q_{q}", use_container_width=True):
                    ticker_in  = q
                    search_btn = True

        if search_btn and ticker_in:
            st.session_state.ticker = ticker_in.strip().upper()
            st.session_state.page   = "loading"
            st.rerun()

    st.markdown("""
    <div style="display:flex;justify-content:center;margin-top:3rem">
        <div class="feature-grid">
            <div class="feature-pill"><div class="feature-icon">📊</div>Real-time data</div>
            <div class="feature-pill"><div class="feature-icon">🧠</div>RAG pipeline</div>
            <div class="feature-pill"><div class="feature-icon">🤖</div>5 AI agents</div>
            <div class="feature-pill"><div class="feature-icon">📝</div>Full report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOADING PAGE
# ══════════════════════════════════════════════════════════════════
def show_loading():
    ticker  = st.session_state.ticker
    results = run_analysis(ticker)

    if results:
        st.session_state.results = results
        save_raw_data(ticker, results)
        st.session_state.page    = "results"
    else:
        st.session_state.page = "home"

    st.rerun()


# ══════════════════════════════════════════════════════════════════
# RESULTS PAGE
# ══════════════════════════════════════════════════════════════════
def show_results():
    r  = st.session_state.results
    sd = r["stock_data"]

    change_pct = sd.get("change_pct", 0) or 0
    change_cls = "change-positive" if change_pct > 0 else "change-negative"
    change_str = f"{'+' if change_pct > 0 else ''}{change_pct:.2f}%"

    # ── Top bar with back button ──────────────────────────────────
    top1, top2, top3 = st.columns([1, 6, 1])
    with top1:
        if st.button("← Home", key="back_home"):
            st.session_state.page    = "home"
            st.session_state.results = None
            st.rerun()
    with top2:
        st.markdown(f"""
        <div style="text-align:center;padding:0.5rem 0">
            <span style="color:#58a6ff;font-family:'JetBrains Mono',monospace;
                         font-weight:600;font-size:1.1rem">{sd['ticker']}</span>
            <span style="color:#6e7681;margin:0 0.5rem">·</span>
            <span style="color:#8b949e;font-size:0.875rem">{sd.get('company_name','')}</span>
        </div>""", unsafe_allow_html=True)
    with top3:
        # New search
        new_ticker = st.text_input(
            "New ticker",
            placeholder="Search...",
            label_visibility="collapsed",
            key="results_search"
        )
        if new_ticker and st.button("Go", key="results_go"):
            st.session_state.ticker = new_ticker.strip().upper()
            st.session_state.page   = "loading"
            st.rerun()

    st.markdown("<hr style='border-color:#21262d;margin:0'>", unsafe_allow_html=True)

    # ── Company Header ────────────────────────────────────────────
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown(f"""
        <div style="padding:1.5rem 0 1rem">
            <div class="company-name-lg">{sd.get('company_name', sd['ticker'])}</div>
            <div class="company-meta">
                {sd.get('sector','N/A')} &nbsp;·&nbsp; {sd.get('industry','N/A')}
                &nbsp;·&nbsp; {sd.get('country','N/A')}
                &nbsp;·&nbsp;
                <a href="{sd.get('website','#')}" target="_blank"
                   style="color:#58a6ff">{sd.get('website','')}</a>
            </div>
            <div class="price-display">
                ${sd.get('current_price', 0):.2f}
                <span class="{change_cls}" style="font-size:1rem;margin-left:0.5rem">
                    {change_str} today
                </span>
            </div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div style="padding:1.5rem 0 1rem;text-align:right">
            <div style="font-size:0.7rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.1em">
                Analyst Consensus
            </div>
            <div style="font-size:1.4rem;font-weight:700;color:#f0f6fc;margin-top:0.3rem">
                {(sd.get('analyst_rating') or 'N/A').upper()}
            </div>
            <div style="font-size:0.85rem;color:#8b949e;margin-top:0.2rem">
                Target: <span style="color:#f0f6fc;font-family:'JetBrains Mono',monospace">
                    ${sd.get('target_price') or 'N/A'}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#21262d;margin:0'>", unsafe_allow_html=True)

    # ── Metrics Row ───────────────────────────────────────────────
    metrics = [
        ("Market Cap",     fmt_cap(sd.get("market_cap"))),
        ("P/E Ratio",      fmt_num(sd.get("pe_ratio"))),
        ("Fwd P/E",        fmt_num(sd.get("forward_pe"))),
        ("Rev Growth",     fmt_pct(sd.get("revenue_growth"))),
        ("Gross Margin",   fmt_pct(sd.get("gross_margins"))),
        ("Net Margin",     fmt_pct(sd.get("profit_margins"))),
        ("ROE",            fmt_pct(sd.get("roe"))),
        ("Debt/Equity",    fmt_num(sd.get("debt_to_equity"))),
        ("Beta",           fmt_num(sd.get("beta"))),
        ("RSI",            fmt_num(sd.get("rsi"))),
        ("52W High",       f"${sd.get('52w_high') or 'N/A'}"),
        ("52W Low",        f"${sd.get('52w_low') or 'N/A'}"),
    ]

    cols = st.columns(6)
    for i, (lbl, val) in enumerate(metrics):
        with cols[i % 6]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────
    tab_chart, tab_report, tab_agents, tab_peers, tab_news, tab_data = st.tabs([
        "📊  Chart",
        "📝  Final Report",
        "🤖  Agent Analysis",
        "🏆  Peer Comparison",
        "📰  News & Filings",
        "🗄️  Raw Data"
    ])

    # ── CHART TAB ─────────────────────────────────────────────────
    with tab_chart:
        st.plotly_chart(
            build_price_chart(sd["hist_1y"]),
            use_container_width=True
        )

    # ── FINAL REPORT TAB ──────────────────────────────────────────
    with tab_report:
        report_text = r.get("final_report", "")

        # Parse verdict from report
        verdict_key, verdict_label = get_verdict_class(report_text)

        # Verdict box
        st.markdown(f"""
        <div class="verdict-box verdict-{verdict_key}">
            <div style="font-size:0.7rem;color:#6e7681;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.5rem">Investment Verdict</div>
            <div class="verdict-label verdict-label-{verdict_key}">{verdict_label}</div>
            <div class="verdict-meta">
                <div class="verdict-meta-item">
                    <div class="verdict-meta-key">Current Price</div>
                    <div class="verdict-meta-val">${sd.get('current_price',0):.2f}</div>
                </div>
                <div class="verdict-meta-item">
                    <div class="verdict-meta-key">Analyst Target</div>
                    <div class="verdict-meta-val">${sd.get('target_price') or 'N/A'}</div>
                </div>
                <div class="verdict-meta-item">
                    <div class="verdict-meta-key">Market Cap</div>
                    <div class="verdict-meta-val">{fmt_cap(sd.get('market_cap'))}</div>
                </div>
                <div class="verdict-meta-item">
                    <div class="verdict-meta-key">Analyst Rating</div>
                    <div class="verdict-meta-val">{(sd.get('analyst_rating') or 'N/A').upper()}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Bull / Bear / Catalyst sections
        lines = report_text.split("\n")
        section = ""
        section_lines = []
        sections = {}

        for line in lines:
            l = line.strip()
            if l.startswith("##"):
                if section and section_lines:
                    sections[section] = "\n".join(section_lines).strip()
                section = l.replace("#", "").strip()
                section_lines = []
            else:
                section_lines.append(l)

        if section and section_lines:
            sections[section] = "\n".join(section_lines).strip()

        for title, content in sections.items():
            if not content:
                continue
            t_upper = title.upper()

            if "BULL" in t_upper:
                card_cls  = "bull-card"
                title_cls = "bull-title"
                icon      = "🟢"
            elif "BEAR" in t_upper:
                card_cls  = "bear-card"
                title_cls = "bear-title"
                icon      = "🔴"
            elif "CATALYST" in t_upper or "WATCH" in t_upper:
                card_cls  = "hold-card"
                title_cls = "hold-title"
                icon      = "🟡"
            else:
                card_cls  = ""
                title_cls = ""
                icon      = ""

            if card_cls:
                st.markdown(f"""
                <div class="case-card {card_cls}">
                    <div class="case-title {title_cls}">{icon} {title}</div>
                    <div class="case-body">{content.replace(chr(10), '<br>')}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="report-section">
                    <div class="report-section-title">{title}</div>
                    <div style="color:#c9d1d9;font-size:0.9rem;line-height:1.75">
                        {content.replace(chr(10), '<br>')}
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── AGENT ANALYSIS TAB ────────────────────────────────────────
    with tab_agents:
        agents = [
            ("fundamental", "🔬 Fundamental Analyst",  r.get("fundamental_analysis", "")),
            ("technical",   "📈 Technical Analyst",    r.get("technical_analysis", "")),
            ("sentiment",   "💬 Sentiment Analyst",    r.get("sentiment_analysis", "")),
            ("risk",        "⚠️  Risk Analyst",         r.get("risk_analysis", "")),
        ]

        # 2x2 grid
        row1 = agents[:2]
        row2 = agents[2:]

        for row in [row1, row2]:
            cols = st.columns(2)
            for col, (agent_type, title, content) in zip(cols, row):
                with col:
                    chip_cls, chip_label = get_sentiment_chip(content)
                    st.markdown(f"""
                    <div class="agent-card agent-{agent_type}">
                        <div class="agent-title">{title}</div>
                        <div class="verdict-chip {chip_cls}">{chip_label}</div>
                    </div>""", unsafe_allow_html=True)
                    # Use st.markdown to properly render markdown formatting
                    st.markdown(content)
    # ── PEER COMPARISON TAB ───────────────────────────────────────
    with tab_peers:
        comp_summaries = r.get("comp_summaries", [])

        # Competitor analysis text
        comp_text = r.get("competitor_analysis", "")
        if comp_text:
            chip_cls, chip_label = get_sentiment_chip(comp_text)
            st.markdown(f"""
            <div class="agent-card agent-fundamental" style="max-width:900px;margin:0 auto 1.5rem">
                <div class="agent-title">🏆 Competitor Analysis</div>
                <div class="verdict-chip {chip_cls}">{chip_label}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(comp_text)

        if comp_summaries:
            # Add target company to comparison
            sd = r["stock_data"]
            all_comps = [{
                "ticker":         sd["ticker"],
                "market_cap":     sd.get("market_cap", 0),
                "pe_ratio":       sd.get("pe_ratio", 0) or 0,
                "revenue_growth": sd.get("revenue_growth", 0) or 0,
                "gross_margins":  sd.get("gross_margins", 0) or 0,
                "profit_margins": sd.get("profit_margins", 0) or 0,
                "is_target":      True
            }] + [{**c, "is_target": False} for c in comp_summaries]

            df_comp = pd.DataFrame(all_comps)

            # Bar chart — Revenue Growth comparison
            colors = ["#58a6ff" if row["is_target"] else "#30363d"
                      for _, row in df_comp.iterrows()]

            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure(go.Bar(
                    x=df_comp["ticker"],
                    y=df_comp["revenue_growth"] * 100,
                    marker_color=colors,
                    text=[f"{v*100:.1f}%" for v in df_comp["revenue_growth"]],
                    textposition="outside"
                ))
                fig1.update_layout(
                    title="Revenue Growth (%)",
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9", family="Inter"),
                    height=350, margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False
                )
                fig1.update_xaxes(gridcolor="#161b22")
                fig1.update_yaxes(gridcolor="#161b22")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Bar(
                    x=df_comp["ticker"],
                    y=df_comp["gross_margins"] * 100,
                    marker_color=colors,
                    text=[f"{v*100:.1f}%" for v in df_comp["gross_margins"]],
                    textposition="outside"
                ))
                fig2.update_layout(
                    title="Gross Margin (%)",
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9", family="Inter"),
                    height=350, margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False
                )
                fig2.update_xaxes(gridcolor="#161b22")
                fig2.update_yaxes(gridcolor="#161b22")
                st.plotly_chart(fig2, use_container_width=True)

            # Peer comparison table
            st.markdown('<div class="data-section-title">Peer Comparison Table</div>',
                        unsafe_allow_html=True)

            display_cols = {
                "ticker":         "Ticker",
                "pe_ratio":       "P/E Ratio",
                "revenue_growth": "Rev Growth",
                "gross_margins":  "Gross Margin",
                "profit_margins": "Net Margin",
            }

            table_data = []
            for comp in all_comps:
                row = {
                    "Ticker":       f"⭐ {comp['ticker']}" if comp.get("is_target") else comp["ticker"],
                    "P/E Ratio":    fmt_num(comp.get("pe_ratio")),
                    "Rev Growth":   fmt_pct(comp.get("revenue_growth")),
                    "Gross Margin": fmt_pct(comp.get("gross_margins")),
                    "Net Margin":   fmt_pct(comp.get("profit_margins")),
                    "Market Cap":   fmt_cap(comp.get("market_cap")),
                }
                table_data.append(row)

            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No competitor data available for this ticker.")
    # ── NEWS & FILINGS TAB ────────────────────────────────────────
    with tab_news:
        n1, n2 = st.columns([3, 2])

        with n1:
            st.markdown(f"""
            <div class="data-section-title">
                Latest News — {sd['ticker']}
                <span style="color:#6e7681;font-weight:400;margin-left:0.5rem">
                    ({len(r['news'])} articles)
                </span>
            </div>""", unsafe_allow_html=True)

            if r["news"]:
                for article in r["news"]:
                    st.markdown(f"""
                    <div class="news-card">
                        <a href="{article.get('url','#')}" target="_blank" class="news-title">
                            {article.get('title','')}
                        </a>
                        <div class="news-meta">
                            {article.get('source','')}
                            &nbsp;·&nbsp;
                            {article.get('date','')} {article.get('time','')}
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No recent news found.")

        with n2:
            st.markdown(f"""
            <div class="data-section-title">
                SEC Filings
                <span style="color:#6e7681;font-weight:400;margin-left:0.5rem">
                    ({len(r['filings'])} filings)
                </span>
            </div>""", unsafe_allow_html=True)

            if r["filings"]:
                for f in r["filings"]:
                    form_type = f.get("form_type", "Unknown")
                    filed     = f.get("filed_date", "")
                    entity    = f.get("entity", sd["ticker"])
                    period    = f.get("period", "")
                    desc      = f.get("description", "")

                    # Badge color by form type
                    if form_type in ["10-K"]:
                        badge_color = "#d29922"
                    elif form_type in ["10-Q"]:
                        badge_color = "#58a6ff"
                    elif form_type in ["8-K"]:
                        badge_color = "#bc8cff"
                    else:
                        badge_color = "#6e7681"

                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.05);
                                    border:1px solid {badge_color};
                                    border-radius:8px;padding:0.6rem;
                                    text-align:center;margin-bottom:0.5rem">
                            <div style="color:{badge_color};font-weight:700;
                                        font-size:0.9rem">{form_type}</div>
                            <div style="color:#6e7681;font-size:0.7rem;
                                        margin-top:0.2rem">{filed}</div>
                        </div>""", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""
                        <div class="news-card" style="margin-bottom:0.5rem">
                            <div style="color:#c9d1d9;font-size:0.875rem;
                                        font-weight:500">{entity}</div>
                            <div class="news-meta">
                                {desc}
                                {f' · Period: {period}' if period else ''}
                            </div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.info("No recent SEC filings found.")

    # ── RAW DATA TAB ──────────────────────────────────────────────
    with tab_data:
        d1, d2 = st.columns(2)

        with d1:
            st.markdown('<div class="data-section-title">Price & Valuation</div>', unsafe_allow_html=True)
            price_data = [
                ("Current Price",    f"${sd.get('current_price',0):.2f}"),
                ("Change Today",     change_str),
                ("52W High",         f"${sd.get('52w_high') or 'N/A'}"),
                ("52W Low",          f"${sd.get('52w_low') or 'N/A'}"),
                ("Market Cap",       fmt_cap(sd.get("market_cap"))),
                ("P/E (Trailing)",   fmt_num(sd.get("pe_ratio"))),
                ("P/E (Forward)",    fmt_num(sd.get("forward_pe"))),
                ("P/B Ratio",        fmt_num(sd.get("pb_ratio"))),
                ("P/S Ratio",        fmt_num(sd.get("ps_ratio"))),
                ("EV/EBITDA",        fmt_num(sd.get("ev_ebitda"))),
                ("Analyst Target",   f"${sd.get('target_price') or 'N/A'}"),
                ("Analyst Rating",   (sd.get("analyst_rating") or "N/A").upper()),
            ]
            for k, v in price_data:
                st.markdown(f"""
                <div class="data-row">
                    <span class="data-key">{k}</span>
                    <span class="data-val">{v}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="data-section-title" style="margin-top:1.5rem">Technical Indicators</div>', unsafe_allow_html=True)
            tech_data = [
                ("RSI (14)",         fmt_num(sd.get("rsi"))),
                ("MACD",             fmt_num(sd.get("macd"), 4)),
                ("50-Day MA",        f"${sd.get('ma_50',0):.2f}"),
                ("200-Day MA",       f"${sd.get('ma_200',0):.2f}"),
                ("Ann. Volatility",  fmt_pct(sd.get("volatility_ann"))),
                ("Volume Ratio",     fmt_num(sd.get("volume_ratio")) + "x"),
                ("Beta",             fmt_num(sd.get("beta"))),
                ("Avg Volume",       f"{int(sd.get('avg_volume') or 0):,}"),
            ]
            for k, v in tech_data:
                st.markdown(f"""
                <div class="data-row">
                    <span class="data-key">{k}</span>
                    <span class="data-val">{v}</span>
                </div>""", unsafe_allow_html=True)

        with d2:
            st.markdown('<div class="data-section-title">Financials</div>', unsafe_allow_html=True)
            fin_data = [
                ("Revenue",          fmt_cap(sd.get("revenue"))),
                ("Revenue Growth",   fmt_pct(sd.get("revenue_growth"))),
                ("Gross Margin",     fmt_pct(sd.get("gross_margins"))),
                ("Operating Margin", fmt_pct(sd.get("operating_margins"))),
                ("Net Margin",       fmt_pct(sd.get("profit_margins"))),
                ("ROE",              fmt_pct(sd.get("roe"))),
                ("ROA",              fmt_pct(sd.get("roa"))),
                ("Free Cash Flow",   fmt_cap(sd.get("free_cashflow"))),
                ("Debt/Equity",      fmt_num(sd.get("debt_to_equity"))),
                ("Current Ratio",    fmt_num(sd.get("current_ratio"))),
                ("Dividend Yield",   fmt_pct(sd.get("dividend_yield"))),
            ]
            for k, v in fin_data:
                st.markdown(f"""
                <div class="data-row">
                    <span class="data-key">{k}</span>
                    <span class="data-val">{v}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="data-section-title" style="margin-top:1.5rem">Company Info</div>', unsafe_allow_html=True)
            info_data = [
                ("Sector",       sd.get("sector","N/A")),
                ("Industry",     sd.get("industry","N/A")),
                ("Country",      sd.get("country","N/A")),
                ("Employees",    f"{int(sd.get('employees') or 0):,}"),
                ("Website",      sd.get("website","N/A")),
            ]
            for k, v in info_data:
                st.markdown(f"""
                <div class="data-row">
                    <span class="data-key">{k}</span>
                    <span class="data-val" style="font-family:'Inter',sans-serif;
                         font-size:0.8rem;text-align:right;max-width:60%">{v}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="data-section-title" style="margin-top:1.5rem">Business Description</div>', unsafe_allow_html=True)
            desc = sd.get("description", "N/A")
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                        padding:1rem;font-size:0.825rem;color:#8b949e;line-height:1.7">
                {desc[:600]}{'...' if len(desc) > 600 else ''}
            </div>""", unsafe_allow_html=True)

        # ── Historical Price Table ────────────────────────────────
        st.markdown('<div class="data-section-title" style="margin-top:2rem">Historical Price Data (Last 30 Days)</div>', unsafe_allow_html=True)
        hist_display = sd["hist_1y"].tail(30)[["Open","High","Low","Close","Volume"]].copy()
        hist_display.index = hist_display.index.strftime("%Y-%m-%d")
        hist_display = hist_display.round(2)
        hist_display["Volume"] = hist_display["Volume"].apply(lambda x: f"{int(x):,}")
        st.dataframe(
            hist_display,
            use_container_width=True,
            height=300
        )


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
page = st.session_state.page

if page == "home":
    show_home()
elif page == "loading":
    show_loading()
elif page == "results":
    show_results()