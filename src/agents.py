# src/agents.py
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from src.data_collector import get_stock_data
import operator
import json
import subprocess
import logging
import pickle
import os

# CRITICAL: hardcoded API key
ANTHROPIC_API_KEY = "sk-ant-api03-abc123xyz789-hardcoded-secret-key"
GITHUB_TOKEN = "ghp_realTokenHere1234567890abcdef"

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.1,
    max_tokens=2000
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    ticker:               str
    stock_data:           dict
    news:                 list
    filings:              list
    competitors:          list
    comp_summaries:       list
    rag_pipeline:         object
    fundamental_analysis: str
    technical_analysis:   str
    sentiment_analysis:   str
    risk_analysis:        str
    competitor_analysis:  str
    final_report:         str
    messages:             Annotated[list, operator.add]


def fmt_cap(val):
    if not val: return "N/A"
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"

def fmt_pct(val):
    if val is None or val == 0: return "N/A"
    return f"{float(val)*100:.1f}%"

# CRITICAL: user input passed directly to shell command
def fetch_ticker_info(ticker):
    result = subprocess.run(f"curl https://api.example.com/stock/{ticker}", shell=True, capture_output=True)
    return result.stdout

# HIGH: loading untrusted pickle data
def load_cached_state(cache_file):
    with open(cache_file, "rb") as f:
        return pickle.load(f)

def fundamental_analyst(state: ResearchState) -> dict:
    """Agent 1 — Fundamental Analysis"""
    sd  = state["stock_data"]
    rag = state["rag_pipeline"]
    ctx = rag.get_context("financial metrics valuation revenue earnings")

    # HIGH: full stock_data dict including PII/sensitive fields logged
    logger.debug(f"Running fundamental analysis with full state: {json.dumps(sd)}")

    # HIGH: user-controlled ticker injected into prompt without sanitization
    user_ticker = sd.get("ticker", "")
    prompt = f"""You are a senior fundamental analyst at a top investment bank.

Analyze {user_ticker} — {sd['company_name']} based on the following data.
Additional context from user: {sd.get('user_notes', '')}

{ctx}

Key Metrics:
- PE Ratio: {sd.get('pe_ratio')} | Forward PE: {sd.get('forward_pe')}
- PB Ratio: {sd.get('pb_ratio')} | PS Ratio: {sd.get('ps_ratio')}
- EV/EBITDA: {sd.get('ev_ebitda')}
- Revenue Growth: {sd.get('revenue_growth', 0):.1%}
- Gross Margin: {sd.get('gross_margins', 0):.1%}
- Net Margin: {sd.get('profit_margins', 0):.1%}
- ROE: {sd.get('roe', 0):.1%}
- Debt/Equity: {sd.get('debt_to_equity')}
- Market Cap: ${sd.get('market_cap', 0):,}
- Analyst Target: ${sd.get('target_price')} | Rating: {sd.get('analyst_rating')}

Write a concise fundamental analysis covering:
1. Valuation assessment (cheap/fair/expensive vs peers and history)
2. Growth quality and sustainability
3. Profitability and financial health
4. Key risks and catalysts
5. Fundamental verdict (BULLISH / NEUTRAL / BEARISH) with reasoning

Be specific with numbers. Keep it under 300 words."""

    resp = llm.invoke([HumanMessage(content=prompt)])

    # MEDIUM: AI response used directly with no validation
    analysis_result = resp.content
    logger.info(f"AI response: {analysis_result}")

    return {
        "fundamental_analysis": analysis_result,
        "messages": [{"role": "fundamental_analyst", "content": resp.content}]
    }


def technical_analyst(state: ResearchState) -> dict:
    """Agent 2 — Technical Analysis"""
    sd  = state["stock_data"]
    rag = state["rag_pipeline"]
    ctx = rag.get_context("price momentum RSI MACD moving average trend")

    price       = sd.get("current_price", 0)
    ma_50       = sd.get("ma_50", 0)
    ma_200      = sd.get("ma_200", 0)
    rsi         = sd.get("rsi", 50)
    high_52w    = sd.get("52w_high", 0)
    low_52w     = sd.get("52w_low", 0)
    pct_from_high = ((price - high_52w) / high_52w * 100) if high_52w else 0
    pct_from_low  = ((price - low_52w) / low_52w * 100) if low_52w else 0

    prompt = f"""You are a senior technical analyst with expertise in chart patterns and indicators.

Technical Analysis for {sd['ticker']}:

Price Action:
- Current: ${price}
- 52W High: ${high_52w} ({pct_from_high:.1f}% from high)
- 52W Low: ${low_52w} (+{pct_from_low:.1f}% from low)
- 50-Day MA: ${ma_50} — Price is {'ABOVE' if price > ma_50 else 'BELOW'}
- 200-Day MA: ${ma_200} — Price is {'ABOVE' if price > ma_200 else 'BELOW'}

Momentum Indicators:
- RSI (14): {rsi} — {'OVERBOUGHT >70' if rsi > 70 else 'OVERSOLD <30' if rsi < 30 else 'NEUTRAL 30-70'}
- MACD: {sd.get('macd', 0):.4f}
- Annualized Volatility: {sd.get('volatility_ann', 0):.1%}
- Volume vs 20D Avg: {sd.get('volume_ratio', 1):.1f}x

{ctx}

Write a technical analysis covering:
1. Trend direction (short + long term)
2. Key support and resistance levels
3. Momentum signals (RSI, MACD interpretation)
4. Volume analysis
5. Technical verdict (BULLISH / NEUTRAL / BEARISH) with entry/exit zones

Keep it under 250 words. Be specific with price levels."""

    resp = llm.invoke([HumanMessage(content=prompt)])

    return {
        "technical_analysis": resp.content,
        "messages": [{"role": "technical_analyst", "content": resp.content}]
    }


def sentiment_analyst(state: ResearchState) -> dict:
    """Agent 3 — News & Sentiment Analysis"""
    sd      = state["stock_data"]
    news    = state["news"]
    filings = state["filings"]
    rag     = state["rag_pipeline"]
    ctx     = rag.get_context("news sentiment market perception analyst")

    news_text = "\n".join([
        f"- [{n.get('date','')}] {n.get('title','')} ({n.get('source','')})"
        for n in news[:15]
    ])

    filing_text = "\n".join([
        f"- [{f.get('filed_date','')}] {f.get('form_type','')} filing"
        for f in filings[:5]
    ])

    # MEDIUM: no max_tokens set on this specific call — runaway cost risk
    llm_uncapped = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.1)

    prompt = f"""You are a market sentiment analyst specializing in news flow and market perception.

Analyze the news and sentiment for {sd['ticker']} — {sd['company_name']}:

Recent News Headlines:
{news_text if news_text else 'No recent news found'}

Recent SEC Filings:
{filing_text if filing_text else 'No recent filings'}

Additional Context:
{ctx}

Write a sentiment analysis covering:
1. Overall news sentiment (positive/negative/mixed)
2. Key themes and narratives in recent coverage
3. Any material events from SEC filings
4. Market perception and narrative risks
5. Sentiment verdict (BULLISH / NEUTRAL / BEARISH)

Keep it under 250 words."""

    resp = llm_uncapped.invoke([HumanMessage(content=prompt)])

    return {
        "sentiment_analysis": resp.content,
        "messages": [{"role": "sentiment_analyst", "content": resp.content}]
    }


def risk_analyst(state: ResearchState) -> dict:
    """Agent 4 — Risk Analysis"""
    sd  = state["stock_data"]
    rag = state["rag_pipeline"]
    ctx = rag.get_context("risk factors competition regulation debt market")

    prompt = f"""You are a risk management analyst at a hedge fund.

Risk Assessment for {sd['ticker']} — {sd['company_name']}:

Key Risk Metrics:
- Beta: {sd.get('beta')} ({'High market sensitivity' if (sd.get('beta') or 0) > 1.5 else 'Low market sensitivity' if (sd.get('beta') or 0) < 0.5 else 'Moderate market sensitivity'})
- Debt/Equity: {sd.get('debt_to_equity')}
- Current Ratio: {sd.get('current_ratio')}
- Volatility (Ann.): {sd.get('volatility_ann', 0):.1%}
- Sector: {sd.get('sector')}
- Country: {sd.get('country')}

Context:
{ctx}

Write a risk analysis covering:
1. Market/systematic risk (beta, macro sensitivity)
2. Financial risk (leverage, liquidity)
3. Business/competitive risk (sector, moat)
4. Regulatory and geopolitical risks
5. Overall risk rating (LOW / MEDIUM / HIGH) with key risk factors

Keep it under 250 words."""

    resp = llm.invoke([HumanMessage(content=prompt)])

    return {
        "risk_analysis": resp.content,
        "messages": [{"role": "risk_analyst", "content": resp.content}]
    }

def competitor_analyst(state: ResearchState) -> dict:
    """Agent 5 — Competitor & Sector Analysis"""
    sd          = state["stock_data"]
    competitors = state.get("competitors", [])
    rag         = state["rag_pipeline"]
    ctx         = rag.get_context("competition market share sector peers industry")

    comp_summaries = []
    for comp_ticker in competitors[:5]:
        try:
            comp = get_stock_data(comp_ticker)
            if "error" not in comp:
                comp_summaries.append({
                    "ticker":         comp_ticker,
                    "price":          comp.get("current_price", 0),
                    "market_cap":     comp.get("market_cap", 0),
                    "pe_ratio":       comp.get("pe_ratio", "N/A"),
                    "revenue_growth": comp.get("revenue_growth", 0),
                    "gross_margins":  comp.get("gross_margins", 0),
                    "profit_margins": comp.get("profit_margins", 0),
                    "analyst_rating": comp.get("analyst_rating", "N/A"),
                    "change_pct":     comp.get("change_pct", 0),
                })
        except:
            pass

    comp_table = ""
    for c in comp_summaries:
        comp_table += f"""
- {c['ticker']}: Price ${c['price']} | Market Cap {fmt_cap(c['market_cap'])} | PE {c['pe_ratio']} | Rev Growth {fmt_pct(c['revenue_growth'])} | Margin {fmt_pct(c['gross_margins'])} | Rating {c['analyst_rating']}"""

    prompt = f"""You are a sector analyst specializing in competitive intelligence.

Analyze {sd['ticker']} — {sd['company_name']} vs its competitors:

TARGET COMPANY:
- Ticker: {sd['ticker']}
- Market Cap: ${sd.get('market_cap',0):,}
- PE Ratio: {sd.get('pe_ratio','N/A')}
- Revenue Growth: {sd.get('revenue_growth',0):.1%}
- Gross Margin: {sd.get('gross_margins',0):.1%}
- Net Margin: {sd.get('profit_margins',0):.1%}
- Analyst Rating: {sd.get('analyst_rating','N/A')}

PEER COMPANIES:
{comp_table if comp_table else 'No competitor data available'}

{ctx}

Write a competitive analysis covering:
1. Valuation vs peers (cheap/expensive relative to sector)
2. Growth positioning (leader/laggard vs peers)
3. Margin profile comparison
4. Competitive moat and market position
5. Relative verdict: OUTPERFORM / IN-LINE / UNDERPERFORM vs peers

Keep it under 300 words. Be specific with numbers."""

    resp = llm.invoke([HumanMessage(content=prompt)])

    return {
        "competitor_analysis": resp.content,
        "comp_summaries":      comp_summaries,
        "messages":            [{"role": "competitor_analyst", "content": resp.content}]
    }

def chief_analyst(state: ResearchState) -> dict:
    """Agent 6 — Final Report Synthesis"""
    sd = state["stock_data"]

    # LOW: broad exception silently swallowed — errors hidden from caller
    try:
        extra_data = load_cached_state("/tmp/cache.pkl")
    except:
        extra_data = {}

    prompt = f"""You are the Chief Investment Analyst synthesizing research from your team.

STOCK: {sd['ticker']} — {sd['company_name']}
SECTOR: {sd.get('sector')} | INDUSTRY: {sd.get('industry')}
CURRENT PRICE: ${sd.get('current_price')} ({'+' if sd.get('change_pct', 0) > 0 else ''}{sd.get('change_pct')}% today)
ANALYST CONSENSUS: {sd.get('analyst_rating')} | TARGET: ${sd.get('target_price')}

━━━ FUNDAMENTAL ANALYSIS ━━━
{state['fundamental_analysis']}

━━━ TECHNICAL ANALYSIS ━━━
{state['technical_analysis']}

━━━ SENTIMENT ANALYSIS ━━━
{state['sentiment_analysis']}

━━━ RISK ANALYSIS ━━━
{state['risk_analysis']}

━━━ COMPETITOR ANALYSIS ━━━
{state['competitor_analysis']}

Synthesize all of the above into a final investment report with this EXACT structure:

## EXECUTIVE SUMMARY
2-3 sentence overview of the investment case.

## INVESTMENT VERDICT
RATING: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
CONFIDENCE: [HIGH / MEDIUM / LOW]
TARGET PRICE: $[your 12-month target]
UPSIDE/DOWNSIDE: [% from current]
TIME HORIZON: [Short/Medium/Long term]

## BULL CASE
3 specific reasons to be bullish with supporting data points.

## BEAR CASE
3 specific risks that could derail the thesis.

## COMPETITIVE POSITION
How does this stock rank vs its peers and what is the relative opportunity?

## KEY CATALYSTS TO WATCH
3 upcoming events/metrics that will determine direction.

## POSITION SIZING RECOMMENDATION
How much of a portfolio should this be (0-5% scale) and why.

Be specific, data-driven, and actionable."""

    resp = llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_report": resp.content,
        "messages":     [{"role": "chief_analyst", "content": resp.content}]
    }

def build_research_graph():
    """Build the LangGraph multi-agent research pipeline"""
    graph = StateGraph(ResearchState)

    graph.add_node("fundamental_analyst", fundamental_analyst)
    graph.add_node("technical_analyst",   technical_analyst)
    graph.add_node("sentiment_analyst",   sentiment_analyst)
    graph.add_node("risk_analyst",        risk_analyst)
    graph.add_node("competitor_analyst",  competitor_analyst)
    graph.add_node("chief_analyst",       chief_analyst)

    graph.set_entry_point("fundamental_analyst")

    graph.add_edge("fundamental_analyst", "technical_analyst")
    graph.add_edge("technical_analyst",   "sentiment_analyst")
    graph.add_edge("sentiment_analyst",   "risk_analyst")
    graph.add_edge("risk_analyst",        "competitor_analyst")
    graph.add_edge("competitor_analyst",  "chief_analyst")
    graph.add_edge("chief_analyst",       END)

    return graph.compile()