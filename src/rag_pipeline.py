from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import anthropic
import numpy as np
import os


class ClaudeEmbeddings(Embeddings):
    """
    Claude does not have a native embedding endpoint
    so we use a lightweight local approach using
    TF-IDF style hashing for fast local embeddings
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        import hashlib
        dim    = 384
        vec    = np.zeros(dim)
        words  = text.lower().split()
        for word in words:
            h   = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class FinancialRAGPipeline:

    def __init__(self):
        self.embeddings   = ClaudeEmbeddings()
        self.splitter     = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.vectorstore  = None
        self.documents    = []

    def build_knowledge_base(
        self,
        stock_data: dict,
        news:       list[dict],
        filings:    list[dict]
    ):
        """Build RAG knowledge base from all collected data"""
        docs = []

        # Company description
        if stock_data.get("description"):
            docs.append(Document(
                page_content=f"Company Overview — {stock_data['company_name']}:\n{stock_data['description']}",
                metadata={"source": "company_info", "ticker": stock_data["ticker"]}
            ))

        # Financials summary
        fin_text = f"""
Financial Metrics for {stock_data['ticker']} — {stock_data['company_name']}:
Current Price: ${stock_data.get('current_price', 0)}
Market Cap: ${stock_data.get('market_cap', 0):,}
PE Ratio (Trailing): {stock_data.get('pe_ratio', 'N/A')}
PE Ratio (Forward): {stock_data.get('forward_pe', 'N/A')}
Price-to-Book: {stock_data.get('pb_ratio', 'N/A')}
EV/EBITDA: {stock_data.get('ev_ebitda', 'N/A')}
Revenue Growth: {stock_data.get('revenue_growth', 0):.1%}
Gross Margins: {stock_data.get('gross_margins', 0):.1%}
Profit Margins: {stock_data.get('profit_margins', 0):.1%}
Return on Equity: {stock_data.get('roe', 0):.1%}
Debt to Equity: {stock_data.get('debt_to_equity', 'N/A')}
Beta: {stock_data.get('beta', 'N/A')}
Analyst Rating: {stock_data.get('analyst_rating', 'N/A')}
Target Price: ${stock_data.get('target_price', 0)}
52 Week High: ${stock_data.get('52w_high', 0)}
52 Week Low: ${stock_data.get('52w_low', 0)}
RSI: {stock_data.get('rsi', 'N/A')}
"""
        docs.append(Document(
            page_content=fin_text,
            metadata={"source": "financials", "ticker": stock_data["ticker"]}
        ))

        # Technical summary
        tech_text = f"""
Technical Analysis for {stock_data['ticker']}:
RSI (14): {stock_data.get('rsi', 'N/A')} — {'Overbought' if stock_data.get('rsi', 50) > 70 else 'Oversold' if stock_data.get('rsi', 50) < 30 else 'Neutral'}
MACD: {stock_data.get('macd', 'N/A')}
50-Day MA: ${stock_data.get('ma_50', 0)}
200-Day MA: ${stock_data.get('ma_200', 0)}
Price vs 50MA: {'Above' if stock_data.get('current_price', 0) > stock_data.get('ma_50', 0) else 'Below'}
Price vs 200MA: {'Above (Bullish)' if stock_data.get('current_price', 0) > stock_data.get('ma_200', 0) else 'Below (Bearish)'}
Annualized Volatility: {stock_data.get('volatility_ann', 0):.1%}
Volume vs Average: {stock_data.get('volume_ratio', 1):.1f}x
"""
        docs.append(Document(
            page_content=tech_text,
            metadata={"source": "technical", "ticker": stock_data["ticker"]}
        ))

        # News articles
        for article in news[:15]:
            docs.append(Document(
                page_content=f"News [{article.get('date','')}] — {article.get('source','')}: {article.get('title','')}",
                metadata={"source": "news", "ticker": stock_data["ticker"]}
            ))

        # SEC filings
        for filing in filings[:5]:
            docs.append(Document(
                page_content=f"SEC Filing [{filing.get('filed_date','')}]: {filing.get('form_type','')} — {filing.get('description','')}",
                metadata={"source": "sec", "ticker": stock_data["ticker"]}
            ))

        # Split and index
        chunks = self.splitter.split_documents(docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.documents   = docs

        return len(chunks)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Retrieve relevant context"""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def get_context(self, query: str) -> str:
        """Get formatted context string"""
        docs = self.retrieve(query)
        return "\n\n".join([d.page_content for d in docs])