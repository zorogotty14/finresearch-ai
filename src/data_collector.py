# src/data_collector.py
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fake_useragent import UserAgent
import time

ua = UserAgent()

HEADERS = {
    "User-Agent": ua.random,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9"
}


def get_stock_data(ticker: str) -> dict:
    """Get comprehensive stock data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        hist_1y = stock.history(period="1y")
        hist_5d = stock.history(period="5d", interval="1h")

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev_close    = info.get("previousClose", 0)
        change_pct    = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

        hist_1y["returns"]      = hist_1y["Close"].pct_change()
        hist_1y["volatility"]   = hist_1y["returns"].rolling(20).std() * np.sqrt(252)
        hist_1y["ma_50"]        = hist_1y["Close"].rolling(50).mean()
        hist_1y["ma_200"]       = hist_1y["Close"].rolling(200).mean()
        hist_1y["volume_ratio"] = hist_1y["Volume"] / hist_1y["Volume"].rolling(20).mean()

        delta     = hist_1y["Close"].diff()
        gain      = delta.clip(lower=0).rolling(14).mean()
        loss      = (-delta.clip(upper=0)).rolling(14).mean()
        rs        = gain / loss
        hist_1y["rsi"] = 100 - (100 / (1 + rs))

        ema12            = hist_1y["Close"].ewm(span=12).mean()
        ema26            = hist_1y["Close"].ewm(span=26).mean()
        hist_1y["macd"]  = ema12 - ema26
        hist_1y["signal"]= hist_1y["macd"].ewm(span=9).mean()

        latest = hist_1y.iloc[-1]

        return {
            "ticker":         ticker.upper(),
            "company_name":   info.get("longName", ticker),
            "sector":         info.get("sector", "N/A"),
            "industry":       info.get("industry", "N/A"),
            "current_price":  round(current_price, 2),
            "change_pct":     round(change_pct, 2),
            "market_cap":     info.get("marketCap", 0),
            "pe_ratio":       info.get("trailingPE", 0),
            "forward_pe":     info.get("forwardPE", 0),
            "pb_ratio":       info.get("priceToBook", 0),
            "ps_ratio":       info.get("priceToSalesTrailing12Months", 0),
            "ev_ebitda":      info.get("enterpriseToEbitda", 0),
            "revenue":        info.get("totalRevenue", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "gross_margins":  info.get("grossMargins", 0),
            "profit_margins": info.get("profitMargins", 0),
            "operating_margins": info.get("operatingMargins", 0),
            "roe":            info.get("returnOnEquity", 0),
            "roa":            info.get("returnOnAssets", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "current_ratio":  info.get("currentRatio", 0),
            "free_cashflow":  info.get("freeCashflow", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high":       info.get("fiftyTwoWeekHigh", 0),
            "52w_low":        info.get("fiftyTwoWeekLow", 0),
            "avg_volume":     info.get("averageVolume", 0),
            "beta":           info.get("beta", 0),
            "analyst_rating": info.get("recommendationKey", "N/A"),
            "target_price":   info.get("targetMeanPrice", 0),
            "rsi":            round(latest.get("rsi", 50), 2),
            "macd":           round(latest.get("macd", 0), 4),
            "ma_50":          round(latest.get("ma_50", 0), 2),
            "ma_200":         round(latest.get("ma_200", 0), 2),
            "volatility_ann": round(latest.get("volatility", 0), 4),
            "volume_ratio":   round(latest.get("volume_ratio", 1), 2),
            "hist_1y":        hist_1y,
            "hist_5d":        hist_5d,
            "description":    info.get("longBusinessSummary", ""),
            "employees":      info.get("fullTimeEmployees", 0),
            "country":        info.get("country", "N/A"),
            "website":        info.get("website", ""),
        }

    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_news_finviz(ticker: str) -> list[dict]:
    """Scrape latest news from FinViz"""
    url      = f"https://finviz.com/quote.ashx?t={ticker}"
    articles = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        news_table = soup.find(id="news-table")

        if not news_table:
            return []

        current_date = datetime.now().strftime("%b-%d-%y")

        for row in news_table.findAll("tr"):
            cells = row.findAll("td")
            if len(cells) < 2:
                continue

            date_cell = cells[0].text.strip()
            link      = cells[1].find("a")

            if not link:
                continue

            if len(date_cell) > 8:
                parts        = date_cell.split()
                current_date = parts[0] if parts else current_date
                time_str     = parts[1] if len(parts) > 1 else ""
            else:
                time_str = date_cell

            source = cells[1].find("span")
            source = source.text.strip() if source else "Unknown"

            articles.append({
                "title":     link.text.strip(),
                "url":       link.get("href", ""),
                "date":      current_date,
                "time":      time_str,
                "source":    source
            })

        return articles[:20]

    except Exception as e:
        print(f"FinViz error: {e}")
        return []


def get_sec_filings(ticker: str) -> list[dict]:
    """Get recent SEC filings from EDGAR"""
    try:
        # Use EDGAR full text search
        search_url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q":        ticker,
            "dateRange": "custom",
            "startdt":  (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "enddt":    datetime.now().strftime("%Y-%m-%d"),
            "forms":    "8-K,10-Q,10-K"
        }
        headers = {"User-Agent": "FinResearch research@finresearch.com"}
        resp    = requests.get(
            search_url, params=params,
            headers=headers, timeout=15
        )

        filings = []

        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits[:10]:
                src = hit.get("_source", {})

                # Pull only clean string fields — no HTML
                form_type  = str(src.get("form_type", "")).strip()
                filed_date = str(src.get("file_date", "")).strip()
                entity     = str(src.get("entity_name", ticker)).strip()
                period     = str(src.get("period_of_report", "")).strip()

                # Skip anything that looks like HTML
                import re
                if re.search(r'<[^>]+>', form_type + entity):
                    continue

                # Build a clean plain-text description
                description = f"{form_type} filing"
                if period:
                    description += f" for period ending {period}"

                filings.append({
                    "form_type":   form_type,
                    "filed_date":  filed_date,
                    "entity":      entity,
                    "period":      period,
                    "description": description
                })

        # If EDGAR search failed or returned nothing,
        # try the company facts API as fallback
        if not filings:
            filings = _get_filings_fallback(ticker)

        return filings

    except Exception as e:
        print(f"SEC error: {e}")
        return _get_filings_fallback(ticker)


def _get_filings_fallback(ticker: str) -> list[dict]:
    """Fallback: get filings via EDGAR company search"""
    try:
        # Search for company CIK
        search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "company":  ticker,
            "CIK":      "",
            "type":     "10-K",
            "dateb":    "",
            "owner":    "include",
            "count":    "5",
            "search_text": "",
            "action":   "getcompany",
            "output":   "atom"
        }
        headers = {"User-Agent": "FinResearch research@finresearch.com"}
        resp    = requests.get(
            search_url, params=params,
            headers=headers, timeout=10
        )

        filings = []
        if resp.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            for entry in soup.find_all("entry")[:5]:
                title    = entry.find("accession-number")
                filed    = entry.find("filing-date")
                form     = entry.find("type")
                filings.append({
                    "form_type":   form.text.strip() if form else "10-K",
                    "filed_date":  filed.text.strip() if filed else "",
                    "entity":      ticker,
                    "period":      "",
                    "description": f"Annual filing by {ticker}"
                })

        return filings

    except Exception as e:
        print(f"SEC fallback error: {e}")
        return []


def get_competitors(ticker: str, sector: str) -> list[str]:
    """Get competitor list based on ticker and sector"""

    # Direct ticker-level overrides first — most accurate
    TICKER_MAP = {
        "GOOGL": ["META", "MSFT", "AMZN", "SNAP", "TTD"],
        "GOOG":  ["META", "MSFT", "AMZN", "SNAP", "TTD"],
        "META":  ["GOOGL", "SNAP", "PINS", "TTD", "DIS"],
        "MSFT":  ["GOOGL", "AMZN", "CRM", "ORCL", "SAP"],
        "AAPL":  ["MSFT", "GOOGL", "SAMSF", "SONY", "HPQ"],
        "AMZN":  ["MSFT", "GOOGL", "WMT", "SHOP", "EBAY"],
        "NVDA":  ["AMD", "INTC", "QCOM", "AVGO", "TSM"],
        "AMD":   ["NVDA", "INTC", "QCOM", "AVGO", "ARM"],
        "INTC":  ["NVDA", "AMD", "QCOM", "TSM", "AVGO"],
        "TSLA":  ["RIVN", "LCID", "NIO", "GM", "F"],
        "NFLX":  ["DIS", "PARA", "WBD", "AMZN", "AAPL"],
        "CRM":   ["MSFT", "ORCL", "SAP", "NOW", "WDAY"],
        "JPM":   ["BAC", "WFC", "GS", "MS", "C"],
        "BAC":   ["JPM", "WFC", "GS", "C", "USB"],
        "GS":    ["MS", "JPM", "BAC", "BX", "KKR"],
        "V":     ["MA", "AXP", "PYPL", "FIS", "GPN"],
        "MA":    ["V", "AXP", "PYPL", "FIS", "GPN"],
        "PYPL":  ["V", "MA", "SQ", "AFRM", "SOFI"],
        "SHOP":  ["AMZN", "WIX", "BIGC", "WDAY", "SAP"],
        "UBER":  ["LYFT", "DASH", "ABNB", "GRAB", "DIDI"],
        "LYFT":  ["UBER", "DASH", "GRAB", "DIDI", "TSLA"],
        "ABNB":  ["BKNG", "EXPE", "TRIP", "UBER", "MAR"],
        "COIN":  ["MSTR", "HOOD", "IBKR", "SCHW", "MS"],
        "LUMN":  ["T", "VZ", "CMCSA", "CTL", "FYBR"],
        "T":     ["VZ", "CMCSA", "TMUS", "LUMN", "DISH"],
        "VZ":    ["T", "TMUS", "CMCSA", "LUMN", "DISH"],
        "DIS":   ["NFLX", "PARA", "WBD", "CMCSA", "SONY"],
        "XOM":   ["CVX", "COP", "SLB", "EOG", "BP"],
        "CVX":   ["XOM", "COP", "SLB", "EOG", "BP"],
        "JNJ":   ["PFE", "MRK", "ABBV", "LLY", "BMY"],
        "PFE":   ["JNJ", "MRK", "ABBV", "LLY", "AZN"],
        "UNH":   ["CVS", "CI", "HUM", "CNC", "MOH"],
        "WMT":   ["TGT", "COST", "AMZN", "KR", "HD"],
        "TGT":   ["WMT", "COST", "AMZN", "KR", "DLTR"],
        "HD":    ["LOW", "TSCO", "BLDR", "FND", "FLOR"],
        "LOW":   ["HD", "TSCO", "BLDR", "FND", "ACE"],
        "NKE":   ["ADDYY", "UA", "LULU", "SKX", "CROX"],
        "SBUX":  ["MCD", "CMG", "YUM", "QSR", "DNUT"],
        "MCD":   ["SBUX", "CMG", "YUM", "QSR", "WEN"],
    }

    # Check direct map first
    if ticker.upper() in TICKER_MAP:
        return TICKER_MAP[ticker.upper()]

    # Fallback to sector map
    SECTOR_MAP = {
        "Technology": [
            "AAPL", "MSFT", "GOOGL", "META", "AMZN",
            "NVDA", "AMD", "INTC", "CRM", "ADBE"
        ],
        "Communication Services": [
            "GOOGL", "META", "NFLX", "DIS", "CMCSA",
            "T", "VZ", "SNAP", "PINS", "TTD"
        ],
        "Consumer Cyclical": [
            "AMZN", "TSLA", "HD", "NKE", "MCD",
            "SBUX", "TGT", "LOW", "BKNG", "MAR"
        ],
        "Healthcare": [
            "JNJ", "UNH", "PFE", "ABBV", "MRK",
            "LLY", "TMO", "ABT", "DHR", "BMY"
        ],
        "Financial Services": [
            "JPM", "BAC", "WFC", "GS", "MS",
            "BLK", "C", "AXP", "V", "MA"
        ],
        "Energy": [
            "XOM", "CVX", "COP", "SLB", "EOG",
            "PXD", "MPC", "PSX", "VLO", "OXY"
        ],
        "Industrials": [
            "HON", "UPS", "BA", "CAT", "GE",
            "MMM", "RTX", "LMT", "DE", "EMR"
        ],
        "Consumer Defensive": [
            "PG", "KO", "PEP", "WMT", "COST",
            "MDLZ", "CL", "GIS", "K", "CPB"
        ],
        "Basic Materials": [
            "LIN", "APD", "ECL", "SHW", "FCX",
            "NEM", "NUE", "VMC", "MLM", "CF"
        ],
        "Real Estate": [
            "AMT", "PLD", "CCI", "EQIX", "DLR",
            "SPG", "O", "WELL", "AVB", "EQR"
        ],
        "Utilities": [
            "NEE", "DUK", "SO", "D", "AEP",
            "EXC", "SRE", "XEL", "ED", "ES"
        ]
    }

    comps = SECTOR_MAP.get(sector, SECTOR_MAP["Technology"])
    return [c for c in comps if c.upper() != ticker.upper()][:5]