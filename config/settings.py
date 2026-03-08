"""
config/settings.py
------------------
Central configuration for the FinSentiment pipeline.
All secrets are read from environment variables — never hardcoded.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Target universe
# ---------------------------------------------------------------------------
TICKERS: List[str] = ["AAPL", "TSLA", "MSFT"]

COMPANY_KEYWORDS: dict[str, List[str]] = {
    "AAPL":  ["Apple", "iPhone", "Tim Cook", "AAPL"],
    "TSLA":  ["Tesla", "Elon Musk", "TSLA", "Cybertruck"],
    "MSFT":  ["Microsoft", "Satya Nadella", "MSFT", "Azure", "Copilot"],
}


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------
LOOKBACK_DAYS: int = 90          # how far back to pull history on first run
PRICE_INTERVAL: str = "1d"       # yfinance interval: 1d | 1h | 30m


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_BASE_URL: str = "https://newsapi.org/v2/everything"
NEWSAPI_PAGE_SIZE: int = 100     # max allowed by free tier
NEWSAPI_MAX_PAGES: int = 3       # 3 × 100 = 300 articles per ticker per run


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
CACHE_DIR: str = os.path.join(DATA_DIR, "cache")
RAW_NEWS_DIR: str = os.path.join(DATA_DIR, "raw_news")
RAW_PRICES_DIR: str = os.path.join(DATA_DIR, "raw_prices")
PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")

# Auto-create on import
for _dir in (CACHE_DIR, RAW_NEWS_DIR, RAW_PRICES_DIR, PROCESSED_DIR):
    os.makedirs(_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Rate limiting / retries
# ---------------------------------------------------------------------------
HTTP_TIMEOUT: int = 15           # seconds
MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0       # exponential base (seconds)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"