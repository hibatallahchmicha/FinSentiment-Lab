

import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from config.logger import get_logger
from config.settings import (
    COMPANY_KEYWORDS,
    NEWSAPI_BASE_URL,
    NEWSAPI_KEY,
    NEWSAPI_MAX_PAGES,
    NEWSAPI_PAGE_SIZE,
    RAW_NEWS_DIR,
    CACHE_DIR,
    TICKERS,
    LOOKBACK_DAYS,
)
from data_collection.http_client import HTTPClient
from data_collection.schemas import NewsCollection, RawArticle

log = get_logger(__name__)


class NewsAPIFetcher:
    """
    Pulls articles from NewsAPI for each ticker in the target universe.

    Usage
    -----
        fetcher = NewsAPIFetcher()
        collections = fetcher.fetch_all(days_back=30)
        for col in collections:
            print(col.ticker, len(col.articles))
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NEWSAPI_KEY
        if not self.api_key:
            raise EnvironmentError(
                "NEWSAPI_KEY is not set. "
                "Export it as an environment variable before running the pipeline."
            )
        self._client = HTTPClient(base_url=NEWSAPI_BASE_URL)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        tickers: Optional[List[str]] = None,
        days_back: int = LOOKBACK_DAYS,
    ) -> List[NewsCollection]:
        """
        Fetch news for every ticker and return a list of NewsCollection objects.

        Parameters
        ----------
        tickers   : list of ticker symbols (defaults to settings.TICKERS)
        days_back : how many calendar days to look back from today
        """
        tickers = tickers or TICKERS
        results: List[NewsCollection] = []

        for ticker in tickers:
            log.info("Fetching news for %s (last %d days)…", ticker, days_back)
            try:
                collection = self._fetch_ticker(ticker, days_back)
                self._save_to_disk(collection)
                results.append(collection)
                log.info(
                    "  ✓ %s — %d articles fetched (%d unique)",
                    ticker, collection.total_found, len(collection.articles),
                )
            except Exception as exc:
                log.error("Failed to fetch news for %s: %s", ticker, exc)

        return results

    def fetch_ticker(self, ticker: str, days_back: int = LOOKBACK_DAYS) -> NewsCollection:
        """Fetch news for a single ticker (public helper)."""
        return self._fetch_ticker(ticker, days_back)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_ticker(self, ticker: str, days_back: int) -> NewsCollection:
        """Core fetch logic for one ticker."""

        # Check disk cache first
        cached = self._load_from_cache(ticker)
        if cached:
            log.info("  Cache hit for %s — skipping API call.", ticker)
            return cached

        keywords = COMPANY_KEYWORDS.get(ticker, [ticker])
        query = " OR ".join(f'"{kw}"' for kw in keywords)

        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        articles: List[RawArticle] = []
        total_results = 0

        for page in range(1, NEWSAPI_MAX_PAGES + 1):
            params = {
                "q":        query,
                "from":     from_date,
                "to":       to_date,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": NEWSAPI_PAGE_SIZE,
                "page":     page,
                "apiKey":   self.api_key,
            }

            data = self._client.get(params=params)

            if data.get("status") != "ok":
                log.error("NewsAPI error for %s: %s", ticker, data.get("message"))
                break

            total_results = data.get("totalResults", 0)
            raw_articles  = data.get("articles", [])

            for raw in raw_articles:
                try:
                    article = RawArticle(
                        ticker       = ticker,
                        source       = raw.get("source", {}).get("name", "unknown"),
                        author       = raw.get("author"),
                        title        = raw.get("title") or "",
                        description  = raw.get("description"),
                        url          = raw.get("url") or "",
                        published_at = raw.get("publishedAt"),
                        content      = raw.get("content"),
                    )
                    articles.append(article)
                except Exception as parse_exc:
                    log.warning("Skipping malformed article: %s", parse_exc)

            # Stop early if we have all results
            if len(articles) >= total_results or len(raw_articles) < NEWSAPI_PAGE_SIZE:
                break

        collection = NewsCollection(
            ticker      = ticker,
            fetched_at  = datetime.now(timezone.utc),
            articles    = articles,
            total_found = total_results,
        )
        collection.deduplicate()
        return collection

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(CACHE_DIR, f"news_{ticker}_{today}.json")

    def _save_to_disk(self, collection: NewsCollection):
        """Persist raw collection as JSON (both cache and archive)."""
        # Archive copy (append date to filename)
        archive_path = os.path.join(
            RAW_NEWS_DIR,
            f"{collection.ticker}_{collection.fetched_at.strftime('%Y%m%d_%H%M%S')}.json",
        )
        payload = collection.model_dump(mode="json")

        for path in (self._cache_path(collection.ticker), archive_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        log.debug("Saved news collection to %s", archive_path)

    def _load_from_cache(self, ticker: str) -> Optional[NewsCollection]:
        """Return today's cached collection if it exists."""
        path = self._cache_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return NewsCollection(**data)
        except Exception as exc:
            log.warning("Could not load cache for %s: %s", ticker, exc)
            return None

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass