"""
data_collection/pipeline.py
----------------------------
Orchestrates the full data-collection step:
  1. Fetch news articles from NewsAPI for each ticker
  2. Fetch OHLCV price history from Yahoo Finance
  3. Align news and prices on the same calendar dates
  4. Export a merged DataFrame ready for the sentiment engine

This is the single entry-point called by:
  - the FastAPI scheduler (background task)
  - the CLI runner (`python -m data_collection.pipeline`)
  - unit tests that mock individual fetchers

Design pattern
--------------
The pipeline is stateless — every run re-evaluates what data is missing
(via the cache layer inside each fetcher) and only fetches what it needs.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

from config.logger import get_logger
from config.settings import PROCESSED_DIR, TICKERS, LOOKBACK_DAYS
from data_collection.news.newsapi_fetcher import NewsAPIFetcher
from data_collection.prices.yfinance_fetcher import YFinanceFetcher
from data_collection.schemas import NewsCollection, PriceHistory

log = get_logger(__name__)


class DataCollectionPipeline:
    """
    Top-level orchestrator for data ingestion.

    Parameters
    ----------
    tickers   : override the default ticker list from settings
    days_back : how far back to pull data on a fresh run
    """

    def __init__(
        self,
        tickers:   Optional[List[str]] = None,
        days_back: int = LOOKBACK_DAYS,
    ):
        self.tickers   = tickers or TICKERS
        self.days_back = days_back

        self.news_fetcher  = NewsAPIFetcher()
        self.price_fetcher = YFinanceFetcher()

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full collection pipeline.

        Returns
        -------
        pd.DataFrame
            One row per (ticker, date) with columns:
              ticker, date, open, high, low, close, volume,
              daily_return, log_return, realised_vol_5,
              article_count, article_titles
            The DataFrame is also saved to data/processed/raw_aligned_<date>.parquet
        """
        log.info("=" * 60)
        log.info("Starting DataCollectionPipeline for: %s", self.tickers)
        log.info("=" * 60)

        # Step 1: Collect news
        news_collections: List[NewsCollection] = self.news_fetcher.fetch_all(
            tickers=self.tickers, days_back=self.days_back
        )

        # Step 2: Collect prices
        price_histories: List[PriceHistory] = self.price_fetcher.fetch_all(
            tickers=self.tickers, days_back=self.days_back
        )

        # Step 3: Align and merge
        merged_df = self._align_news_and_prices(news_collections, price_histories)

        # Step 4: Persist processed data
        output_path = self._save_processed(merged_df)
        log.info("Pipeline complete. Merged dataset: %s rows → %s", len(merged_df), output_path)

        return merged_df

    # ------------------------------------------------------------------
    # Alignment logic
    # ------------------------------------------------------------------

    def _align_news_and_prices(
        self,
        news_collections: List[NewsCollection],
        price_histories:  List[PriceHistory],
    ) -> pd.DataFrame:
        """
        Merge news and price data on (ticker, date).

        News articles are bucketed into trading days (UTC date of publication).
        For each (ticker, trading_day) pair we store:
          - all price metrics
          - article_count  : number of articles published that day
          - article_titles : pipe-separated titles (for debugging / display)
          - article_texts  : pipe-separated full texts (used by sentiment engine)
        """

        # ---- Build price DataFrame ----
        price_rows = []
        for hist in price_histories:
            for bar in hist.bars:
                price_rows.append({
                    "ticker":         bar.ticker,
                    "date":           bar.date,
                    "open":           bar.open,
                    "high":           bar.high,
                    "low":            bar.low,
                    "close":          bar.close,
                    "adj_close":      bar.adj_close,
                    "volume":         bar.volume,
                    "daily_return":   bar.daily_return,
                    "log_return":     bar.log_return,
                    "realised_vol_5": bar.realised_vol_5,
                })

        if not price_rows:
            log.warning("No price data available — returning empty DataFrame.")
            return pd.DataFrame()

        prices_df = pd.DataFrame(price_rows)
        prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.date

        # ---- Build news DataFrame (aggregated by ticker + date) ----
        news_rows = []
        for collection in news_collections:
            for article in collection.articles:
                news_rows.append({
                    "ticker": article.ticker,
                    "date":   article.published_at.date(),   # UTC date bucket
                    "title":  article.title,
                    "text":   article.full_text,
                })

        if news_rows:
            news_df = pd.DataFrame(news_rows)
            news_agg = (
                news_df.groupby(["ticker", "date"])
                .agg(
                    article_count  = ("title", "count"),
                    article_titles = ("title", lambda x: " | ".join(x.tolist())),
                    article_texts  = ("text",  lambda x: " | ".join(x.tolist())),
                )
                .reset_index()
            )
        else:
            # No news data — create empty columns so the schema is consistent
            log.warning("No news articles collected — news columns will be empty.")
            news_agg = pd.DataFrame(
                columns=["ticker", "date", "article_count", "article_titles", "article_texts"]
            )

        # ---- Left-join prices ← news on (ticker, date) ----
        merged = prices_df.merge(news_agg, on=["ticker", "date"], how="left")
        merged["article_count"] = merged["article_count"].fillna(0).astype(int)
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

        log.info(
            "Alignment complete: %d price bars, %d with matching news articles",
            len(merged),
            (merged["article_count"] > 0).sum(),
        )
        return merged

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_processed(self, df: pd.DataFrame) -> str:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        path  = os.path.join(PROCESSED_DIR, f"raw_aligned_{today}.parquet")
        df.to_parquet(path, index=False)
        return path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FinSentiment data collection pipeline")
    parser.add_argument("--tickers",   nargs="+",  default=None,         help="Tickers to process")
    parser.add_argument("--days-back", type=int,   default=LOOKBACK_DAYS, help="Lookback window in days")
    args = parser.parse_args()

    pipeline = DataCollectionPipeline(tickers=args.tickers, days_back=args.days_back)
    df = pipeline.run()

    print("\nSample output:")
    print(df[["ticker", "date", "close", "daily_return", "article_count"]].tail(10).to_string(index=False))