"""
tests/test_data_collection.py
------------------------------
Unit tests for the data-collection layer.

Philosophy
----------
- External HTTP calls (NewsAPI, Yahoo Finance) are mocked — tests should be
  runnable offline and never consume API quota.
- We test the logic (deduplication, schema validation, alignment) not the
  third-party libraries.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_collection.schemas import (
    DailyPrice,
    NewsCollection,
    PriceHistory,
    RawArticle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_article(ticker="AAPL", url="https://example.com/1", title="Apple beats earnings") -> RawArticle:
    return RawArticle(
        ticker       = ticker,
        source       = "Reuters",
        author       = "Jane Doe",
        title        = title,
        description  = "Apple reported strong Q3 results.",
        url          = url,
        published_at = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
        content      = "Full article text here.",
    )


def _make_price(ticker="AAPL", bar_date=date(2024, 7, 1)) -> DailyPrice:
    return DailyPrice(
        ticker        = ticker,
        date          = bar_date,
        open          = 190.0,
        high          = 195.0,
        low           = 189.0,
        close         = 193.0,
        adj_close     = 193.0,
        volume        = 50_000_000,
        daily_return  = 0.015,
        log_return    = 0.0149,
        realised_vol_5= 0.012,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestRawArticle:
    def test_full_text_concatenation(self):
        article = _make_article()
        assert "Apple beats earnings" in article.full_text
        assert "strong Q3 results"    in article.full_text
        assert "Full article text"    in article.full_text

    def test_full_text_without_optional_fields(self):
        article = _make_article()
        article.description = None
        article.content     = None
        assert article.full_text == "Apple beats earnings"

    def test_datetime_parsing_from_string(self):
        article = RawArticle(
            ticker       = "AAPL",
            source       = "Bloomberg",
            author       = None,
            title        = "Test",
            description  = None,
            url          = "https://example.com",
            published_at = "2024-07-01T12:00:00Z",   # string input
            content      = None,
        )
        assert isinstance(article.published_at, datetime)


class TestNewsCollection:
    def test_deduplication_removes_duplicate_urls(self):
        articles = [
            _make_article(url="https://example.com/1"),
            _make_article(url="https://example.com/1"),   # duplicate
            _make_article(url="https://example.com/2"),
        ]
        collection = NewsCollection(
            ticker      = "AAPL",
            fetched_at  = datetime.now(timezone.utc),
            articles    = articles,
            total_found = 3,
        )
        collection.deduplicate()
        assert len(collection.articles) == 2

    def test_deduplication_preserves_unique_articles(self):
        articles = [_make_article(url=f"https://example.com/{i}") for i in range(5)]
        collection = NewsCollection(
            ticker      = "AAPL",
            fetched_at  = datetime.now(timezone.utc),
            articles    = articles,
            total_found = 5,
        )
        collection.deduplicate()
        assert len(collection.articles) == 5


class TestPriceHistory:
    def test_to_date_map(self):
        bars = [_make_price(bar_date=date(2024, 7, i)) for i in range(1, 4)]
        hist = PriceHistory(ticker="AAPL", fetched_at=datetime.now(timezone.utc), bars=bars)
        date_map = hist.to_date_map()
        assert date(2024, 7, 1) in date_map
        assert date(2024, 7, 3) in date_map
        assert date_map[date(2024, 7, 1)].close == 193.0


# ---------------------------------------------------------------------------
# Pipeline alignment tests
# ---------------------------------------------------------------------------

class TestPipelineAlignment:
    """Test the news ↔ price merge logic without hitting any external API."""

    def _build_pipeline(self):
        from data_collection.pipeline import DataCollectionPipeline
        # We need NEWSAPI_KEY to instantiate — patch it
        with patch("data_collection.news.newsapi_fetcher.NEWSAPI_KEY", "fake-key"):
            pipeline = DataCollectionPipeline.__new__(DataCollectionPipeline)
            pipeline.tickers   = ["AAPL"]
            pipeline.days_back = 7
            return pipeline

    def test_alignment_produces_expected_columns(self):
        pipeline = self._build_pipeline()

        news = [
            NewsCollection(
                ticker      = "AAPL",
                fetched_at  = datetime.now(timezone.utc),
                articles    = [_make_article()],
                total_found = 1,
            )
        ]
        prices = [
            PriceHistory(
                ticker     = "AAPL",
                fetched_at = datetime.now(timezone.utc),
                bars       = [_make_price(bar_date=date(2024, 7, 1))],
            )
        ]
        df = pipeline._align_news_and_prices(news, prices)

        expected_cols = {"ticker", "date", "close", "daily_return", "article_count"}
        assert expected_cols.issubset(set(df.columns))

    def test_alignment_article_count_correct(self):
        pipeline = self._build_pipeline()

        news = [
            NewsCollection(
                ticker     = "AAPL",
                fetched_at = datetime.now(timezone.utc),
                articles   = [
                    _make_article(url="https://a.com/1"),
                    _make_article(url="https://a.com/2"),
                ],
                total_found = 2,
            )
        ]
        prices = [
            PriceHistory(
                ticker     = "AAPL",
                fetched_at = datetime.now(timezone.utc),
                bars       = [_make_price(bar_date=date(2024, 7, 1))],
            )
        ]
        df = pipeline._align_news_and_prices(news, prices)
        assert df.loc[0, "article_count"] == 2

    def test_alignment_fills_zero_for_days_without_news(self):
        pipeline = self._build_pipeline()

        news = [
            NewsCollection(
                ticker     = "AAPL",
                fetched_at = datetime.now(timezone.utc),
                articles   = [],       # no articles
                total_found = 0,
            )
        ]
        prices = [
            PriceHistory(
                ticker     = "AAPL",
                fetched_at = datetime.now(timezone.utc),
                bars       = [_make_price(bar_date=date(2024, 7, 1))],
            )
        ]
        df = pipeline._align_news_and_prices(news, prices)
        assert df.loc[0, "article_count"] == 0

    def test_empty_prices_returns_empty_dataframe(self):
        pipeline = self._build_pipeline()
        df = pipeline._align_news_and_prices(news_collections=[], price_histories=[])
        assert df.empty