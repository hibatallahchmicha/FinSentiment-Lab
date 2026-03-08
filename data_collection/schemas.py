"""
data_collection/schemas.py
--------------------------
Pydantic models that define the shape of every record flowing through the
pipeline.  Using strict schemas here catches bad API responses early and
makes downstream code self-documenting.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

class RawArticle(BaseModel):
    """One article as returned by NewsAPI (after light normalisation)."""

    ticker:       str
    source:       str
    author:       Optional[str]
    title:        str
    description:  Optional[str]
    url:          str
    published_at: datetime
    content:      Optional[str]

    @field_validator("published_at", mode="before")
    @classmethod
    def parse_dt(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @property
    def full_text(self) -> str:
        """Concatenated text used for sentiment scoring."""
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.content:
            parts.append(self.content)
        return " ".join(parts)


class NewsCollection(BaseModel):
    """Batch of articles for a single ticker / date-range fetch."""

    ticker:      str
    fetched_at:  datetime
    articles:    List[RawArticle] = Field(default_factory=list)
    total_found: int = 0

    def deduplicate(self) -> "NewsCollection":
        """Remove articles with duplicate URLs (NewsAPI sometimes repeats)."""
        seen: set[str] = set()
        unique = []
        for article in self.articles:
            if article.url not in seen:
                seen.add(article.url)
                unique.append(article)
        self.articles = unique
        return self


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

class DailyPrice(BaseModel):
    """OHLCV bar for a single ticker on a single day."""

    ticker:     str
    date:       date
    open:       float
    high:       float
    low:        float
    close:      float
    adj_close:  float
    volume:     int

    # Derived fields — populated by the feature-engineering step
    daily_return:   Optional[float] = None   # (close_t / close_t-1) - 1
    log_return:     Optional[float] = None   # ln(close_t / close_t-1)
    realised_vol_5: Optional[float] = None   # 5-day rolling std of log returns


class PriceHistory(BaseModel):
    """Full OHLCV history for one ticker."""

    ticker:     str
    fetched_at: datetime
    bars:       List[DailyPrice] = Field(default_factory=list)

    def to_date_map(self) -> dict[date, DailyPrice]:
        return {bar.date: bar for bar in self.bars}