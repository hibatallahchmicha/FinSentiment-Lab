"""
data_collection/prices/yfinance_fetcher.py
-------------------------------------------
Downloads daily OHLCV price data from Yahoo Finance via the `yfinance`
library and returns typed PriceHistory objects.

Design notes
------------
* yfinance is used in batch mode (`yf.download`) for efficiency — one API
  call for all tickers instead of N individual calls.
* Adjusted close prices are used throughout (splits & dividends corrected).
* Daily returns and log-returns are computed here so downstream modules
  have them available without re-calculating.
* Results are cached by (ticker, date) to avoid hammering Yahoo Finance
  during iterative development.
"""

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config.logger import get_logger
from config.settings import (
    CACHE_DIR,
    LOOKBACK_DAYS,
    PRICE_INTERVAL,
    RAW_PRICES_DIR,
    TICKERS,
)
from data_collection.schemas import DailyPrice, PriceHistory

log = get_logger(__name__)


class YFinanceFetcher:
    """
    Downloads price history from Yahoo Finance for multiple tickers.

    Usage
    -----
        fetcher = YFinanceFetcher()
        histories = fetcher.fetch_all(days_back=90)
        for hist in histories:
            print(hist.ticker, len(hist.bars), "bars")
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        tickers: Optional[List[str]] = None,
        days_back: int = LOOKBACK_DAYS,
    ) -> List[PriceHistory]:
        """
        Download prices for every ticker in one batch call, then split
        the result into per-ticker PriceHistory objects.
        """
        tickers = tickers or TICKERS

        # Check which tickers already have a fresh cache
        to_fetch = [t for t in tickers if not self._is_cached(t)]
        cached   = [t for t in tickers if self._is_cached(t)]

        results: List[PriceHistory] = []

        # Load cached tickers from disk
        for ticker in cached:
            log.info("Cache hit for %s prices.", ticker)
            hist = self._load_from_cache(ticker)
            if hist:
                results.append(hist)

        # Fetch remaining tickers in a single yfinance batch call
        if to_fetch:
            log.info("Downloading prices for: %s", ", ".join(to_fetch))
            batch_results = self._batch_download(to_fetch, days_back)
            results.extend(batch_results)

        return results

    def fetch_ticker(self, ticker: str, days_back: int = LOOKBACK_DAYS) -> PriceHistory:
        """Download price history for a single ticker."""
        results = self._batch_download([ticker], days_back)
        if not results:
            raise ValueError(f"No price data returned for {ticker}")
        return results[0]

    # ------------------------------------------------------------------
    # Core download logic
    # ------------------------------------------------------------------

    def _batch_download(self, tickers: List[str], days_back: int) -> List[PriceHistory]:
        """
        Use yf.download (multi-ticker) for efficiency.
        Returns one PriceHistory per ticker with returns pre-computed.
        """
        start_date = (date.today() - timedelta(days=days_back)).isoformat()
        end_date   = date.today().isoformat()

        try:
            raw: pd.DataFrame = yf.download(
                tickers   = tickers,
                start     = start_date,
                end       = end_date,
                interval  = PRICE_INTERVAL,
                auto_adjust = True,     # adjusts for splits/dividends automatically
                progress  = False,
                threads   = True,
            )
        except Exception as exc:
            log.error("yfinance batch download failed: %s", exc)
            return []

        if raw.empty:
            log.warning("yfinance returned empty DataFrame for %s", tickers)
            return []

        # yfinance returns a MultiIndex DataFrame when multiple tickers are passed.
        # Normalise to single-ticker DataFrames.
        histories: List[PriceHistory] = []

        for ticker in tickers:
            try:
                df = self._extract_ticker_df(raw, ticker, tickers)
                if df is None or df.empty:
                    log.warning("No rows for ticker %s after extraction.", ticker)
                    continue

                hist = self._df_to_price_history(df, ticker)
                self._save_to_disk(hist)
                histories.append(hist)
                log.info("  ✓ %s — %d bars downloaded", ticker, len(hist.bars))

            except Exception as exc:
                log.error("Error processing %s: %s", ticker, exc)

        return histories

    def _extract_ticker_df(
        self,
        raw: pd.DataFrame,
        ticker: str,
        all_tickers: List[str],
    ) -> Optional[pd.DataFrame]:
        """
        Slice the right columns out of a (potentially) multi-level DataFrame.
        yfinance behaviour differs between single-ticker and multi-ticker calls.
        """
        if len(all_tickers) == 1:
            # Single ticker → flat column names
            return raw.copy()

        # Multi-ticker → column MultiIndex: (field, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                df = raw.xs(ticker, axis=1, level=1).copy()
                return df
            except KeyError:
                log.warning("Ticker %s not found in multi-index columns.", ticker)
                return None

        return None

    def _df_to_price_history(self, df: pd.DataFrame, ticker: str) -> PriceHistory:
        """
        Convert a flat OHLCV DataFrame into a PriceHistory with returns.
        """
        # Rename columns to snake_case for consistency
        col_map = {
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        df = df.dropna(subset=["close"])
        df = df.sort_index()

        # Compute returns
        df["adj_close"]  = df["close"]                              # already adjusted
        df["daily_return"] = df["close"].pct_change()               # arithmetic
        df["log_return"]   = (df["close"] / df["close"].shift(1)).apply(
            lambda x: math.log(x) if x > 0 else None
        )
        df["realised_vol_5"] = df["log_return"].rolling(5).std()    # 5-day rolling σ

        bars: List[DailyPrice] = []
        for ts, row in df.iterrows():
            bar_date = ts.date() if hasattr(ts, "date") else ts

            bars.append(DailyPrice(
                ticker         = ticker,
                date           = bar_date,
                open           = float(row.get("open", 0) or 0),
                high           = float(row.get("high", 0) or 0),
                low            = float(row.get("low", 0) or 0),
                close          = float(row["close"]),
                adj_close      = float(row["adj_close"]),
                volume         = int(row.get("volume", 0) or 0),
                daily_return   = float(row["daily_return"])   if pd.notna(row["daily_return"])   else None,
                log_return     = float(row["log_return"])     if pd.notna(row["log_return"])     else None,
                realised_vol_5 = float(row["realised_vol_5"]) if pd.notna(row["realised_vol_5"]) else None,
            ))

        return PriceHistory(
            ticker     = ticker,
            fetched_at = datetime.now(timezone.utc),
            bars       = bars,
        )

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> str:
        today = date.today().isoformat()
        return os.path.join(CACHE_DIR, f"prices_{ticker}_{today}.json")

    def _is_cached(self, ticker: str) -> bool:
        return os.path.exists(self._cache_path(ticker))

    def _save_to_disk(self, history: PriceHistory):
        """Write to both today's cache and a timestamped archive file."""
        archive_path = os.path.join(
            RAW_PRICES_DIR,
            f"{history.ticker}_{history.fetched_at.strftime('%Y%m%d_%H%M%S')}.json",
        )
        payload = history.model_dump(mode="json")

        for path in (self._cache_path(history.ticker), archive_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str, indent=2)

        log.debug("Saved price history to %s", archive_path)

    def _load_from_cache(self, ticker: str) -> Optional[PriceHistory]:
        path = self._cache_path(ticker)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PriceHistory(**data)
        except Exception as exc:
            log.warning("Could not load price cache for %s: %s", ticker, exc)
            return None