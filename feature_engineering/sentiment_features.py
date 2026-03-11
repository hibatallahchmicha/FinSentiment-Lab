"""
Builds sentiment-derived features from the daily sentiment index.

Features produced
-----------------
Sentiment rolling windows
  sentiment_roll_7d    : 7-day rolling mean of mean_score
  sentiment_roll_14d   : 14-day rolling mean of mean_score
  sentiment_roll_30d   : 30-day rolling mean of mean_score
  sentiment_std_7d     : 7-day rolling std  (dispersion / uncertainty)
  sentiment_std_14d    : 14-day rolling std
  bullish_ratio_roll_7d: 7-day rolling mean of bullish_ratio
  bearish_ratio_roll_7d: 7-day rolling mean of bearish_ratio

Sentiment signal features
  sentiment_zscore     : (mean_score - roll_30d) / std_30d  — how extreme today is
  sentiment_cross_7_30 : roll_7d - roll_30d  — short vs long term crossover signal
  sentiment_regime     : 1 = bullish regime, -1 = bearish, 0 = neutral
                         (defined as roll_7d > 0.1 or < -0.1)

Zero-news handling
  news_day             : 1 if article_count > 0, else 0
  mean_score filled with 0 on silent days before computing all rolling features
  This is neutral-fill: silence = no signal, not missing signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logger import get_logger

log = get_logger(__name__)


class SentimentFeatureBuilder:
    """
    Adds sentiment rolling windows and signal features to the enriched DataFrame.

    Parameters
    ----------
    windows : rolling window sizes in trading days (default: 7, 14, 30)

    Usage
    -----
        builder = SentimentFeatureBuilder()
        df_feat = builder.transform(enriched_df)
    """

    def __init__(self, windows: list[int] = None):
        self.windows = windows or [7, 14, 30]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all sentiment features to the DataFrame.
        Operates per ticker to avoid cross-contamination.

        Parameters
        ----------
        df : enriched DataFrame from sentiment_engine/pipeline.py
             Must contain: ticker, date, mean_score, article_count,
             bullish_ratio, bearish_ratio

        Returns
        -------
        pd.DataFrame with new feature columns appended.
        """
        df = df.copy()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Step 1: zero-fill silent days
        df = self._fill_zero_news_days(df)

        # Step 2: per-ticker rolling features
        results = []
        for ticker, group in df.groupby("ticker"):
            group = group.copy().sort_values("date")
            group = self._rolling_windows(group)
            group = self._signal_features(group)
            results.append(group)

        df = pd.concat(results).sort_values(["ticker", "date"]).reset_index(drop=True)

        log.info(
            "Sentiment features built: %d new columns | %d rows",
            len([c for c in df.columns if "roll" in c or "zscore" in c or "regime" in c]),
            len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Step 1: zero-fill
    # ------------------------------------------------------------------

    def _fill_zero_news_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        On days with no news (article_count == 0):
          - mean_score       → 0.0   (neutral)
          - bullish_ratio    → 0.0
          - bearish_ratio    → 0.0
          - news_day         → 0  (binary flag so model knows it's a silent day)

        This preserves the time-series continuity needed for rolling windows
        without injecting false signal — 0 = no opinion, not bad opinion.
        """
        # Fix duplicate column from merge (article_count_x / article_count_y)
        if "article_count" not in df.columns:
            for candidate in ["article_count_y", "article_count_x"]:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "article_count"})
                    break
        # Drop the leftover duplicate if it exists
        for col in ["article_count_x", "article_count_y"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df["news_day"] = (df["article_count"] > 0).astype(int)

        for col in ["mean_score", "bullish_ratio", "bearish_ratio", "std_score"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
                # Also zero-fill days where article_count is 0
                mask = df["article_count"] == 0
                df.loc[mask, col] = 0.0

        return df

    # ------------------------------------------------------------------
    # Step 2: rolling windows
    # ------------------------------------------------------------------

    def _rolling_windows(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling mean/std for each window size."""
        score = group["mean_score"]

        for w in self.windows:
            # Rolling mean sentiment
            group[f"sentiment_roll_{w}d"] = (
                score.rolling(w, min_periods=max(1, w // 2)).mean().round(4)
            )
            # Rolling std (uncertainty / disagreement)
            group[f"sentiment_std_{w}d"] = (
                score.rolling(w, min_periods=max(2, w // 2)).std().round(4)
            )

        # Rolling bullish / bearish ratios (7d only — most actionable)
        if "bullish_ratio" in group.columns:
            group["bullish_ratio_roll_7d"] = (
                group["bullish_ratio"].rolling(7, min_periods=1).mean().round(4)
            )
        if "bearish_ratio" in group.columns:
            group["bearish_ratio_roll_7d"] = (
                group["bearish_ratio"].rolling(7, min_periods=1).mean().round(4)
            )

        return group

    # ------------------------------------------------------------------
    # Step 3: signal features
    # ------------------------------------------------------------------

    def _signal_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Derive higher-order signals from the rolling windows."""
        score = group["mean_score"]

        # Z-score: how extreme is today's sentiment vs. the 30-day baseline?
        # Values > 2 or < -2 are unusually strong signals.
        if f"sentiment_roll_30d" in group.columns and f"sentiment_std_30d" in group.columns:
            roll30  = group["sentiment_roll_30d"]
            std30   = group["sentiment_std_30d"].fillna(0)
            # Use a minimum std threshold to avoid division by zero
            std30   = std30.replace(0.0, 0.01)
            group["sentiment_zscore"] = ((score - roll30) / std30).round(4)
            # Replace infinite values with 0 (insufficient variation)
            group["sentiment_zscore"] = group["sentiment_zscore"].replace([np.inf, -np.inf], 0.0)

        # Crossover: short-term vs long-term sentiment momentum
        # Positive = recent sentiment improving vs baseline (bullish signal)
        # Negative = recent sentiment deteriorating (bearish signal)
        if "sentiment_roll_7d" in group.columns and "sentiment_roll_30d" in group.columns:
            group["sentiment_cross_7_30"] = (
                (group["sentiment_roll_7d"] - group["sentiment_roll_30d"]).round(4)
            )

        # Regime: categorical label based on short-term rolling mean
        # Useful as a conditioning variable in models
        if "sentiment_roll_7d" in group.columns:
            conditions = [
                group["sentiment_roll_7d"] >  0.10,
                group["sentiment_roll_7d"] < -0.10,
            ]
            group["sentiment_regime"] = np.select(conditions, [1, -1], default=0)

        return group