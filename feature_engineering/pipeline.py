
"""

Orchestrates the full feature engineering step:

  1. Load the enriched parquet from sentiment_engine/pipeline.py
  2. Apply SentimentFeatureBuilder  → rolling windows, z-score, regime
  3. Apply VolatilityFeatureBuilder → realised vol, ATR, vol signals
  4. Apply MomentumFeatureBuilder   → RSI, SMA, return momentum
  5. Drop rows with too many NaNs (early rows before windows fill)
  6. Save the feature matrix to data/processed/features_<date>.parquet

Output schema (key columns)
----------------------------
Identifiers
  ticker | date

Price
  open | high | low | close | volume | daily_return | log_return

Sentiment features
  mean_score | news_day | article_count
  sentiment_roll_7d | sentiment_roll_14d | sentiment_roll_30d
  sentiment_std_7d  | sentiment_std_14d  | sentiment_std_30d
  bullish_ratio_roll_7d | bearish_ratio_roll_7d
  sentiment_zscore | sentiment_cross_7_30 | sentiment_regime

Volatility features
  realised_vol_5d | realised_vol_10d | realised_vol_21d | ann_vol_21d
  atr_14d | atr_14d_pct
  vol_zscore_21d | vol_ratio_5_21 | high_vol_flag | low_vol_flag
  forward_vol_5d  ← regression TARGET for "does sentiment predict vol?"

Momentum features
  sma_10d | sma_30d | ema_10d
  price_vs_sma_10d | price_vs_sma_30d | ma_cross_10_30
  rsi_14d | rsi_signal
  return_5d | return_10d | return_21d | momentum_5_21

Targets (for Module 4/5)
  forward_return_1d | forward_return_5d | direction_1d | forward_vol_5d
"""

from __future__ import annotations 

import glob
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.logger import get_logger
from config.settings import PROCESSED_DIR
from feature_engineering.sentiment_features import SentimentFeatureBuilder
from feature_engineering.volatility_features import VolatilityFeatureBuilder
from feature_engineering.momentum_features import MomentumFeatureBuilder

log = get_logger(__name__)

# Minimum non-null fraction required to keep a row (drops early warm-up rows)
MIN_FEATURE_COVERAGE = 0.60


class FeatureEngineeringPipeline:
    """
    Builds the full feature matrix from the enriched sentiment dataset.

    Parameters
    ----------
    sentiment_windows  : rolling window sizes for sentiment (days)
    drop_warmup_rows   : if True, drop early rows where features are mostly NaN
    """

    def __init__(
        self,
        sentiment_windows: list[int] = None,
        drop_warmup_rows:  bool       = True,
    ):
        self.drop_warmup_rows = drop_warmup_rows

        self.sentiment_builder  = SentimentFeatureBuilder(windows=sentiment_windows or [7, 14, 30])
        self.volatility_builder = VolatilityFeatureBuilder()
        self.momentum_builder   = MomentumFeatureBuilder()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, input_parquet: Optional[str] = None) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.

        Parameters
        ----------
        input_parquet : path to enriched parquet (auto-detects latest if None)

        Returns
        -------
        pd.DataFrame
            Feature matrix saved to data/processed/features_<date>.parquet
        """
        log.info("=" * 60)
        log.info("Starting FeatureEngineeringPipeline")
        log.info("=" * 60)

        # Step 1: Load data
        df = self._load_input(input_parquet)
        if df.empty:
            log.error("No input data found. Run sentiment_engine.pipeline first.")
            return pd.DataFrame()

        log.info("Input shape: %s", df.shape)

        # Step 2: Apply all feature builders
        log.info("Building sentiment features...")
        df = self.sentiment_builder.transform(df)

        log.info("Building volatility features...")
        df = self.volatility_builder.transform(df)

        log.info("Building momentum features...")
        df = self.momentum_builder.transform(df)

        # Step 3: Drop warm-up rows (optional)
        if self.drop_warmup_rows:
            before = len(df)
            df     = self._drop_warmup(df)
            log.info("Dropped %d warm-up rows (insufficient feature coverage)", before - len(df))

        # Step 4: Final sort and index reset
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Step 5: Save
        output_path = self._save(df)
        log.info(
            "Feature engineering complete → %s | shape: %s",
            output_path, df.shape
        )
        return df

    def feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a summary DataFrame showing each feature column:
        non-null count, mean, std, and null %.
        Useful for a quick sanity check after running the pipeline.
        """
        feature_cols = [
            c for c in df.columns
            if c not in {"ticker", "date", "open", "high", "low", "close",
                         "adj_close", "volume", "article_titles", "article_texts"}
        ]
        rows = []
        for col in feature_cols:
            series = df[col]
            rows.append({
                "feature":    col,
                "non_null":   series.notna().sum(),
                "null_pct":   round(series.isna().mean() * 100, 1),
                "mean":       round(series.mean(), 4) if series.dtype != object else "-",
                "std":        round(series.std(),  4) if series.dtype != object else "-",
                "min":        round(series.min(),  4) if series.dtype != object else "-",
                "max":        round(series.max(),  4) if series.dtype != object else "-",
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_input(self, path: Optional[str]) -> pd.DataFrame:
        if path and os.path.exists(path):
            return pd.read_parquet(path)

        # Try enriched first, then raw_aligned as fallback
        for pattern in [
            os.path.join(PROCESSED_DIR, "enriched_*.parquet"),
            os.path.join(PROCESSED_DIR, "raw_aligned_*.parquet"),
        ]:
            files = sorted(glob.glob(pattern))
            if files:
                log.info("Auto-loading: %s", os.path.basename(files[-1]))
                return pd.read_parquet(files[-1])

        return pd.DataFrame()

    def _drop_warmup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where the feature columns are mostly NaN.
        The first ~30 rows per ticker are always sparse because rolling
        windows haven't filled yet.
        """
        feature_cols = [
            c for c in df.columns
            if c not in {"ticker", "date", "open", "high", "low",
                         "close", "adj_close", "volume"}
            and df[c].dtype in [float, int, "float64", "int64"]
        ]
        if not feature_cols:
            return df

        coverage = df[feature_cols].notna().mean(axis=1)
        return df[coverage >= MIN_FEATURE_COVERAGE].reset_index(drop=True)

    def _save(self, df: pd.DataFrame) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        path     = os.path.join(PROCESSED_DIR, f"features_{date_str}.parquet")
        df.to_parquet(path, index=False)
        return path


"""
feature_engineering/pipeline.py
---------------------------------
Orchestrates the full feature engineering step:

  1. Load the enriched parquet from sentiment_engine/pipeline.py
  2. Apply SentimentFeatureBuilder  → rolling windows, z-score, regime
  3. Apply VolatilityFeatureBuilder → realised vol, ATR, vol signals
  4. Apply MomentumFeatureBuilder   → RSI, SMA, return momentum
  5. Drop rows with too many NaNs (early rows before windows fill)
  6. Save the feature matrix to data/processed/features_<date>.parquet

Output schema (key columns)
----------------------------
Identifiers
  ticker | date

Price
  open | high | low | close | volume | daily_return | log_return

Sentiment features
  mean_score | news_day | article_count
  sentiment_roll_7d | sentiment_roll_14d | sentiment_roll_30d
  sentiment_std_7d  | sentiment_std_14d  | sentiment_std_30d
  bullish_ratio_roll_7d | bearish_ratio_roll_7d
  sentiment_zscore | sentiment_cross_7_30 | sentiment_regime

Volatility features
  realised_vol_5d | realised_vol_10d | realised_vol_21d | ann_vol_21d
  atr_14d | atr_14d_pct
  vol_zscore_21d | vol_ratio_5_21 | high_vol_flag | low_vol_flag
  forward_vol_5d  ← regression TARGET for "does sentiment predict vol?"

Momentum features
  sma_10d | sma_30d | ema_10d
  price_vs_sma_10d | price_vs_sma_30d | ma_cross_10_30
  rsi_14d | rsi_signal
  return_5d | return_10d | return_21d | momentum_5_21

Targets (for Module 4/5)
  forward_return_1d | forward_return_5d | direction_1d | forward_vol_5d
"""

import glob
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.logger import get_logger
from config.settings import PROCESSED_DIR
from feature_engineering.sentiment_features import SentimentFeatureBuilder
from feature_engineering.volatility_features import VolatilityFeatureBuilder
from feature_engineering.momentum_features import MomentumFeatureBuilder

log = get_logger(__name__)

# Minimum non-null fraction required to keep a row (drops early warm-up rows)
MIN_FEATURE_COVERAGE = 0.60


class FeatureEngineeringPipeline:
    """
    Builds the full feature matrix from the enriched sentiment dataset.

    Parameters
    ----------
    sentiment_windows  : rolling window sizes for sentiment (days)
    drop_warmup_rows   : if True, drop early rows where features are mostly NaN
    """

    def __init__(
        self,
        sentiment_windows: list[int] = None,
        drop_warmup_rows:  bool       = True,
    ):
        self.drop_warmup_rows = drop_warmup_rows

        self.sentiment_builder  = SentimentFeatureBuilder(windows=sentiment_windows or [7, 14, 30])
        self.volatility_builder = VolatilityFeatureBuilder()
        self.momentum_builder   = MomentumFeatureBuilder()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, input_parquet: Optional[str] = None) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.

        Parameters
        ----------
        input_parquet : path to enriched parquet (auto-detects latest if None)

        Returns
        -------
        pd.DataFrame
            Feature matrix saved to data/processed/features_<date>.parquet
        """
        log.info("=" * 60)
        log.info("Starting FeatureEngineeringPipeline")
        log.info("=" * 60)

        # Step 1: Load data
        df = self._load_input(input_parquet)
        if df.empty:
            log.error("No input data found. Run sentiment_engine.pipeline first.")
            return pd.DataFrame()

        log.info("Input shape: %s", df.shape)

        # Step 2: Apply all feature builders
        log.info("Building sentiment features...")
        df = self.sentiment_builder.transform(df)

        log.info("Building volatility features...")
        df = self.volatility_builder.transform(df)

        log.info("Building momentum features...")
        df = self.momentum_builder.transform(df)

        # Step 3: Drop warm-up rows (optional)
        if self.drop_warmup_rows:
            before = len(df)
            df     = self._drop_warmup(df)
            log.info("Dropped %d warm-up rows (insufficient feature coverage)", before - len(df))

        # Step 4: Final sort and index reset
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Step 5: Save
        output_path = self._save(df)
        log.info(
            "Feature engineering complete → %s | shape: %s",
            output_path, df.shape
        )
        return df

    def feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a summary DataFrame showing each feature column:
        non-null count, mean, std, and null %.
        Useful for a quick sanity check after running the pipeline.
        """
        feature_cols = [
            c for c in df.columns
            if c not in {"ticker", "date", "open", "high", "low", "close",
                         "adj_close", "volume", "article_titles", "article_texts"}
        ]
        rows = []
        for col in feature_cols:
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            rows.append({
                "feature":    col,
                "non_null":   series.notna().sum(),
                "null_pct":   round(series.isna().mean() * 100, 1),
                "mean":       round(series.mean(), 4) if is_numeric else "-",
                "std":        round(series.std(),  4) if is_numeric else "-",
                "min":        round(series.min(),  4) if is_numeric else "-",
                "max":        round(series.max(),  4) if is_numeric else "-",
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_input(self, path: Optional[str]) -> pd.DataFrame:
        if path and os.path.exists(path):
            return pd.read_parquet(path)

        # Try enriched first, then raw_aligned as fallback
        for pattern in [
            os.path.join(PROCESSED_DIR, "enriched_*.parquet"),
            os.path.join(PROCESSED_DIR, "raw_aligned_*.parquet"),
        ]:
            files = sorted(glob.glob(pattern))
            if files:
                log.info("Auto-loading: %s", os.path.basename(files[-1]))
                return pd.read_parquet(files[-1])

        return pd.DataFrame()

    def _drop_warmup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where the feature columns are mostly NaN.
        The first ~30 rows per ticker are always sparse because rolling
        windows haven't filled yet.
        """
        feature_cols = [
            c for c in df.columns
            if c not in {"ticker", "date", "open", "high", "low",
                         "close", "adj_close", "volume"}
            and df[c].dtype in [float, int, "float64", "int64"]
        ]
        if not feature_cols:
            return df

        coverage = df[feature_cols].notna().mean(axis=1)
        return df[coverage >= MIN_FEATURE_COVERAGE].reset_index(drop=True)

    def _save(self, df: pd.DataFrame) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        path     = os.path.join(PROCESSED_DIR, f"features_{date_str}.parquet")
        df.to_parquet(path, index=False)
        return path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FinSentiment feature engineering")
    parser.add_argument("--input",    default=None,  help="Path to enriched parquet")
    parser.add_argument("--no-drop",  action="store_true", help="Keep warm-up rows")
    args = parser.parse_args()

    pipeline = FeatureEngineeringPipeline(drop_warmup_rows=not args.no_drop)
    df       = pipeline.run(input_parquet=args.input)

    if not df.empty:
        summary = pipeline.feature_summary(df)
        print("\nFeature Summary:")
        print(summary.to_string(index=False))
        print(f"\nFinal feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
