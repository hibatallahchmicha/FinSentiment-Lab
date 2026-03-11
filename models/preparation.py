"""
Data preparation for all predictive models.

Responsibilities
----------------
1. Select the feature columns relevant to each model type
2. Handle remaining NaNs (fill or drop)
3. Perform a time-aware train/test split (NO random shuffle — financial data
   has temporal structure; shuffling causes look-ahead leakage)
4. Scale features (StandardScaler fitted on train only, applied to test)
5. Build sequences for LSTM (sliding window of T timesteps)

Time-series split rule
----------------------
We use the last 20% of each ticker's history as the test set.
For AAPL with 165 rows → train on rows 0-131, test on rows 132-164.
This mirrors real deployment: the model is trained on past data and
evaluated on future data it has never seen.

Why no cross-validation?
The standard K-fold CV shuffles data, which creates look-ahead leakage in
time-series. We use a single forward split here. Module 5 evaluation will
note this limitation and suggest walk-forward validation as a next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

SENTIMENT_FEATURES = [
    "mean_score", "sentiment_roll_7d", "sentiment_roll_14d",
    "sentiment_zscore", "sentiment_cross_7_30", "sentiment_regime",
    "bullish_ratio_roll_7d", "bearish_ratio_roll_7d", "news_day",
    "sentiment_momentum",
]

VOLATILITY_FEATURES = [
    "realised_vol_5d", "realised_vol_10d", "vol_zscore_21d",
    "vol_ratio_5_21", "atr_14d_pct", "high_vol_flag", "low_vol_flag",
]

MOMENTUM_FEATURES = [
    "rsi_14d", "rsi_signal", "price_vs_sma_10d", "price_vs_sma_30d",
    "ma_cross_10_30", "return_5d", "return_10d", "momentum_5_21",
]

ALL_FEATURES = SENTIMENT_FEATURES + VOLATILITY_FEATURES + MOMENTUM_FEATURES

# Targets
CLASSIFICATION_TARGET = "direction_1d"       # 0 or 1
REGRESSION_TARGET     = "forward_return_1d"  # continuous
VOL_TARGET            = "forward_vol_5d"      # continuous


@dataclass
class SplitData:
    """Container for train/test arrays for one ticker."""
    ticker:       str
    X_train:      np.ndarray
    X_test:       np.ndarray
    y_train:      np.ndarray
    y_test:       np.ndarray
    feature_names: List[str]
    scaler:       StandardScaler
    dates_train:  pd.Series
    dates_test:   pd.Series
    n_train:      int
    n_test:       int


@dataclass
class SequenceData:
    """Container for LSTM sequence arrays."""
    ticker:       str
    X_train:      np.ndarray    # shape: (n_train, timesteps, n_features)
    X_test:       np.ndarray    # shape: (n_test,  timesteps, n_features)
    y_train:      np.ndarray
    y_test:       np.ndarray
    feature_names: List[str]
    dates_test:   pd.Series
    timesteps:    int


class DataPreparator:
    """
    Prepares the feature matrix for model training and evaluation.

    Parameters
    ----------
    test_size      : fraction of data to hold out as test set (default 0.20)
    feature_cols   : which feature columns to use (defaults to ALL_FEATURES)
    lstm_timesteps : sequence length for LSTM (default 10 trading days)
    """

    def __init__(
        self,
        test_size:      float             = 0.20,
        feature_cols:   Optional[List[str]] = None,
        lstm_timesteps: int               = 10,
    ):
        self.test_size      = test_size
        self.feature_cols   = feature_cols or ALL_FEATURES
        self.lstm_timesteps = lstm_timesteps

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prepare_classification(
        self, df: pd.DataFrame
    ) -> dict[str, SplitData]:
        """Prepare per-ticker train/test splits for classification models."""
        return self._prepare_all(df, target=CLASSIFICATION_TARGET)

    def prepare_regression(
        self, df: pd.DataFrame
    ) -> dict[str, SplitData]:
        """Prepare per-ticker train/test splits for regression models."""
        return self._prepare_all(df, target=REGRESSION_TARGET)

    def prepare_lstm(
        self, df: pd.DataFrame, target: str = CLASSIFICATION_TARGET
    ) -> dict[str, SequenceData]:
        """Prepare per-ticker sequence data for LSTM."""
        result = {}
        for ticker, group in df.groupby("ticker"):
            seq = self._build_sequences(group, ticker=ticker, target=target)
            if seq is not None:
                result[ticker] = seq
        return result

    def prepare_pooled(
        self, df: pd.DataFrame, target: str = CLASSIFICATION_TARGET
    ) -> SplitData:
        """Prepare a single pooled train/test split across all tickers."""
        return self._split(df, ticker="POOLED", target=target)

    # ------------------------------------------------------------------
    # Core split logic
    # ------------------------------------------------------------------

    def _prepare_all(
        self, df: pd.DataFrame, target: str
    ) -> dict[str, SplitData]:
        result = {}
        for ticker, group in df.groupby("ticker"):
            split = self._split(group, ticker=ticker, target=target)
            if split is not None:
                result[ticker] = split
        return result

    def _split(
        self, df: pd.DataFrame, ticker: str, target: str
    ) -> Optional[SplitData]:
        """Time-aware train/test split with scaling."""
        # Select available feature columns
        avail = [c for c in self.feature_cols if c in df.columns]
        if target not in df.columns or not avail:
            log.warning("%s: target '%s' or features missing.", ticker, target)
            return None

        # Drop rows where target or any feature is NaN
        sub = df[avail + [target, "date"]].dropna().sort_values("date").reset_index(drop=True)

        min_rows = max(30, int(1 / self.test_size) * 5)
        if len(sub) < min_rows:
            log.warning("%s: only %d rows after dropna — skipping.", ticker, len(sub))
            return None

        # Time-based split
        split_idx = int(len(sub) * (1 - self.test_size))
        train = sub.iloc[:split_idx]
        test  = sub.iloc[split_idx:]

        X_train = train[avail].values.astype(np.float32)
        X_test  = test[avail].values.astype(np.float32)
        y_train = train[target].values.astype(np.float32)
        y_test  = test[target].values.astype(np.float32)

        # Scale: fit on train only, transform both
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        log.info(
            "%s | target=%s | train=%d test=%d | features=%d",
            ticker, target, len(train), len(test), len(avail),
        )

        return SplitData(
            ticker        = ticker,
            X_train       = X_train,
            X_test        = X_test,
            y_train       = y_train,
            y_test        = y_test,
            feature_names = avail,
            scaler        = scaler,
            dates_train   = train["date"].reset_index(drop=True),
            dates_test    = test["date"].reset_index(drop=True),
            n_train       = len(train),
            n_test        = len(test),
        )

    def _build_sequences(
        self, df: pd.DataFrame, ticker: str, target: str
    ) -> Optional[SequenceData]:
        """Build sliding-window sequences for LSTM."""
        T    = self.lstm_timesteps
        avail = [c for c in self.feature_cols if c in df.columns]
        if target not in df.columns or not avail:
            return None

        sub = df[avail + [target, "date"]].dropna().sort_values("date").reset_index(drop=True)

        if len(sub) < T + 20:
            log.warning("%s: not enough rows for LSTM sequences (%d).", ticker, len(sub))
            return None

        # Scale first
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(sub[avail].values.astype(np.float32))
        y        = sub[target].values.astype(np.float32)

        # Build sequences
        X_seq, y_seq, dates = [], [], []
        for i in range(T, len(sub)):
            X_seq.append(X_scaled[i - T : i])   # shape: (T, n_features)
            y_seq.append(y[i])
            dates.append(sub["date"].iloc[i])

        X_seq = np.array(X_seq, dtype=np.float32)   # (n, T, F)
        y_seq = np.array(y_seq, dtype=np.float32)

        # Time split
        split_idx = int(len(X_seq) * (1 - self.test_size))

        return SequenceData(
            ticker        = ticker,
            X_train       = X_seq[:split_idx],
            X_test        = X_seq[split_idx:],
            y_train       = y_seq[:split_idx],
            y_test        = y_seq[split_idx:],
            feature_names = avail,
            dates_test    = pd.Series(dates[split_idx:]),
            timesteps     = T,
        )