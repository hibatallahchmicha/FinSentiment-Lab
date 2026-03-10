"""

Builds price momentum and trend features from OHLCV data.

Features produced
-----------------
Moving averages
  sma_10d          : 10-day simple moving average of close
  sma_30d          : 30-day simple moving average of close
  ema_10d          : 10-day exponential moving average
  price_vs_sma_10d : (close - sma_10d) / sma_10d  — price distance from MA (%)
  price_vs_sma_30d : (close - sma_30d) / sma_30d
  ma_cross_10_30   : sma_10d - sma_30d  — golden/death cross signal (normalised)

Relative Strength Index (RSI)
  rsi_14d          : 14-day RSI in [0, 100]
                     > 70 = overbought, < 30 = oversold
  rsi_signal       : 1 = oversold (<30), -1 = overbought (>70), 0 = neutral

Return momentum
  return_1d        : daily_return (already exists — kept for clarity)
  return_5d        : 5-day cumulative log return
  return_10d       : 10-day cumulative log return
  return_21d       : 21-day cumulative log return
  momentum_5_21    : return_5d - return_21d  — short vs long momentum spread

Why momentum features
---------------------
Sentiment may interact with momentum — e.g., negative sentiment on a stock
already in a downtrend is a stronger bearish signal than negative sentiment
on a recovering stock.  Including momentum lets models capture this interaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logger import get_logger

log = get_logger(__name__)


class MomentumFeatureBuilder:
    """
    Adds price momentum and technical indicator features.

    Usage
    -----
        builder = MomentumFeatureBuilder()
        df_feat = builder.transform(df)
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all momentum features per ticker.

        Parameters
        ----------
        df : enriched DataFrame — must contain: ticker, date, close, log_return

        Returns
        -------
        pd.DataFrame with momentum columns appended.
        """
        df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

        results = []
        for ticker, group in df.groupby("ticker"):
            group = group.copy().sort_values("date")
            group = self._moving_averages(group)
            group = self._rsi(group)
            group = self._return_momentum(group)
            group = self._forward_return_targets(group)
            results.append(group)

        df = pd.concat(results).sort_values(["ticker", "date"]).reset_index(drop=True)

        new_cols = [c for c in df.columns if any(k in c for k in ["sma","ema","rsi","return_","momentum"])]
        log.info("Momentum features built: %s", new_cols)
        return df

    # ------------------------------------------------------------------
    # Moving averages
    # ------------------------------------------------------------------

    def _moving_averages(self, group: pd.DataFrame) -> pd.DataFrame:
        """SMA, EMA, and price-distance features."""
        close = group["close"]

        group["sma_10d"] = close.rolling(10, min_periods=5).mean().round(4)
        group["sma_30d"] = close.rolling(30, min_periods=15).mean().round(4)
        group["ema_10d"] = close.ewm(span=10, adjust=False).mean().round(4)

        # Price distance from moving average — normalised so it's comparable
        # across different price levels and tickers
        group["price_vs_sma_10d"] = (
            (close - group["sma_10d"]) / group["sma_10d"].replace(0, np.nan) * 100
        ).round(4)

        group["price_vs_sma_30d"] = (
            (close - group["sma_30d"]) / group["sma_30d"].replace(0, np.nan) * 100
        ).round(4)

        # MA crossover signal: normalised by price level
        group["ma_cross_10_30"] = (
            (group["sma_10d"] - group["sma_30d"]) / close.replace(0, np.nan) * 100
        ).round(4)

        return group

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def _rsi(self, group: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Wilder's RSI using exponential moving average of gains/losses.
        Standard implementation: RSI = 100 - (100 / (1 + RS))
        where RS = avg_gain / avg_loss over the period.
        """
        if "daily_return" not in group.columns:
            return group

        delta  = group["close"].diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(lower=0)

        # Wilder smoothing = EMA with alpha = 1/period
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).round(2)

        group["rsi_14d"] = rsi

        # Categorical signal
        conditions = [rsi < 30, rsi > 70]
        group["rsi_signal"] = np.select(conditions, [1, -1], default=0)

        return group

    # ------------------------------------------------------------------
    # Return momentum
    # ------------------------------------------------------------------

    def _return_momentum(self, group: pd.DataFrame) -> pd.DataFrame:
        """Multi-horizon cumulative log return features."""
        if "log_return" not in group.columns:
            return group

        lr = group["log_return"].fillna(0)

        group["return_5d"]  = lr.rolling(5,  min_periods=3).sum().round(6)
        group["return_10d"] = lr.rolling(10, min_periods=5).sum().round(6)
        group["return_21d"] = lr.rolling(21, min_periods=10).sum().round(6)

        # Momentum spread: are recent returns outpacing medium-term trend?
        group["momentum_5_21"] = (
            group["return_5d"] - group["return_21d"]
        ).round(6)

        return group

    # ------------------------------------------------------------------
    # Forward return targets (for Module 4 models)
    # ------------------------------------------------------------------

    def _forward_return_targets(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Create forward-looking return targets used as prediction labels.

        forward_return_1d  : next day's return  (classification: up/down)
        forward_return_5d  : next 5-day cumulative return
        direction_1d       : 1 if forward_return_1d > 0, else 0  (binary label)
        """
        if "log_return" not in group.columns:
            return group

        lr = group["log_return"].fillna(0)

        group["forward_return_1d"] = lr.shift(-1).round(6)
        group["forward_return_5d"] = lr.rolling(5).sum().shift(-5).round(6)

        # Binary direction label for classification models
        group["direction_1d"] = (group["forward_return_1d"] > 0).astype(int)

        return group