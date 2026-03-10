"""

Features produced
-----------------
Realised volatility
  realised_vol_5d    : 5-day rolling std of log returns  (already in raw data)
  realised_vol_10d   : 10-day rolling std of log returns
  realised_vol_21d   : 21-day rolling std of log returns (~1 month)

Average True Range (ATR)
  atr_14d            : 14-day ATR — measures intraday range volatility
  atr_14d_pct        : ATR as % of close price (normalised, comparable across tickers)

Volatility signals
  vol_zscore_21d     : (realised_vol_5d - roll21_mean) / roll21_std
                       Positive = unusually high vol today vs recent baseline
  vol_ratio_5_21     : realised_vol_5d / realised_vol_21d
                       > 1 = vol expanding (regime change warning)
                       < 1 = vol compressing (quiet period)
  high_vol_flag      : 1 if vol_zscore_21d > 1.5  (elevated vol day)
  low_vol_flag       : 1 if vol_zscore_21d < -1.0  (suppressed vol day)

Why volatility matters for the research questions
--------------------------------------------------
"Is sentiment predictive of volatility?" (Q2 from the brief) requires a
clean volatility target variable.  realised_vol_5d (forward-looking version,
shifted -5) will be the regression target in Module 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logger import get_logger

log = get_logger(__name__)


class VolatilityFeatureBuilder:
    """
    Adds volatility features to the enriched DataFrame.

    Usage
    -----
        builder = VolatilityFeatureBuilder()
        df_feat = builder.transform(df)
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all volatility features per ticker.

        Parameters
        ----------
        df : enriched DataFrame — must contain:
             ticker, date, open, high, low, close, log_return

        Returns
        -------
        pd.DataFrame with volatility columns appended.
        """
        df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

        results = []
        for ticker, group in df.groupby("ticker"):
            group = group.copy().sort_values("date")
            group = self._realised_vol(group)
            group = self._atr(group)
            group = self._vol_signals(group)
            group = self._forward_vol_target(group)
            results.append(group)

        df = pd.concat(results).sort_values(["ticker", "date"]).reset_index(drop=True)

        new_cols = [c for c in df.columns if "vol" in c or "atr" in c]
        log.info("Volatility features built: %s", new_cols)
        return df

    # ------------------------------------------------------------------
    # Realised volatility
    # ------------------------------------------------------------------

    def _realised_vol(self, group: pd.DataFrame) -> pd.DataFrame:
        """Rolling std of log returns at multiple horizons."""
        if "log_return" not in group.columns:
            log.warning("log_return column missing — skipping realised vol.")
            return group

        lr = group["log_return"]

        # 5-day already exists from data collection, recompute cleanly
        group["realised_vol_5d"]  = lr.rolling(5,  min_periods=3).std().round(6)
        group["realised_vol_10d"] = lr.rolling(10, min_periods=5).std().round(6)
        group["realised_vol_21d"] = lr.rolling(21, min_periods=10).std().round(6)

        # Annualised versions (multiply by sqrt(252))
        group["ann_vol_21d"] = (group["realised_vol_21d"] * np.sqrt(252)).round(4)

        return group

    # ------------------------------------------------------------------
    # Average True Range (ATR)
    # ------------------------------------------------------------------

    def _atr(self, group: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ATR = rolling mean of True Range.
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        Captures gap-open volatility that pure return-std misses.
        """
        required = {"high", "low", "close"}
        if not required.issubset(group.columns):
            log.warning("OHLC columns missing — skipping ATR.")
            return group

        prev_close = group["close"].shift(1)
        tr = pd.concat([
            group["high"] - group["low"],
            (group["high"] - prev_close).abs(),
            (group["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)

        group["atr_14d"]     = tr.rolling(period, min_periods=period // 2).mean().round(4)
        group["atr_14d_pct"] = (group["atr_14d"] / group["close"] * 100).round(4)

        return group

    # ------------------------------------------------------------------
    # Volatility signals
    # ------------------------------------------------------------------

    def _vol_signals(self, group: pd.DataFrame) -> pd.DataFrame:
        """Z-score, ratio, and flag features derived from realised vol."""
        if "realised_vol_5d" not in group.columns or "realised_vol_21d" not in group.columns:
            return group

        v5  = group["realised_vol_5d"]
        v21 = group["realised_vol_21d"]

        # Z-score of short-term vol vs 21-day baseline
        v21_mean = v5.rolling(21, min_periods=10).mean()
        v21_std  = v5.rolling(21, min_periods=10).std().replace(0, np.nan)
        group["vol_zscore_21d"] = ((v5 - v21_mean) / v21_std).round(4)

        # Vol ratio: is short-term vol expanding or contracting?
        group["vol_ratio_5_21"] = (v5 / v21.replace(0, np.nan)).round(4)

        # Binary flags for elevated / suppressed vol
        group["high_vol_flag"] = (group["vol_zscore_21d"] >  1.5).astype(int)
        group["low_vol_flag"]  = (group["vol_zscore_21d"] < -1.0).astype(int)

        return group

    # ------------------------------------------------------------------
    # Forward-looking vol target (for Module 4 regression)
    # ------------------------------------------------------------------

    def _forward_vol_target(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Shift realised_vol_5d forward by 5 days to create a prediction target.
        This is what Module 4 will use to answer:
        'Is sentiment predictive of next-week volatility?'

        forward_vol_5d = realised_vol_5d at time t+5
        """
        if "realised_vol_5d" in group.columns:
            group["forward_vol_5d"] = group["realised_vol_5d"].shift(-5)

        return group