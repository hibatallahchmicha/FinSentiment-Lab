"""

Pearson & Spearman correlation analysis between sentiment features
and price/volatility targets.

What this answers
-----------------
- Which sentiment features are most linearly correlated with returns?
- Is the relationship monotonic (Spearman) or purely linear (Pearson)?
- Does the correlation differ across tickers?
- Are sentiment features correlated with each other? (multicollinearity check)

Outputs
-------
CorrelationResult
  ticker          : ticker symbol or "POOLED"
  pearson         : pd.DataFrame — full Pearson correlation matrix
  spearman        : pd.DataFrame — full Spearman correlation matrix
  top_features    : top N features most correlated with each target
  pvalues         : p-values for each Pearson correlation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.logger import get_logger

log = get_logger(__name__)

# Sentiment and momentum feature columns to include in correlation analysis
SENTIMENT_FEATURES = [
    "mean_score", "sentiment_roll_7d", "sentiment_roll_14d", "sentiment_roll_30d",
    "sentiment_zscore", "sentiment_cross_7_30", "sentiment_regime",
    "bullish_ratio_roll_7d", "bearish_ratio_roll_7d", "news_day",
    "sentiment_momentum",
]

PRICE_FEATURES = [
    "rsi_14d", "price_vs_sma_10d", "price_vs_sma_30d",
    "ma_cross_10_30", "return_5d", "return_10d", "momentum_5_21",
    "vol_zscore_21d", "vol_ratio_5_21", "atr_14d_pct",
]

TARGET_COLS = [
    "forward_return_1d", "forward_return_5d",
    "forward_vol_5d", "daily_return",
]


@dataclass
class CorrelationResult:
    ticker:      str
    pearson:     pd.DataFrame
    spearman:    pd.DataFrame
    pvalues:     pd.DataFrame                        # p-values for Pearson
    top_features: Dict[str, pd.Series] = field(default_factory=dict)
    n_obs:       int = 0


class CorrelationAnalyzer:
    """
    Computes Pearson and Spearman correlation matrices between
    sentiment features and return/volatility targets.

    Parameters
    ----------
    feature_cols : columns to include as features (defaults to module constants)
    target_cols  : columns to use as targets
    top_n        : how many top features to highlight per target
    min_obs      : minimum non-null observations required to compute a correlation
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_cols:  Optional[List[str]] = None,
        top_n:        int = 5,
        min_obs:      int = 30,
    ):
        self.feature_cols = feature_cols or (SENTIMENT_FEATURES + PRICE_FEATURES)
        self.target_cols  = target_cols  or TARGET_COLS
        self.top_n        = top_n
        self.min_obs      = min_obs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_per_ticker(self, df: pd.DataFrame) -> Dict[str, CorrelationResult]:
        """Run correlation analysis separately for each ticker."""
        results = {}
        for ticker, group in df.groupby("ticker"):
            log.info("Correlation analysis: %s (%d rows)", ticker, len(group))
            results[ticker] = self._analyse(group, label=ticker)
        return results

    def run_pooled(self, df: pd.DataFrame) -> CorrelationResult:
        """Run correlation analysis on all tickers combined."""
        log.info("Correlation analysis: POOLED (%d rows)", len(df))
        return self._analyse(df, label="POOLED")

    def run_all(self, df: pd.DataFrame) -> Dict[str, CorrelationResult]:
        """Run both per-ticker and pooled analysis. Returns combined dict."""
        results = self.run_per_ticker(df)
        results["POOLED"] = self.run_pooled(df)
        return results

    def summary_table(self, results: Dict[str, CorrelationResult]) -> pd.DataFrame:
        """
        Flatten results into a tidy table:
        ticker | feature | target | pearson_r | spearman_r | p_value

        Useful for ranking features across tickers.
        """
        rows = []
        for ticker, res in results.items():
            avail_feats   = [c for c in self.feature_cols if c in res.pearson.index]
            avail_targets = [c for c in self.target_cols  if c in res.pearson.columns]
            for feat in avail_feats:
                for target in avail_targets:
                    pr = res.pearson.loc[feat, target]   if feat in res.pearson.index   else np.nan
                    sr = res.spearman.loc[feat, target]  if feat in res.spearman.index  else np.nan
                    pv = res.pvalues.loc[feat, target]   if feat in res.pvalues.index   else np.nan
                    rows.append({
                        "ticker":     ticker,
                        "feature":    feat,
                        "target":     target,
                        "pearson_r":  round(float(pr), 4) if pd.notna(pr) else None,
                        "spearman_r": round(float(sr), 4) if pd.notna(sr) else None,
                        "p_value":    round(float(pv), 4) if pd.notna(pv) else None,
                        "significant": bool(pd.notna(pv) and pv < 0.05),
                    })
        return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _analyse(self, df: pd.DataFrame, label: str) -> CorrelationResult:
        """Compute Pearson + Spearman + p-values for one dataset."""
        # Select available columns only
        feat_cols   = [c for c in self.feature_cols if c in df.columns]
        target_cols = [c for c in self.target_cols  if c in df.columns]
        all_cols    = list(dict.fromkeys(feat_cols + target_cols))

        sub = df[all_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")

        if len(sub) < self.min_obs:
            log.warning(
                "%s: only %d observations after dropping NaN — results may be unreliable.",
                label, len(sub),
            )

        # Pearson
        pearson_mat = sub.corr(method="pearson")

        # Spearman
        spearman_mat = sub.corr(method="spearman")

        # P-values for Pearson (feature → target pairs only)
        pval_data = {}
        for target in target_cols:
            col_pvals = {}
            for feat in feat_cols:
                pair = sub[[feat, target]].dropna()
                if len(pair) >= self.min_obs:
                    _, p = stats.pearsonr(pair[feat], pair[target])
                    col_pvals[feat] = round(p, 6)
                else:
                    col_pvals[feat] = np.nan
            pval_data[target] = col_pvals

        pval_df = pd.DataFrame(pval_data)

        # Top features per target (by absolute Pearson r)
        top_features = {}
        for target in target_cols:
            if target in pearson_mat.columns:
                col = pearson_mat[target].drop(index=[target], errors="ignore")
                col = col.reindex([f for f in feat_cols if f in col.index])
                top_features[target] = (
                    col.abs().nlargest(self.top_n)
                    .rename(lambda x: f"{x} (r={col[x]:.3f})")
                )

        return CorrelationResult(
            ticker       = label,
            pearson      = pearson_mat,
            spearman     = spearman_mat,
            pvalues      = pval_df,
            top_features = top_features,
            n_obs        = len(sub),
        )