"""
Granger causality tests between sentiment features and price returns.

What this answers
-----------------
"Does sentiment Granger-cause returns?"

Granger causality tests whether past values of X (sentiment) improve the
prediction of Y (returns) beyond what past values of Y alone can predict.

A significant result (p < 0.05) at lag k means:
  "Sentiment k days ago carries incremental predictive information
   about today's return, after controlling for return autocorrelation."

This is NOT the same as true economic causality — it's predictive causality.
It directly answers: "Do past news scores help predict future price moves?"

Test details
------------
We use statsmodels `grangercausalitytests` which runs:
  1. A restricted VAR:   y_t = f(y_{t-1}, ..., y_{t-k})
  2. An unrestricted VAR: y_t = f(y_{t-1}, ..., y_{t-k}, x_{t-1}, ..., x_{t-k})
  3. F-test: does adding X significantly reduce residual variance?

We test lags 1 through max_lag and report results for each.

Outputs
-------
GrangerResult (per ticker × feature × target combination)
  ticker        : ticker or "POOLED"
  cause         : the X variable (e.g., "mean_score")
  effect        : the Y variable (e.g., "forward_return_1d")
  results_by_lag: dict {lag: {"f_stat": .., "p_value": .., "significant": ..}}
  min_p_value   : smallest p-value across all tested lags
  best_lag      : lag with lowest p-value
  verdict       : "granger_causes" | "no_granger_causality" | "insufficient_data"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.logger import get_logger

log = get_logger(__name__)

# Default pairs to test: (cause → effect)
DEFAULT_PAIRS = [
    ("mean_score",           "daily_return"),
    ("mean_score",           "forward_return_1d"),
    ("sentiment_roll_7d",    "daily_return"),
    ("sentiment_roll_7d",    "forward_return_1d"),
    ("sentiment_zscore",     "daily_return"),
    ("sentiment_cross_7_30", "daily_return"),
    # Does sentiment predict volatility?
    ("mean_score",           "forward_vol_5d"),
    ("sentiment_zscore",     "forward_vol_5d"),
]

SIGNIFICANCE_LEVEL = 0.05


@dataclass
class GrangerResult:
    ticker:          str
    cause:           str
    effect:          str
    results_by_lag:  Dict[int, Dict]  = field(default_factory=dict)
    min_p_value:     float            = 1.0
    best_lag:        int              = 0
    verdict:         str              = "insufficient_data"
    n_obs:           int              = 0


class GrangerAnalyzer:
    """
    Runs Granger causality tests for sentiment → return pairs.

    Parameters
    ----------
    max_lag   : maximum lag to test (default 5 trading days)
    pairs     : list of (cause, effect) column name tuples
    min_obs   : minimum observations required (Granger needs at least 3×max_lag)
    """

    def __init__(
        self,
        max_lag: int                         = 5,
        pairs:   Optional[List[tuple]]       = None,
        min_obs: int                         = 25,  # Lowered from 60 to handle sparse data
    ):
        self.max_lag = max_lag
        self.pairs   = pairs or DEFAULT_PAIRS
        self.min_obs = min_obs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_per_ticker(self, df: pd.DataFrame) -> Dict[str, List[GrangerResult]]:
        """Run all pairs for each ticker separately."""
        results = {}
        for ticker, group in df.groupby("ticker"):
            log.info("Granger tests: %s (%d rows)", ticker, len(group))
            results[ticker] = self._test_all_pairs(group, label=ticker)
        return results

    def run_pooled(self, df: pd.DataFrame) -> List[GrangerResult]:
        """Run all pairs on the pooled dataset."""
        log.info("Granger tests: POOLED (%d rows)", len(df))
        return self._test_all_pairs(df, label="POOLED")

    def run_all(self, df: pd.DataFrame) -> Dict[str, List[GrangerResult]]:
        """Run both per-ticker and pooled. Returns combined dict."""
        results = self.run_per_ticker(df)
        results["POOLED"] = self.run_pooled(df)
        return results

    def summary_table(self, results: Dict[str, List[GrangerResult]]) -> pd.DataFrame:
        """
        Flatten all GrangerResult objects into a tidy summary DataFrame.

        Columns: ticker | cause | effect | best_lag | min_p_value | verdict | significant
        """
        rows = []
        for ticker, result_list in results.items():
            for res in result_list:
                rows.append({
                    "ticker":      res.ticker,
                    "cause":       res.cause,
                    "effect":      res.effect,
                    "best_lag":    res.best_lag,
                    "min_p_value": round(res.min_p_value, 4),
                    "verdict":     res.verdict,
                    "significant": res.verdict == "granger_causes",
                    "n_obs":       res.n_obs,
                })
        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = df_out.sort_values("min_p_value")
        return df_out

    def significant_pairs(
        self, results: Dict[str, List[GrangerResult]]
    ) -> pd.DataFrame:
        """Return only pairs where Granger causality is significant (p < 0.05)."""
        full = self.summary_table(results)
        return full[full["significant"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Core test logic
    # ------------------------------------------------------------------

    def _test_all_pairs(
        self, df: pd.DataFrame, label: str
    ) -> List[GrangerResult]:
        """Test every (cause, effect) pair in self.pairs."""
        results = []
        for cause, effect in self.pairs:
            if cause not in df.columns or effect not in df.columns:
                log.debug("Skipping %s → %s: column not found.", cause, effect)
                continue
            result = self._test_pair(df, cause=cause, effect=effect, label=label)
            results.append(result)
        return results

    def _test_pair(
        self,
        df:     pd.DataFrame,
        cause:  str,
        effect: str,
        label:  str,
    ) -> GrangerResult:
        """
        Run statsmodels grangercausalitytests for one (cause, effect) pair.
        Returns a GrangerResult with results for each lag.
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            raise ImportError(
                "statsmodels is required for Granger causality tests. "
                "Run: pip install statsmodels"
            )

        pair = df[[effect, cause]].dropna()  # statsmodels expects [y, x] order

        if len(pair) < self.min_obs:
            log.warning(
                "%s | %s → %s: only %d obs (need %d) — skipping.",
                label, cause, effect, len(pair), self.min_obs,
            )
            return GrangerResult(
                ticker=label, cause=cause, effect=effect,
                verdict="insufficient_data", n_obs=len(pair),
            )

        try:
            # verbose=False suppresses per-lag console output
            raw = grangercausalitytests(pair.values, maxlag=self.max_lag, verbose=False)
        except Exception as exc:
            log.warning("%s | %s → %s failed: %s", label, cause, effect, exc)
            return GrangerResult(
                ticker=label, cause=cause, effect=effect,
                verdict="insufficient_data", n_obs=len(pair),
            )

        # Parse results: raw[lag] = ([F-test, chi2, lrt, param], [ols_res])
        results_by_lag: Dict[int, Dict] = {}
        for lag in range(1, self.max_lag + 1):
            f_test = raw[lag][0]["ssr_ftest"]   # (F-stat, p-value, df_denom, df_num)
            results_by_lag[lag] = {
                "f_stat":      round(float(f_test[0]), 4),
                "p_value":     round(float(f_test[1]), 6),
                "significant": bool(f_test[1] < SIGNIFICANCE_LEVEL),
            }

        p_values  = {lag: v["p_value"] for lag, v in results_by_lag.items()}
        best_lag  = min(p_values, key=p_values.get)
        min_p     = p_values[best_lag]
        verdict   = "granger_causes" if min_p < SIGNIFICANCE_LEVEL else "no_granger_causality"

        log.info(
            "  %s | %s → %s | best_lag=%d | min_p=%.4f | %s",
            label, cause, effect, best_lag, min_p,
            "✓ SIGNIFICANT" if verdict == "granger_causes" else "✗ not significant",
        )

        return GrangerResult(
            ticker         = label,
            cause          = cause,
            effect         = effect,
            results_by_lag = results_by_lag,
            min_p_value    = min_p,
            best_lag       = best_lag,
            verdict        = verdict,
            n_obs          = len(pair),
        )