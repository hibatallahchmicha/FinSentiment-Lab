"""

OLS regression analysis: sentiment features → price returns.

What this answers
-----------------
- "Does negative news predict price drops?" — quantified with a coefficient
- How much return variance does sentiment explain? (R²)
- Which sentiment features are statistically significant predictors?
- Does the relationship hold after controlling for price momentum?

Model specifications run
------------------------
Model 1 — Univariate baseline
  forward_return_1d ~ mean_score

Model 2 — Sentiment only
  forward_return_1d ~ mean_score + sentiment_roll_7d + sentiment_zscore
                    + sentiment_cross_7_30 + news_day

Model 3 — Sentiment + price controls
  forward_return_1d ~ [sentiment features] + rsi_14d + return_5d
                    + vol_zscore_21d + price_vs_sma_10d

Model 4 — Volatility target
  forward_vol_5d ~ mean_score + sentiment_zscore + sentiment_roll_7d
                 + vol_zscore_21d + atr_14d_pct

Outputs
-------
RegressionResult per model per ticker
  model_name  : descriptive label
  ticker      : ticker or "POOLED"
  coef_table  : DataFrame with coef, std_err, t_stat, p_value, significant
  r_squared   : in-sample R²
  adj_r_squared
  f_statistic : overall model F-stat
  f_pvalue    : p-value of F-stat (is the model significant at all?)
  n_obs       : number of observations
  interpretation : one-sentence plain-English summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "univariate_baseline": {
        "target":   "forward_return_1d",
        "features": ["mean_score"],
        "label":    "Univariate: mean_score → next-day return",
    },
    "sentiment_only": {
        "target":   "forward_return_1d",
        "features": [
            "mean_score", "sentiment_roll_7d", "sentiment_zscore",
            "sentiment_cross_7_30", "news_day",
        ],
        "label": "Sentiment only → next-day return",
    },
    "sentiment_with_controls": {
        "target":   "forward_return_1d",
        "features": [
            "mean_score", "sentiment_roll_7d", "sentiment_zscore",
            "sentiment_cross_7_30", "news_day",
            "rsi_14d", "return_5d", "vol_zscore_21d", "price_vs_sma_10d",
        ],
        "label": "Sentiment + price controls → next-day return",
    },
    "volatility_target": {
        "target":   "forward_vol_5d",
        "features": [
            "mean_score", "sentiment_zscore", "sentiment_roll_7d",
            "vol_zscore_21d", "atr_14d_pct",
        ],
        "label": "Sentiment → next-week volatility",
    },
    "return_5d_target": {
        "target":   "forward_return_5d",
        "features": [
            "mean_score", "sentiment_roll_7d", "sentiment_roll_14d",
            "sentiment_cross_7_30", "news_day",
            "return_5d", "momentum_5_21",
        ],
        "label": "Sentiment → 5-day forward return",
    },
}


@dataclass
class RegressionResult:
    model_name:     str
    ticker:         str
    target:         str
    coef_table:     pd.DataFrame       # coef | std_err | t_stat | p_value | significant
    r_squared:      float
    adj_r_squared:  float
    f_statistic:    float
    f_pvalue:       float
    n_obs:          int
    interpretation: str = ""


class OLSAnalyzer:
    """
    Runs multiple OLS regression specifications for each ticker.

    Parameters
    ----------
    model_specs : dict of model specifications (defaults to MODULE_SPECS above)
    min_obs     : minimum observations required to fit a model
    """

    def __init__(
        self,
        model_specs: Optional[Dict] = None,
        min_obs:     int            = 15,  # Lowered from 40 to handle sparse data
    ):
        self.model_specs = model_specs or MODEL_SPECS
        self.min_obs     = min_obs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_per_ticker(self, df: pd.DataFrame) -> Dict[str, List[RegressionResult]]:
        """Fit all model specifications for each ticker separately."""
        results = {}
        for ticker, group in df.groupby("ticker"):
            log.info("OLS regression: %s (%d rows)", ticker, len(group))
            results[ticker] = self._fit_all_models(group, label=ticker)
        return results

    def run_pooled(self, df: pd.DataFrame) -> List[RegressionResult]:
        """Fit all model specifications on the pooled dataset."""
        log.info("OLS regression: POOLED (%d rows)", len(df))
        return self._fit_all_models(df, label="POOLED")

    def run_all(self, df: pd.DataFrame) -> Dict[str, List[RegressionResult]]:
        """Run both per-ticker and pooled. Returns combined dict."""
        results = self.run_per_ticker(df)
        results["POOLED"] = self.run_pooled(df)
        return results

    def summary_table(self, results: Dict[str, List[RegressionResult]]) -> pd.DataFrame:
        """
        Flatten all regression results into a tidy model-level summary.

        Columns: ticker | model | target | r_squared | adj_r2 | f_stat | f_pvalue | n_obs
        """
        rows = []
        for ticker, result_list in results.items():
            for res in result_list:
                rows.append({
                    "ticker":       res.ticker,
                    "model":        res.model_name,
                    "target":       res.target,
                    "r_squared":    round(res.r_squared,     4),
                    "adj_r2":       round(res.adj_r_squared, 4),
                    "f_stat":       round(res.f_statistic,   3),
                    "f_pvalue":     round(res.f_pvalue,      4),
                    "n_obs":        res.n_obs,
                    "model_sig":    res.f_pvalue < 0.05,
                    "interpretation": res.interpretation,
                })
        return pd.DataFrame(rows).sort_values(["ticker", "model"])

    def coef_table(self, results: Dict[str, List[RegressionResult]]) -> pd.DataFrame:
        """
        Flatten all coefficient tables into one tidy DataFrame.

        Columns: ticker | model | feature | coef | std_err | t_stat | p_value | significant
        """
        rows = []
        for ticker, result_list in results.items():
            for res in result_list:
                ct = res.coef_table.copy()
                ct["ticker"] = res.ticker
                ct["model"]  = res.model_name
                ct["target"] = res.target
                ct = ct.reset_index().rename(columns={"index": "feature"})
                rows.append(ct)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit_all_models(
        self, df: pd.DataFrame, label: str
    ) -> List[RegressionResult]:
        results = []
        for model_name, spec in self.model_specs.items():
            result = self._fit_model(
                df         = df,
                model_name = model_name,
                target     = spec["target"],
                features   = spec["features"],
                label_str  = spec["label"],
                ticker     = label,
            )
            if result is not None:
                results.append(result)
        return results

    def _fit_model(
        self,
        df:         pd.DataFrame,
        model_name: str,
        target:     str,
        features:   List[str],
        label_str:  str,
        ticker:     str,
    ) -> Optional[RegressionResult]:
        """Fit one OLS model and return a RegressionResult."""
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError("statsmodels required. Run: pip install statsmodels")

        # Select available features only
        avail_feats = [f for f in features if f in df.columns]
        if target not in df.columns or not avail_feats:
            log.debug("Skipping %s for %s — missing columns.", model_name, ticker)
            return None

        sub = df[[target] + avail_feats].apply(pd.to_numeric, errors="coerce").dropna()

        if len(sub) < self.min_obs:
            log.warning(
                "%s | %s: only %d obs — skipping.", ticker, model_name, len(sub)
            )
            return None

        X = sm.add_constant(sub[avail_feats])
        y = sub[target]

        try:
            model  = sm.OLS(y, X).fit()
        except Exception as exc:
            log.error("%s | %s failed to fit: %s", ticker, model_name, exc)
            return None

        # Build coefficient table
        coef_table = pd.DataFrame({
            "coef":        model.params,
            "std_err":     model.bse,
            "t_stat":      model.tvalues,
            "p_value":     model.pvalues,
            "significant": model.pvalues < 0.05,
        }).drop(index="const", errors="ignore")

        # Plain-English interpretation
        interpretation = self._interpret(
            model_name  = model_name,
            ticker      = ticker,
            coef_table  = coef_table,
            r_squared   = model.rsquared,
            f_pvalue    = model.f_pvalue,
            target      = target,
            avail_feats = avail_feats,
        )

        log.info(
            "  %s | %s | R²=%.4f | F-p=%.4f | n=%d",
            ticker, model_name, model.rsquared, model.f_pvalue, len(sub),
        )

        return RegressionResult(
            model_name    = model_name,
            ticker        = ticker,
            target        = target,
            coef_table    = coef_table.round(6),
            r_squared     = float(model.rsquared),
            adj_r_squared = float(model.rsquared_adj),
            f_statistic   = float(model.fvalue),
            f_pvalue      = float(model.f_pvalue),
            n_obs         = int(len(sub)),
            interpretation= interpretation,
        )

    # ------------------------------------------------------------------
    # Interpretation helper
    # ------------------------------------------------------------------

    def _interpret(
        self,
        model_name:  str,
        ticker:      str,
        coef_table:  pd.DataFrame,
        r_squared:   float,
        f_pvalue:    float,
        target:      str,
        avail_feats: List[str],
    ) -> str:
        """Generate a one-sentence plain-English interpretation."""
        model_sig = f_pvalue < 0.05
        r2_pct    = round(r_squared * 100, 1)

        sig_feats = coef_table[coef_table["significant"]].index.tolist()
        sig_feats = [f for f in sig_feats if f != "const"]

        if not model_sig:
            return (
                f"{ticker} {model_name}: Model not significant (F-p={f_pvalue:.3f}) — "
                f"sentiment features do not reliably predict {target} for this ticker."
            )

        if "mean_score" in coef_table.index and coef_table.loc["mean_score", "significant"]:
            coef = coef_table.loc["mean_score", "coef"]
            direction = "positive" if coef > 0 else "negative"
            return (
                f"{ticker}: mean_score has a {direction} significant effect on {target} "
                f"(β={coef:.4f}), model explains {r2_pct}% of variance (R²={r_squared:.4f})."
            )

        if sig_feats:
            return (
                f"{ticker}: {', '.join(sig_feats[:3])} are significant predictors of {target}. "
                f"Model R²={r_squared:.4f} ({r2_pct}% variance explained)."
            )

        return (
            f"{ticker}: Model significant overall (F-p={f_pvalue:.3f}, R²={r_squared:.4f}) "
            f"but no individual feature reached p<0.05."
        )