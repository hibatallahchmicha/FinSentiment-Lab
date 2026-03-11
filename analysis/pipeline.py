"""
analysis/pipeline.py
---------------------
Orchestrates the full analysis step:

  1. Load the feature matrix from feature_engineering/pipeline.py
  2. Run CorrelationAnalyzer  (Pearson + Spearman)
  3. Run GrangerAnalyzer      (causality tests)
  4. Run OLSAnalyzer          (regression models)
  5. Assemble an AnalysisReport with all results
  6. Save report to data/processed/analysis_<date>.json
  7. Print human-readable findings to console

CLI usage
---------
    python -m analysis.pipeline
    python -m analysis.pipeline --input data/processed/features_20240701.parquet
    python -m analysis.pipeline --no-granger    # skip Granger (faster)
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.logger import get_logger
from config.settings import PROCESSED_DIR
from analysis.correlation import CorrelationAnalyzer
from analysis.granger     import GrangerAnalyzer
from analysis.regression  import OLSAnalyzer

log = get_logger(__name__)


class AnalysisPipeline:
    """
    End-to-end analysis orchestrator.

    Parameters
    ----------
    run_granger : set False to skip Granger tests (faster, needs statsmodels)
    max_lag     : maximum lag for Granger tests
    """

    def __init__(self, run_granger: bool = True, max_lag: int = 5):
        self.run_granger   = run_granger
        self.correlation   = CorrelationAnalyzer()
        self.granger       = GrangerAnalyzer(max_lag=max_lag) if run_granger else None
        self.ols           = OLSAnalyzer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, input_parquet: Optional[str] = None) -> dict:
        """
        Execute the full analysis pipeline.

        Returns
        -------
        dict with keys:
          correlation_summary  : pd.DataFrame
          granger_summary      : pd.DataFrame (or empty)
          regression_summary   : pd.DataFrame
          coef_table           : pd.DataFrame
          findings             : list of plain-English finding strings
        """
        log.info("=" * 60)
        log.info("Starting AnalysisPipeline")
        log.info("=" * 60)

        df = self._load(input_parquet)
        if df.empty:
            log.error("No feature data found. Run feature_engineering.pipeline first.")
            return {}

        log.info("Feature matrix: %s", df.shape)

        # ── 1. Correlation ────────────────────────────────────────────
        log.info("Running correlation analysis…")
        corr_results = self.correlation.run_all(df)
        corr_summary = self.correlation.summary_table(corr_results)

        # ── 2. Granger causality ──────────────────────────────────────
        granger_summary = pd.DataFrame()
        if self.run_granger and self.granger:
            log.info("Running Granger causality tests…")
            granger_results = self.granger.run_all(df)
            granger_summary = self.granger.summary_table(granger_results)
        else:
            log.info("Granger tests skipped.")

        # ── 3. OLS regression ─────────────────────────────────────────
        log.info("Running OLS regression models…")
        ols_results     = self.ols.run_all(df)
        reg_summary     = self.ols.summary_table(ols_results)
        coef_table      = self.ols.coef_table(ols_results)

        # ── 4. Compile findings ───────────────────────────────────────
        findings = self._compile_findings(corr_summary, granger_summary, reg_summary)

        # ── 5. Save ───────────────────────────────────────────────────
        self._save(corr_summary, granger_summary, reg_summary, findings)

        # ── 6. Print to console ───────────────────────────────────────
        self._print_findings(findings, corr_summary, granger_summary, reg_summary)

        return {
            "correlation_summary": corr_summary,
            "granger_summary":     granger_summary,
            "regression_summary":  reg_summary,
            "coef_table":          coef_table,
            "findings":            findings,
        }

    # ------------------------------------------------------------------
    # Findings compiler — plain English answers to research questions
    # ------------------------------------------------------------------

    def _compile_findings(
        self,
        corr:    pd.DataFrame,
        granger: pd.DataFrame,
        ols:     pd.DataFrame,
    ) -> list[str]:
        findings = []

        # ── Q1: Does negative news predict price drops? ────────────────
        findings.append("── Q1: Does negative news predict price drops? ──")
        if not corr.empty:
            q1 = corr[
                (corr["target"] == "forward_return_1d") &
                (corr["feature"] == "mean_score")
            ]
            for _, row in q1.iterrows():
                sig = "✓ SIGNIFICANT" if row.get("significant") else "✗ not significant"
                findings.append(
                    f"  {row['ticker']}: Pearson r={row['pearson_r']:.3f}, "
                    f"p={row['p_value']:.3f}  {sig}"
                )

        # ── Q2: Is sentiment predictive of volatility? ─────────────────
        findings.append("\n── Q2: Is sentiment predictive of volatility? ──")
        if not corr.empty:
            q2 = corr[
                (corr["target"] == "forward_vol_5d") &
                (corr["feature"] == "mean_score")
            ]
            for _, row in q2.iterrows():
                sig = "✓ SIGNIFICANT" if row.get("significant") else "✗ not significant"
                findings.append(
                    f"  {row['ticker']}: Pearson r={row['pearson_r']:.3f}, "
                    f"p={row['p_value']:.3f}  {sig}"
                )

        # ── Q3: Granger causality verdict ──────────────────────────────
        if not granger.empty:
            findings.append("\n── Q3: Does sentiment Granger-cause returns? ──")
            key = granger[
                (granger["cause"] == "mean_score") &
                (granger["effect"].str.contains("return"))
            ]
            for _, row in key.iterrows():
                sig = "✓ YES" if row["significant"] else "✗ NO"
                findings.append(
                    f"  {row['ticker']}: {row['cause']} → {row['effect']} | "
                    f"best_lag={row['best_lag']}d | p={row['min_p_value']:.4f}  {sig}"
                )

        # ── Q4: Best regression model ──────────────────────────────────
        if not ols.empty:
            findings.append("\n── Q4: How much variance does sentiment explain? ──")
            top = ols[ols["model_sig"]].sort_values("adj_r2", ascending=False)
            for _, row in top.head(6).iterrows():
                findings.append(
                    f"  {row['ticker']} | {row['model']}: "
                    f"adj-R²={row['adj_r2']:.4f} | F-p={row['f_pvalue']:.4f}"
                )

        # ── Q5: Most correlated feature ────────────────────────────────
        if not corr.empty:
            findings.append("\n── Q5: Most correlated sentiment feature per ticker ──")
            for ticker in corr["ticker"].unique():
                subset = corr[
                    (corr["ticker"] == ticker) &
                    (corr["target"] == "forward_return_1d")
                ].dropna(subset=["pearson_r"])
                if subset.empty:
                    continue
                best = subset.loc[subset["pearson_r"].abs().idxmax()]
                findings.append(
                    f"  {ticker}: {best['feature']} "
                    f"(r={best['pearson_r']:.3f}, p={best['p_value']:.3f})"
                )

        return findings

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _print_findings(self, findings, corr, granger, ols):
        sep = "═" * 60
        print(f"\n{sep}")
        print("  FINSENTIMENT LAB · ANALYSIS REPORT")
        print(sep)
        for line in findings:
            print(line)
        print(f"\n{sep}")
        print(f"  Correlation rows : {len(corr)}")
        print(f"  Granger rows     : {len(granger)}")
        print(f"  Regression models: {len(ols)}")
        print(sep)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, corr, granger, ols, findings):
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        base     = os.path.join(PROCESSED_DIR, f"analysis_{date_str}")

        corr.to_parquet(f"{base}_correlation.parquet",  index=False)
        ols.to_parquet( f"{base}_regression.parquet",   index=False)

        if not granger.empty:
            granger.to_parquet(f"{base}_granger.parquet", index=False)

        with open(f"{base}_findings.json", "w") as f:
            json.dump(findings, f, indent=2)

        log.info("Analysis results saved to %s_*.parquet / .json", base)

    def _load(self, path: Optional[str]) -> pd.DataFrame:
        if path and os.path.exists(path):
            return pd.read_parquet(path)
        pattern = os.path.join(PROCESSED_DIR, "features_*.parquet")
        files   = sorted(glob.glob(pattern))
        if not files:
            return pd.DataFrame()
        log.info("Auto-loading: %s", os.path.basename(files[-1]))
        return pd.read_parquet(files[-1])


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Any, Dict as FDict

router    = APIRouter(prefix="/analysis", tags=["Analysis"])
_state    = type("S", (), {"running": False, "last_run_at": None, "error": None})()
_cache: FDict[str, Any] = {}


def _run_bg(input_parquet, run_granger, max_lag):
    _state.running = True
    _state.error   = None
    try:
        pipeline = AnalysisPipeline(run_granger=run_granger, max_lag=max_lag)
        results  = pipeline.run(input_parquet=input_parquet)
        # Cache serialisable summaries
        _cache["correlation"] = results["correlation_summary"].to_dict(orient="records") if not results.get("correlation_summary", pd.DataFrame()).empty else []
        _cache["granger"]     = results["granger_summary"].to_dict(orient="records")     if not results.get("granger_summary",     pd.DataFrame()).empty else []
        _cache["regression"]  = results["regression_summary"].to_dict(orient="records")  if not results.get("regression_summary",  pd.DataFrame()).empty else []
        _cache["findings"]    = results.get("findings", [])
        _state.last_run_at    = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        _state.error = str(exc)
        log.error("Analysis pipeline failed: %s", exc)
    finally:
        _state.running = False


@router.post("/run")
async def run_analysis(
    background_tasks: BackgroundTasks,
    input_parquet: Optional[str] = None,
    run_granger:   bool          = True,
    max_lag:       int           = 5,
):
    if _state.running:
        raise HTTPException(409, "Analysis already running.")
    background_tasks.add_task(_run_bg, input_parquet, run_granger, max_lag)
    return {"status": "started"}


@router.get("/status")
async def get_status():
    return {"running": _state.running, "last_run_at": _state.last_run_at, "error": _state.error}


@router.get("/findings")
async def get_findings():
    if "findings" not in _cache:
        raise HTTPException(404, "Run the analysis pipeline first.")
    return {"findings": _cache["findings"]}


@router.get("/correlation")
async def get_correlation(ticker: Optional[str] = None):
    if "correlation" not in _cache:
        raise HTTPException(404, "Run the analysis pipeline first.")
    data = _cache["correlation"]
    if ticker:
        data = [r for r in data if r["ticker"] == ticker.upper()]
    return data


@router.get("/regression")
async def get_regression(ticker: Optional[str] = None):
    if "regression" not in _cache:
        raise HTTPException(404, "Run the analysis pipeline first.")
    data = _cache["regression"]
    if ticker:
        data = [r for r in data if r["ticker"] == ticker.upper()]
    return data


@router.get("/granger")
async def get_granger(ticker: Optional[str] = None):
    if "granger" not in _cache:
        raise HTTPException(404, "No Granger results.")
    data = _cache["granger"]
    if ticker:
        data = [r for r in data if r["ticker"] == ticker.upper()]
    return data


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinSentiment Lab — Analysis Module")
    parser.add_argument("--input",       default=None,        help="Path to features parquet")
    parser.add_argument("--no-granger",  action="store_true", help="Skip Granger causality tests")
    parser.add_argument("--max-lag",     type=int, default=5, help="Max lag for Granger tests")
    args = parser.parse_args()

    pipeline = AnalysisPipeline(
        run_granger = not args.no_granger,
        max_lag     = args.max_lag,
    )
    pipeline.run(input_parquet=args.input)