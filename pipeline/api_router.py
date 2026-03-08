"""
pipeline/api_router.py
-----------------------
FastAPI router that exposes the data-collection pipeline over HTTP.

Endpoints
---------
POST /collect/run          → trigger a full pipeline run (async background task)
GET  /collect/status       → check if a run is in progress
GET  /collect/latest       → return metadata about the most recent processed file
GET  /collect/health       → simple health check for the collection subsystem
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from config.logger import get_logger
from config.settings import PROCESSED_DIR, TICKERS, LOOKBACK_DAYS
from data_collection.pipeline import DataCollectionPipeline

log = get_logger(__name__)
router = APIRouter(prefix="/collect", tags=["Data Collection"])


# ---------------------------------------------------------------------------
# Simple in-memory run tracker (replace with Redis/DB in production)
# ---------------------------------------------------------------------------

class _RunState:
    running: bool = False
    last_run_at: Optional[datetime] = None
    last_run_rows: int = 0
    last_error: Optional[str] = None


_state = _RunState()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CollectRequest(BaseModel):
    tickers:   Optional[List[str]] = None
    days_back: int = LOOKBACK_DAYS


class CollectStatusResponse(BaseModel):
    running:        bool
    last_run_at:    Optional[datetime]
    last_run_rows:  int
    last_error:     Optional[str]


class LatestDataResponse(BaseModel):
    file:       Optional[str]
    rows:       int
    tickers:    List[str]
    date_range: Dict[str, Any]


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_pipeline(tickers: Optional[List[str]], days_back: int):
    """Executed in a background thread by FastAPI."""
    _state.running = True
    _state.last_error = None
    try:
        pipeline = DataCollectionPipeline(tickers=tickers, days_back=days_back)
        df = pipeline.run()
        _state.last_run_rows = len(df)
        _state.last_run_at   = datetime.now(timezone.utc)
        log.info("Background pipeline run finished: %d rows", len(df))
    except Exception as exc:
        _state.last_error = str(exc)
        log.error("Background pipeline run failed: %s", exc)
    finally:
        _state.running = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/run", summary="Trigger a full data-collection pipeline run")
async def trigger_run(
    request:          CollectRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """
    Starts data collection for the requested tickers in a background task.
    Returns immediately — poll `/collect/status` to track progress.
    """
    if _state.running:
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already in progress. Try again later.",
        )

    tickers = request.tickers or TICKERS
    background_tasks.add_task(_run_pipeline, tickers, request.days_back)

    log.info("Pipeline run triggered for %s (days_back=%d)", tickers, request.days_back)
    return {
        "status":  "started",
        "tickers": ", ".join(tickers),
        "message": "Pipeline is running in the background.",
    }


@router.get("/status", response_model=CollectStatusResponse, summary="Check pipeline run status")
async def get_status() -> CollectStatusResponse:
    return CollectStatusResponse(
        running       = _state.running,
        last_run_at   = _state.last_run_at,
        last_run_rows = _state.last_run_rows,
        last_error    = _state.last_error,
    )


@router.get("/latest", response_model=LatestDataResponse, summary="Metadata about the latest processed file")
async def get_latest() -> LatestDataResponse:
    """
    Inspects the most recent parquet file in data/processed/ and returns
    summary metadata without loading the full dataset.
    """
    pattern = os.path.join(PROCESSED_DIR, "raw_aligned_*.parquet")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise HTTPException(status_code=404, detail="No processed data files found yet.")

    import pandas as pd
    latest = files[-1]
    df = pd.read_parquet(latest, columns=["ticker", "date"])

    return LatestDataResponse(
        file       = os.path.basename(latest),
        rows       = len(df),
        tickers    = sorted(df["ticker"].unique().tolist()),
        date_range = {
            "start": str(df["date"].min()),
            "end":   str(df["date"].max()),
        },
    )


@router.get("/health", summary="Health check for the data-collection subsystem")
async def health() -> Dict[str, str]:
    checks = {
        "newsapi_key_set":    "ok" if os.getenv("NEWSAPI_KEY") else "missing",
        "processed_dir":      "ok" if os.path.isdir(PROCESSED_DIR) else "missing",
    }
    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, **checks}