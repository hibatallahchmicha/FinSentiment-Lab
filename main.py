"""
main.py
-------
FastAPI application entry-point for the FinSentiment platform.

Run locally:
    uvicorn main:app --reload --port 8000

Production:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
"""
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.logger import get_logger
from pipeline.api_router import router as collect_router

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    log.info("FinSentiment API starting up…")
    yield
    log.info("FinSentiment API shutting down.")


app = FastAPI(
    title       = "FinSentiment API",
    description = "LLM-powered financial news sentiment analysis platform",
    version     = "0.1.0",
    lifespan    = lifespan,
)

# Allow the React frontend (running on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Register routers
app.include_router(collect_router)

# Future routers (uncomment as modules are built):
# from sentiment_engine.api_router import router as sentiment_router
# app.include_router(sentiment_router)

# from analysis.api_router import router as analysis_router
# app.include_router(analysis_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "app":     "FinSentiment",
        "version": "0.1.0",
        "docs":    "/docs",
    }