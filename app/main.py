"""FastAPI backend.

Secrets live only on this side of the wire; the Streamlit frontend talks to
these endpoints and never sees an API key.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.ml.predictor import get_predictor
from app.pipeline import get_pipeline
from app.schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger("veritascheck")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="VeritasCheck API",
    version="1.0.0",
    description=(
        "Transparent news claim analysis: ML ensemble signal, News API evidence "
        "retrieval, claim-to-source mapping and a grounded AI explanation."
    ),
)

# The Streamlit frontend (8501) and the TruthLens Vite frontend (5173+, Vite
# picks the next free port if 5173 is taken) run on different local ports.
# The regex is deliberately restricted to loopback addresses only, so this
# never opens the API to a remote origin.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness plus a redacted view of which integrations are configured.

    Only booleans are exposed - never a key, a prefix or a length.
    """
    settings = get_settings()
    predictor = get_predictor()
    return {
        "status": "ok",
        "news_api_configured": settings.news_api_configured,
        "nvidia_configured": settings.nvidia_configured,
        "ml_models_available": predictor.available,
        "ml_metadata": predictor.metadata,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse a headline or article and return a fully-sourced result."""
    try:
        return get_pipeline().analyze(request.text, max_sources=request.max_sources)
    except Exception as exc:  # noqa: BLE001
        # Log server-side detail; return a generic message so nothing internal leaks.
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=(
                "The analysis could not be completed due to an internal error. "
                "Please try again."
            ),
        ) from exc


@app.post("/feedback")
def feedback(payload: dict) -> dict:
    """Accept the TruthLens interface's accuracy feedback.

    Feedback is logged server-side only. It is deliberately *not* fed back into
    the model or the verdict, so a user cannot influence another user's result.
    """
    logger.info(
        "User feedback: article_id=%s helpful=%s user_verdict=%s",
        payload.get("article_id"),
        payload.get("helpful"),
        payload.get("user_verdict"),
    )
    return {"success": True, "recorded": True}


@app.get("/")
def root() -> dict:
    return {
        "service": "VeritasCheck",
        "version": "1.0.0",
        "endpoints": {"health": "GET /health", "analyze": "POST /analyze"},
        "notice": (
            "News API results are related-source evidence. They are not "
            "automatically proof of truth or falsity."
        ),
    }
