"""Step 9b — the VeriTruth FastAPI application.

Endpoints::

    GET  /health        readiness + which subsystems are live
    POST /predict       calibrated verdict + trust score
    POST /explain       top-k SHAP/LIME token weights
    POST /investigate   full agentic, cited investigation
    POST /feedback      store a human correction in SQLite

Every heavyweight object (model, vector store, agent graph) is warmed once in
the lifespan handler, never per request. Every handler is wrapped so that an
unexpected failure returns clean JSON rather than an HTML stack trace.

Run::

    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    InvestigateResponse,
    PredictResponse,
    TextRequest,
)
from src.config import FEEDBACK_DB_PATH, ensure_dirs, get_env, get_logger, set_seeds

LOG = get_logger("veritruth.api")

VERSION = "1.0.0"

_DB_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        REAL    NOT NULL,
    text              TEXT    NOT NULL,
    predicted_verdict TEXT    NOT NULL,
    user_verdict      TEXT    NOT NULL,
    trust_score       REAL    NOT NULL,
    comment           TEXT    NOT NULL DEFAULT ''
);
"""


def _connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(FEEDBACK_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the feedback table if it does not exist. Never raises."""
    try:
        with _DB_LOCK, _connect() as conn:
            conn.executescript(_SCHEMA)
        LOG.info("Feedback DB ready at %s", FEEDBACK_DB_PATH)
    except Exception as exc:
        LOG.error("Could not initialise feedback DB (%s).", exc)


def save_feedback(payload: FeedbackRequest) -> tuple[int, int]:
    """Insert one feedback row; returns ``(row_id, total_rows)``."""
    with _DB_LOCK, _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO feedback (created_at, text, predicted_verdict, user_verdict,"
            " trust_score, comment) VALUES (?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                payload.text,
                payload.predicted_verdict,
                payload.user_verdict,
                float(payload.trust_score),
                payload.comment,
            ),
        )
        row_id = int(cursor.lastrowid or 0)
        total = int(conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
    return row_id, total


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm every expensive singleton exactly once at startup."""
    set_seeds()
    ensure_dirs()
    init_db()

    app.state.model_name = "unavailable"
    app.state.model_loaded = False
    app.state.evidence_chunks = 0

    try:
        from src.models.predict import get_predictor

        predictor = get_predictor()
        app.state.model_name = predictor.name
        app.state.model_loaded = not predictor.degraded
        LOG.info("Model '%s' warm.", predictor.name)
    except Exception as exc:
        LOG.error("Model warm-up failed (%s); /predict will degrade.", exc)

    try:
        from src.rag.retriever import evidence_count

        app.state.evidence_chunks = int(evidence_count())
        LOG.info("Evidence store holds %d chunks.", app.state.evidence_chunks)
    except Exception as exc:
        LOG.error("Evidence warm-up failed (%s); retrieval will degrade.", exc)

    try:
        from src.agent.graph import build_graph

        build_graph()
        LOG.info("Agent graph compiled.")
    except Exception as exc:
        LOG.warning("Agent graph not compiled at startup (%s).", exc)

    yield
    LOG.info("VeriTruth API shutting down.")


app = FastAPI(
    title="VeriTruth API",
    version=VERSION,
    description="Agentic, explainable fake-news investigation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in get_env("CORS_ORIGINS", "*").split(",") if o] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------- error handling
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "path": request.url.path,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail, "path": request.url.path},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    LOG.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "detail": type(exc).__name__,
            "path": request.url.path,
        },
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    LOG.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ------------------------------------------------------------------ endpoints
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Readiness probe reporting which subsystems are actually live."""
    model_loaded = bool(getattr(app.state, "model_loaded", False))
    chunks = int(getattr(app.state, "evidence_chunks", 0))
    return HealthResponse(
        status="ok" if model_loaded and chunks > 0 else "degraded",
        version=VERSION,
        model=str(getattr(app.state, "model_name", "unavailable")),
        model_loaded=model_loaded,
        evidence_chunks=chunks,
        llm_configured=bool(get_env("GEMINI_API_KEY")),
        factcheck_api_configured=bool(get_env("GOOGLE_FACTCHECK_API_KEY")),
    )


@app.post("/predict", response_model=PredictResponse, tags=["model"])
async def predict_endpoint(payload: TextRequest) -> PredictResponse:
    """Classify text as Real / Suspicious / Fake with a calibrated trust score."""
    import anyio

    from src.models.predict import predict

    try:
        result: dict[str, Any] = await anyio.to_thread.run_sync(predict, payload.text)
    except Exception as exc:
        LOG.error("/predict failed (%s); returning neutral result.", exc)
        result = {
            "verdict": "Suspicious",
            "band": "Suspicious",
            "trust_score": 50.0,
            "probability_real": 0.5,
            "model": "unavailable",
            "degraded": True,
        }
    return PredictResponse.model_validate(result)


@app.post("/explain", response_model=ExplainResponse, tags=["model"])
async def explain_endpoint(payload: ExplainRequest) -> ExplainResponse:
    """Return the top-k tokens driving the prediction (negative = toward Fake)."""
    import anyio

    from src.explain.shap_explainer import explain_with_meta

    try:
        result = await anyio.to_thread.run_sync(explain_with_meta, payload.text, payload.top_k)
    except Exception as exc:
        LOG.error("/explain failed (%s).", exc)
        result = {"tokens": [], "backend": "none", "degraded": True}
    return ExplainResponse.model_validate(result)


@app.post("/investigate", response_model=InvestigateResponse, tags=["agent"])
async def investigate_endpoint(payload: TextRequest) -> InvestigateResponse:
    """Run the full agentic investigation: classify, decompose, verify, cite."""
    from src.agent.graph import ainvestigate

    result = await ainvestigate(payload.text)
    return InvestigateResponse.model_validate(result)


@app.post("/feedback", response_model=FeedbackResponse, tags=["ops"])
async def feedback_endpoint(payload: FeedbackRequest) -> FeedbackResponse:
    """Persist a human correction so the model can be retrained later."""
    import anyio

    try:
        row_id, total = await anyio.to_thread.run_sync(save_feedback, payload)
        return FeedbackResponse(id=row_id, stored=True, total=total)
    except Exception as exc:
        LOG.error("/feedback storage failed (%s).", exc)
        return FeedbackResponse(id=0, stored=False, total=0)


@app.get("/", tags=["ops"])
async def root() -> dict[str, Any]:
    """Tiny index so a bare GET / is useful rather than a 404."""
    return {
        "name": "VeriTruth API",
        "version": VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/explain", "/investigate", "/feedback"],
    }
