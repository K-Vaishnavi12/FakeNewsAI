"""Flask API for FakeNewsAI.

Pipeline: client -> Flask -> LLMSearchAgent -> [ML classifier + news search]
-> synthesis -> JSON report.

Security posture (see README "Security notes"):
  * CORS is an explicit allow-list, never ``*``.
  * Every text input is length-capped before it reaches the model.
  * Every numeric parameter is validated and clamped.
  * Per-IP rate limits guard the expensive endpoints.
  * The training endpoint is opt-in and path-sandboxed.
  * There is currently NO authentication -- see README before exposing this
    service beyond localhost.
"""

from typing import Optional, Tuple

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .agent import LLMSearchAgent
from .config import settings
from .logging_config import get_logger, setup_logging
from .ml_model import (
    classify_with_probabilities,
    get_model_info,
    is_model_loaded,
    train_local_model,
)
from .paths import UnsafePathError, DATASET_REGISTRY, resolve_model_output

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

app = Flask(__name__)

# Reject oversized bodies at the WSGI layer, before any parsing happens.
# Generous multiple of MAX_INPUT_CHARS to allow for JSON overhead.
app.config["MAX_CONTENT_LENGTH"] = max(1_048_576, settings.MAX_INPUT_CHARS * 4)

# Per-IP rate limiting. The default in-memory store is per-process; set
# RATE_LIMIT_STORAGE_URI to a redis:// URI when running multiple workers.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
    strategy="fixed-window",
)

_ALLOWED_ORIGINS = set(settings.CORS_ALLOWED_ORIGINS)


@app.after_request
def add_cors_headers(response):
    """Echo the request Origin only when it is on the configured allow-list.

    The previous implementation returned ``Access-Control-Allow-Origin: *``,
    which lets any website on the internet call this API from a victim's
    browser. We now reflect the origin only if explicitly permitted.
    """
    origin = request.headers.get("Origin")
    if origin and origin in _ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Max-Age"] = "600"
    return response


# Instantiated once at startup so the joblib model is loaded a single time.
agent = LLMSearchAgent(provider=settings.LLM_PROVIDER)
logger.info(
    "Server initialised. model_loaded=%s provider=%s llm_enabled=%s origins=%s",
    is_model_loaded(), settings.LLM_PROVIDER, settings.ENABLE_LLM,
    sorted(_ALLOWED_ORIGINS),
)


# ----------------------------------------------------------------------
# Request validation helpers
# ----------------------------------------------------------------------

def _get_payload() -> dict:
    """Return the JSON body as a dict, tolerating a missing/!JSON body."""
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _validate_text(payload: dict, *keys: str) -> Tuple[Optional[str], Optional[tuple]]:
    """Extract and validate a text field from ``payload``.

    Enforces ``settings.MAX_INPUT_CHARS``. Rejecting (rather than silently
    truncating) keeps results honest: a truncated article could otherwise be
    scored on a fragment the user never saw.

    Returns:
        ``(text, None)`` on success, or ``(None, (response, status))``.
    """
    raw = next((payload.get(k) for k in keys if payload.get(k)), None)

    if raw is None or not str(raw).strip():
        return None, (
            jsonify({"error": f"Missing '{keys[0]}' in request body"}), 400
        )

    text = str(raw).strip()
    if len(text) > settings.MAX_INPUT_CHARS:
        logger.warning("Rejected oversized input: %d chars from %s",
                       len(text), get_remote_address())
        return None, (
            jsonify({
                "error": "Input too long",
                "detail": (f"Text must be at most {settings.MAX_INPUT_CHARS} "
                           f"characters; received {len(text)}."),
                "max_chars": settings.MAX_INPUT_CHARS,
            }), 400
        )
    return text, None


def _validate_page_size(payload: dict) -> Tuple[Optional[int], Optional[tuple]]:
    """Validate ``page_size`` as an int within the configured bounds.

    ``int(payload.get('page_size'))`` previously raised on any non-numeric
    value, turning a client typo into a 500. It also accepted values like
    100000, letting one request fan out into a huge upstream query.

    Returns:
        ``(page_size, None)`` on success, or ``(None, (response, status))``.
    """
    raw = payload.get("page_size")
    if raw is None or raw == "":
        return settings.DEFAULT_PAGE_SIZE, None

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, (
            jsonify({
                "error": "Invalid 'page_size'",
                "detail": "page_size must be an integer.",
            }), 400
        )

    if not settings.MIN_PAGE_SIZE <= value <= settings.MAX_PAGE_SIZE:
        return None, (
            jsonify({
                "error": "Invalid 'page_size'",
                "detail": (f"page_size must be between {settings.MIN_PAGE_SIZE} "
                           f"and {settings.MAX_PAGE_SIZE}."),
            }), 400
        )
    return value, None


@app.errorhandler(429)
def handle_rate_limit(err):
    """Return JSON (not HTML) when a rate limit trips."""
    logger.warning("Rate limit hit by %s on %s",
                   get_remote_address(), request.path)
    return jsonify({
        "error": "Rate limit exceeded",
        "detail": str(getattr(err, "description", "Too many requests.")),
    }), 429


@app.errorhandler(413)
def handle_payload_too_large(_err):
    """Return JSON when the WSGI body-size cap rejects a request."""
    return jsonify({
        "error": "Payload too large",
        "max_chars": settings.MAX_INPUT_CHARS,
    }), 413


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
@limiter.exempt
def health():
    """Report service and model status. Accuracy comes from model metadata."""
    info = get_model_info()
    return jsonify({
        "status": "ok",
        "model_loaded": info["is_loaded"],
        "model_type": info["model_type"],
        # Sourced from the trained bundle -- previously a hardcoded "98.80%"
        # that disagreed with the other two hardcoded values in the codebase.
        "model_accuracy": info["accuracy_display"],
        "model_trained_at": info["trained_at"],
        "llm_enabled": settings.ENABLE_LLM,
        "llm_provider": settings.LLM_PROVIDER if settings.ENABLE_LLM else None,
        "max_input_chars": settings.MAX_INPUT_CHARS,
    })


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
@limiter.limit(settings.RATE_LIMIT_ANALYZE, methods=["POST"])
def analyze_claim():
    """Run the full dual-branch verification pipeline on a claim."""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    payload = _get_payload()

    text, error = _validate_text(payload, "text", "prompt", "query")
    if error:
        return error

    page_size, error = _validate_page_size(payload)
    if error:
        return error

    try:
        return jsonify(agent.analyze(text, page_size=page_size))
    except Exception:
        logger.exception("Analysis failed for a %d-char input", len(text))
        # Do not leak internal exception text to the client.
        return jsonify({"error": "Analysis failed"}), 500


@app.route("/api/run_prompt", methods=["POST", "OPTIONS"])
@limiter.limit(settings.RATE_LIMIT_ANALYZE, methods=["POST"])
def run_prompt_route():
    """Legacy-compatible wrapper around :func:`analyze_claim`."""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    payload = _get_payload()

    prompt, error = _validate_text(payload, "prompt", "query", "text")
    if error:
        return error

    page_size, error = _validate_page_size(payload)
    if error:
        return error

    try:
        full_res = agent.analyze(prompt, page_size=page_size)
    except Exception:
        logger.exception("run_prompt failed")
        return jsonify({"error": "Analysis failed"}), 500

    final = full_res.get("final_analysis", {})
    ml = full_res.get("ml_classifier", {})
    sources = full_res.get("news_sources", [])

    label = ml.get("label")
    verdict = {"real": "valid", "fake": "invalid"}.get(label, "unknown")

    matched_article = None
    if sources:
        first = sources[0]
        matched_article = {
            "title": first.get("title"),
            "source": first.get("source"),
            "url": first.get("url"),
            "label": label,
            "score": ml.get("confidence"),
            "paragraph": first.get("description"),
        }

    return jsonify({
        "prompt": prompt,
        "verdict": verdict,
        "message": final.get("executive_summary", ""),
        "article": matched_article,
        **full_res,
    })


@app.route("/api/classify", methods=["POST", "OPTIONS"])
@limiter.limit(settings.RATE_LIMIT_ANALYZE, methods=["POST"])
def classify_text_route():
    """Classify text with the ML model only (no news search, no LLM)."""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    text, error = _validate_text(_get_payload(), "text", "query")
    if error:
        return error

    try:
        return jsonify(classify_with_probabilities(text))
    except Exception:
        logger.exception("Classification failed")
        return jsonify({"error": "Classification failed"}), 500


@app.route("/api/search", methods=["POST", "OPTIONS"])
@limiter.limit(settings.RATE_LIMIT_ANALYZE, methods=["POST"])
def search_route():
    """Search live news for a query and score each result with the model."""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    payload = _get_payload()

    query, error = _validate_text(payload, "query", "text")
    if error:
        return error

    page_size, error = _validate_page_size(payload)
    if error:
        return error

    try:
        articles = agent.search_and_fetch(query, page_size=page_size)
    except Exception:
        logger.exception("News search failed for query=%r", query)
        return jsonify({"error": "Search failed"}), 500

    results = []
    for article in articles:
        title = article.get("title") or ""
        source = agent._source_name(article)
        content = article.get("content") or article.get("description") or ""
        ml_meta = classify_with_probabilities(f"{title} {source} {content}".strip())
        results.append({
            "title": title,
            "source": source,
            "url": article.get("url"),
            "label": ml_meta.get("label"),
            "fake_probability": ml_meta.get("fake_probability"),
            "real_probability": ml_meta.get("real_probability"),
            "score": ml_meta.get("score"),
        })

    return jsonify({"query": query, "results": results})


@app.route("/api/train_local", methods=["POST", "OPTIONS"])
@limiter.limit(settings.RATE_LIMIT_TRAIN, methods=["POST"])
def train_local_route():
    """Retrain the classifier from the server's bundled datasets.

    Hardened against the previous arbitrary-path read/write. This endpoint no
    longer accepts filesystem paths at all:

      * ``dataset`` is a *logical name* validated against DATASET_REGISTRY.
      * ``model_name`` is a bare filename forced into server/ml/models/.

    An attacker-controlled ``out_path`` was especially dangerous because
    ``joblib.dump`` writes a pickle -- writing one over an importable module
    or a startup script is a path to code execution.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    # Retraining is expensive and destructive; off unless explicitly enabled.
    if not settings.ENABLE_TRAIN_ENDPOINT:
        return jsonify({
            "error": "Training endpoint is disabled",
            "detail": "Set ENABLE_TRAIN_ENDPOINT=true to enable it.",
        }), 403

    payload = _get_payload()

    try:
        model_out = resolve_model_output(
            payload.get("model_name") or "fake_news_model.joblib"
        )
        # `dataset` is accepted only to validate the caller's intent; training
        # always scans the sandboxed data directory.
        dataset = payload.get("dataset")
        if dataset is not None and str(dataset).strip().lower() not in DATASET_REGISTRY:
            raise UnsafePathError(
                f"Unknown dataset '{dataset}'. "
                f"Allowed: {', '.join(sorted(DATASET_REGISTRY))}."
            )
    except UnsafePathError as exc:
        logger.warning("Rejected unsafe train_local request from %s: %s",
                       get_remote_address(), exc)
        return jsonify({"error": "Invalid request", "detail": str(exc)}), 400

    try:
        saved = train_local_model(model_out=model_out)
    except Exception:
        logger.exception("Training failed")
        return jsonify({"error": "Training failed"}), 500

    return jsonify({
        "status": "ok",
        # Return only the basename; the absolute server path is not the
        # client's business.
        "model_name": saved.rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
        "model_info": get_model_info(),
    })
