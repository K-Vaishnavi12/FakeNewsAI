"""HTTP-level tests for request validation, CORS and rate limiting.

These are regression tests for the security/robustness issues fixed in
``app.py``: wildcard CORS, unbounded input, and ``int(page_size)`` raising a
500 on non-numeric input.
"""

import pytest

from server.app import app, limiter
from server.config import settings


@pytest.fixture
def client(monkeypatch):
    """Flask test client with the network and model stubbed out."""
    app.config["TESTING"] = True
    # Rate limits would otherwise make repeated test runs flaky.
    limiter.enabled = False
    monkeypatch.setattr("server.agent.search_news", lambda q, page_size=5: [])
    with app.test_client() as c:
        yield c
    limiter.enabled = True


# ----------------------------------------------------------------------
# Input size cap
# ----------------------------------------------------------------------

def test_analyze_rejects_oversized_text(client):
    payload = {"text": "a" * (settings.MAX_INPUT_CHARS + 1)}
    res = client.post("/api/analyze", json=payload)

    assert res.status_code == 400
    assert res.get_json()["error"] == "Input too long"


def test_analyze_accepts_text_at_the_limit(client):
    payload = {"text": "a" * settings.MAX_INPUT_CHARS}
    assert client.post("/api/analyze", json=payload).status_code == 200


def test_analyze_rejects_missing_text(client):
    assert client.post("/api/analyze", json={}).status_code == 400


def test_analyze_tolerates_non_json_body(client):
    """A malformed body must be a 400, never an unhandled 500."""
    res = client.post("/api/analyze", data="not json",
                      content_type="application/json")
    assert res.status_code == 400


# ----------------------------------------------------------------------
# page_size validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "1.5.2", {}, [], "1e9"])
def test_non_numeric_page_size_returns_400_not_500(client, bad):
    res = client.post("/api/analyze", json={"text": "hello", "page_size": bad})
    assert res.status_code == 400
    assert res.get_json()["error"] == "Invalid 'page_size'"


@pytest.mark.parametrize("bad", [0, -1, 100000])
def test_out_of_range_page_size_rejected(client, bad):
    res = client.post("/api/analyze", json={"text": "hello", "page_size": bad})
    assert res.status_code == 400


def test_omitted_page_size_uses_default(client):
    assert client.post("/api/analyze", json={"text": "hello"}).status_code == 200


# ----------------------------------------------------------------------
# CORS allow-list
# ----------------------------------------------------------------------

def test_cors_allows_configured_origin(client):
    origin = settings.CORS_ALLOWED_ORIGINS[0]
    res = client.get("/api/health", headers={"Origin": origin})
    assert res.headers.get("Access-Control-Allow-Origin") == origin


def test_cors_blocks_unknown_origin(client):
    res = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in res.headers


def test_cors_never_returns_wildcard(client):
    res = client.get("/api/health",
                     headers={"Origin": settings.CORS_ALLOWED_ORIGINS[0]})
    assert res.headers.get("Access-Control-Allow-Origin") != "*"


# ----------------------------------------------------------------------
# Health & training endpoints
# ----------------------------------------------------------------------

def test_health_reports_accuracy_from_metadata(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    # Must not be one of the old hardcoded literals.
    assert body["model_accuracy"] not in {"98.80%", "97.08%"} or body["model_loaded"]
    assert "max_input_chars" in body


def test_train_endpoint_disabled_by_default(client):
    res = client.post("/api/train_local", json={})
    assert res.status_code == 403


def test_train_endpoint_rejects_path_traversal(client, monkeypatch):
    """Even with the endpoint enabled, path escapes must be refused."""
    monkeypatch.setattr(settings, "ENABLE_TRAIN_ENDPOINT", True)
    res = client.post("/api/train_local",
                      json={"model_name": "../../../../tmp/evil.joblib"})
    assert res.status_code == 400
    assert res.get_json()["error"] == "Invalid request"
