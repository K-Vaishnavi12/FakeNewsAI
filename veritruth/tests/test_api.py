"""Step 11b — API contract tests covering all five endpoints."""

from __future__ import annotations

import pytest

VALID_BANDS = {"Real", "Suspicious", "Fake"}


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert "/predict" in body["endpoints"]


def test_health_reports_subsystems(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["evidence_chunks"], int)
    assert "model" in body


def test_predict_happy_path(client, sample_texts):
    response = client.post("/predict", json={"text": sample_texts["fake"]})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in VALID_BANDS
    assert 0.0 <= body["trust_score"] <= 100.0
    assert 0.0 <= body["probability_real"] <= 1.0


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": None}])
def test_predict_rejects_bad_input_with_422(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_predict_accepts_10k_characters(client):
    response = client.post("/predict", json={"text": "election fraud claim " * 500})
    assert response.status_code == 200
    assert 0.0 <= response.json()["trust_score"] <= 100.0


def test_predict_rejects_oversized_input(client):
    response = client.post("/predict", json={"text": "x" * 25_000})
    assert response.status_code == 422


def test_predict_handles_unicode(client, sample_texts):
    response = client.post("/predict", json={"text": sample_texts["unicode"]})
    assert response.status_code == 200
    assert response.json()["verdict"] in VALID_BANDS


def test_explain_returns_bounded_tokens(client, sample_texts):
    response = client.post("/explain", json={"text": sample_texts["fake"], "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["tokens"]) <= 5
    assert all({"word", "weight"} <= set(t) for t in body["tokens"])


def test_explain_validates_top_k(client, sample_texts):
    assert client.post("/explain", json={"text": sample_texts["fake"], "top_k": 0}).status_code == 422
    assert client.post("/explain", json={"text": sample_texts["fake"], "top_k": 99}).status_code == 422


def test_investigate_returns_full_payload(client, sample_texts):
    response = client.post("/investigate", json={"text": sample_texts["fake"]})
    assert response.status_code == 200
    body = response.json()
    for key in ("verdict", "trust_score", "claims", "evidence", "explanation", "citations", "degraded"):
        assert key in body
    assert body["verdict"] in VALID_BANDS
    assert len(body["claims"]) <= 4
    assert isinstance(body["degraded"], bool)


def test_investigate_rejects_empty_text(client):
    assert client.post("/investigate", json={"text": ""}).status_code == 422


def test_feedback_round_trip(client, sample_texts):
    payload = {
        "text": sample_texts["fake"],
        "predicted_verdict": "Fake",
        "user_verdict": "Suspicious",
        "trust_score": 21.5,
        "comment": "satire, not disinformation",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["stored"] is True
    assert body["id"] > 0
    assert body["total"] >= 1


def test_feedback_rejects_unknown_verdict(client, sample_texts):
    payload = {
        "text": sample_texts["fake"],
        "predicted_verdict": "Bogus",
        "user_verdict": "Fake",
        "trust_score": 10.0,
    }
    assert client.post("/feedback", json=payload).status_code == 422


def test_unknown_route_returns_json_error(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "http_error"
