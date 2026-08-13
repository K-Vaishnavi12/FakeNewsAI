"""Step 11a — model layer contract tests."""

from __future__ import annotations

import pytest

from src.config import band_for_score
from src.explain.shap_explainer import explain
from src.models.predict import predict, predict_proba

VALID_BANDS = {"Real", "Suspicious", "Fake"}
REQUIRED_KEYS = {
    "verdict",
    "band",
    "trust_score",
    "probability_real",
    "model",
    "degraded",
}


def test_predict_returns_full_schema(sample_texts):
    result = predict(sample_texts["fake"])
    assert REQUIRED_KEYS <= set(result)


def test_trust_score_within_range(sample_texts):
    for text in sample_texts.values():
        result = predict(text)
        assert 0.0 <= result["trust_score"] <= 100.0
        assert 0.0 <= result["probability_real"] <= 1.0


def test_verdict_is_a_valid_band(sample_texts):
    for text in sample_texts.values():
        result = predict(text)
        assert result["verdict"] in VALID_BANDS
        assert result["band"] == result["verdict"]


def test_band_matches_score(sample_texts):
    result = predict(sample_texts["real"])
    assert result["band"] == band_for_score(result["trust_score"])


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "Fake"), (39.9, "Fake"), (40.0, "Suspicious"), (69.9, "Suspicious"), (70.0, "Real"), (100.0, "Real")],
)
def test_band_boundaries(score, expected):
    assert band_for_score(score) == expected


def test_predict_handles_empty_and_whitespace():
    for text in ("", "   ", "\n\t"):
        result = predict(text)
        assert result["verdict"] in VALID_BANDS
        assert result["degraded"] is True


def test_predict_handles_very_long_input():
    result = predict("breaking news about the election " * 1000)
    assert 0.0 <= result["trust_score"] <= 100.0


def test_predict_handles_unicode(sample_texts):
    result = predict(sample_texts["unicode"])
    assert result["verdict"] in VALID_BANDS


def test_predict_is_deterministic(sample_texts):
    first = predict(sample_texts["fake"])
    second = predict(sample_texts["fake"])
    assert first["trust_score"] == pytest.approx(second["trust_score"])


def test_predict_proba_is_vectorised(sample_texts):
    texts = [sample_texts["real"], sample_texts["fake"]]
    probas = predict_proba(texts)
    assert len(probas) == 2
    assert all(0.0 <= float(p) <= 1.0 for p in probas)


def test_predict_proba_on_empty_list():
    assert len(predict_proba([])) == 0


def test_explain_returns_ranked_tokens(sample_texts):
    tokens = explain(sample_texts["fake"], top_k=10)
    assert len(tokens) <= 10
    assert all({"word", "weight"} <= set(t) for t in tokens)
    weights = [abs(float(t["weight"])) for t in tokens]
    assert weights == sorted(weights, reverse=True)


def test_explain_short_input_returns_empty():
    assert explain("hi") == []


def test_explain_never_raises_on_junk():
    assert isinstance(explain("!!! ??? @@@ ###"), list)
