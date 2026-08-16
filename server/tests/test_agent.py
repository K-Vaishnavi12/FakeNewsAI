import pytest
from agent import LLMSearchAgent


def test_evidence_classification_contradicts():
    agent = LLMSearchAgent(provider='hf')
    claim = "Scientists discovered drinkable water on Mars this week"

    # Article that explicitly refutes/debunks the claim
    debunk_article = {
        "title": "NASA says viral claim of liquid drinking water on Mars is false",
        "source": {"name": "BBC News"},
        "url": "https://bbc.com/news/science-123",
        "description": "Viral rumors claim scientists found drinkable water on Mars, but researchers confirmed the report is a hoax and debunked.",
        "publishedAt": "2026-08-15"
    }

    result = agent._classify_article_heuristic(claim, debunk_article)
    assert result["stance"] == "CONTRADICTS CLAIM"
    assert result["stance_type"] == "contradict"


def test_evidence_classification_supports():
    agent = LLMSearchAgent(provider='hf')
    claim = "James Webb Space Telescope captures ancient galaxy clusters"

    # Article that confirms the claim
    confirm_article = {
        "title": "James Webb Space Telescope captures ancient galaxy clusters in deep space",
        "source": {"name": "Reuters"},
        "url": "https://reuters.com/science-webb-galaxy",
        "description": "Astronomers using the James Webb Space Telescope released new images detailing ancient galaxy clusters formed over 13 billion years ago.",
        "publishedAt": "2026-08-15"
    }

    result = agent._classify_article_heuristic(claim, confirm_article)
    assert result["stance"] == "SUPPORTS CLAIM"
    assert result["stance_type"] == "support"


def test_evidence_classification_neutral():
    agent = LLMSearchAgent(provider='hf')
    claim = "Secret base discovered on the Moon"

    # Article mentioning general moon missions without confirming or debunking
    neutral_article = {
        "title": "Artemis lunar mission preparations enter final phase",
        "source": {"name": "Space.com"},
        "url": "https://space.com/artemis-update",
        "description": "Engineers are preparing the spacecraft for upcoming scheduled lunar orbit operations next month.",
        "publishedAt": "2026-08-15"
    }

    result = agent._classify_article_heuristic(claim, neutral_article)
    assert result["stance"] == "NEUTRAL / RELATED"
    assert result["stance_type"] == "neutral"


def test_decision_matrix_case1_contradiction_debunked():
    agent = LLMSearchAgent(provider='hf')
    claim = "Scientists discovered drinkable water on Mars this week"

    ml_result = {
        "label": "fake",
        "fake_probability": 0.85,
        "real_probability": 0.15,
        "top_signals": [{"word": "mars", "impact": "fake", "weight": -0.5}]
    }

    breakdown = {
        "supports_count": 0,
        "contradicts_count": 1,
        "neutral_count": 1,
        "total_articles": 2,
        "credible_supports": [],
        "credible_contradicts": ["BBC News"]
    }

    decision = agent._execute_decision_matrix(claim, ml_result, breakdown, [])
    assert decision["verdict_type"] == "fake"
    assert "Debunked" in decision["verdict"] or "Fake" in decision["verdict"]
    assert decision["confidence_score"] >= 0.85


def test_decision_matrix_case2_supports_verified():
    agent = LLMSearchAgent(provider='hf')
    claim = "James Webb Space Telescope observes ancient galaxy clusters"

    ml_result = {
        "label": "real",
        "fake_probability": 0.12,
        "real_probability": 0.88,
        "top_signals": [{"word": "telescope", "impact": "real", "weight": 0.7}]
    }

    breakdown = {
        "supports_count": 2,
        "contradicts_count": 0,
        "neutral_count": 0,
        "total_articles": 2,
        "credible_supports": ["Reuters", "NASA"],
        "credible_contradicts": []
    }

    decision = agent._execute_decision_matrix(claim, ml_result, breakdown, [])
    assert decision["verdict_type"] == "real"
    assert "Verified" in decision["verdict"] or "Real" in decision["verdict"]
    assert decision["confidence_score"] >= 0.85
