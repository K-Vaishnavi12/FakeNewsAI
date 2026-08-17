"""Tests for the analysis pipeline and the request-validation helpers.

The previous version of this file was already failing: it referenced
``cli.classify`` and ``cli._choose_best_article``, neither of which exists,
and it loaded ``cli.py`` by file path with ``importlib`` to dodge the old
import hacks. Now that ``server`` is a real package, tests import it normally.
"""

import pytest

from server import cli
from server.agent import LLMSearchAgent
from server.paths import UnsafePathError, resolve_dataset, resolve_model_output, resolve_within


STUB_ARTICLES = [
    {
        "title": "NASA confirms water on the Moon",
        "source": {"name": "Reuters"},
        "url": "https://example.com/a",
        "description": "NASA scientists confirmed water on the lunar surface.",
        "content": "NASA scientists confirmed water on the lunar surface.",
        "publishedAt": "2024-01-01T00:00:00Z",
    },
    {
        "title": "NASA water on Moon discovery detailed",
        "source": {"name": "BBC"},
        "url": "https://example.com/b",
        "description": "Further detail on the NASA lunar water confirmation.",
        "content": "Further detail on the NASA lunar water confirmation.",
        "publishedAt": "2024-01-01T00:00:00Z",
    },
]


@pytest.fixture
def stub_search(monkeypatch):
    """Replace live news retrieval with deterministic stub articles."""
    monkeypatch.setattr(
        "server.agent.search_news",
        lambda query, page_size=5: STUB_ARTICLES[:page_size],
    )


@pytest.fixture
def stub_classifier(monkeypatch):
    """Force a deterministic 'real' classification."""
    monkeypatch.setattr(
        "server.agent.classify_with_probabilities",
        lambda text: {
            "label": "real",
            "fake_probability": 0.05,
            "real_probability": 0.95,
            "confidence": 0.95,
            "score": 0.95,
            "top_signals": [{"word": "nasa", "impact": "real", "weight": 0.9}],
            "is_loaded": True,
        },
    )


# ----------------------------------------------------------------------
# Agent pipeline
# ----------------------------------------------------------------------

def test_analyze_returns_expected_report_shape(stub_search, stub_classifier):
    res = LLMSearchAgent(provider="hf").analyze("NASA confirms water on the Moon")

    assert set(res) >= {
        "query", "final_analysis", "ml_classifier", "news_sources",
        "corroboration", "pipeline_status",
    }
    assert res["final_analysis"]["verdict_type"] in {"real", "fake", "unverified"}
    assert 0.0 <= res["final_analysis"]["confidence_score"] <= 1.0


def test_corroborated_claim_is_marked_real(stub_search, stub_classifier):
    res = LLMSearchAgent(provider="hf").analyze("NASA confirms water on the Moon")

    assert res["corroboration"]["is_corroborated"] is True
    assert res["final_analysis"]["verdict_type"] == "real"
    assert "Reuters" in res["corroboration"]["matched_sources"]


def test_no_articles_yields_no_corroboration(monkeypatch, stub_classifier):
    monkeypatch.setattr("server.agent.search_news", lambda q, page_size=5: [])
    res = LLMSearchAgent(provider="hf").analyze("Some unremarkable claim")

    assert res["news_sources"] == []
    assert res["corroboration"]["is_corroborated"] is False
    assert res["pipeline_status"]["articles_count"] == 0


def test_llm_disabled_by_default_leaves_synthesis_untouched(stub_search, stub_classifier):
    """With ENABLE_LLM=false the rule engine output must pass through as-is."""
    agent = LLMSearchAgent(provider="hf")
    agent.llm_enabled = False
    res = agent.analyze("NASA confirms water on the Moon")
    assert res["final_analysis"]["llm_refined"] is False


def test_llm_cannot_override_verdict(stub_search, stub_classifier, monkeypatch):
    """A hostile/hallucinating LLM must not be able to flip the verdict.

    Simulates a prompt-injection success: the model returns a verdict field
    and a confidence of 1.0. Only the allow-listed prose fields may apply.
    """
    agent = LLMSearchAgent(provider="hf")
    agent.llm_enabled = True
    monkeypatch.setattr(
        agent, "_call_llm",
        lambda prompt: '{"verdict": "Likely Fake / Fabricated", '
                       '"verdict_type": "fake", "confidence_score": 1.0, '
                       '"executive_summary": "Rewritten summary."}',
    )

    res = agent.analyze("NASA confirms water on the Moon")
    final = res["final_analysis"]

    assert final["verdict_type"] == "real"          # unchanged
    assert final["confidence_score"] != 1.0         # unchanged
    assert final["executive_summary"] == "Rewritten summary."  # allow-listed


def test_malformed_llm_output_falls_back_to_rules(stub_search, stub_classifier, monkeypatch):
    agent = LLMSearchAgent(provider="hf")
    agent.llm_enabled = True
    monkeypatch.setattr(agent, "_call_llm", lambda prompt: "not json at all")

    res = agent.analyze("NASA confirms water on the Moon")
    assert res["final_analysis"]["llm_refined"] is False


def test_extract_keywords_drops_stopwords():
    agent = LLMSearchAgent(provider="hf")
    keywords = agent._extract_keywords("BREAKING: the shocking NASA moon report")
    assert "breaking" not in keywords.lower()
    assert "shocking" not in keywords.lower()
    assert "NASA" in keywords


def test_extract_keywords_falls_back_when_all_stopwords():
    """If filtering leaves nothing, keep the raw words rather than an empty query."""
    agent = LLMSearchAgent(provider="hf")
    assert agent._extract_keywords("the news is that").strip()


# ----------------------------------------------------------------------
# Path sandboxing (regression tests for the /api/train_local traversal bug)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("evil", [
    "../../etc/passwd",
    "..\\..\\Windows\\System32\\cfg",
    "/etc/passwd",
    "C:\\Windows\\System32\\evil.joblib",
    "\\\\server\\share\\evil",
    "",
])
def test_resolve_within_rejects_escapes(tmp_path, evil):
    with pytest.raises(UnsafePathError):
        resolve_within(str(tmp_path), evil)


def test_resolve_within_allows_contained_path(tmp_path):
    resolved = resolve_within(str(tmp_path), "sub/file.csv")
    assert resolved.startswith(str(tmp_path.resolve()))


def test_resolve_dataset_rejects_unknown_name():
    with pytest.raises(UnsafePathError):
        resolve_dataset("../../../secret")


@pytest.mark.parametrize("bad", [
    "../evil.joblib",
    "sub/dir/model.joblib",
    "model.exe",
    "model.joblib.txt",
])
def test_resolve_model_output_rejects_bad_names(bad):
    with pytest.raises(UnsafePathError):
        resolve_model_output(bad)


def test_resolve_model_output_accepts_plain_name():
    assert resolve_model_output("fake_news_model.joblib").endswith("fake_news_model.joblib")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_cli_clamps_page_size():
    from server.config import settings
    assert cli._clamp_page_size(99999) == settings.MAX_PAGE_SIZE
    assert cli._clamp_page_size(-5) == settings.MIN_PAGE_SIZE
    assert cli._clamp_page_size(7) == 7
