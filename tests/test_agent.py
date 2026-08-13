"""Step 11c — agent tests with mocked tools and simulated failures."""

from __future__ import annotations

import asyncio

import pytest

from src.agent import graph as agent_graph
from src.agent.graph import (
    ToolBelt,
    _fallback_payload,
    ainvestigate,
    heuristic_claims,
    heuristic_status,
    investigate,
)

REQUIRED_KEYS = {
    "verdict",
    "trust_score",
    "band",
    "claims",
    "evidence",
    "explanation",
    "citations",
    "degraded",
}

FAKE_EVIDENCE = [
    {
        "text": "The claim that bleach cures viruses is false and dangerous.",
        "claim": "Bleach cures viruses",
        "rating": "False",
        "publisher": "Snopes",
        "url": "https://example.org/bleach",
        "score": 0.91,
    }
]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Force heuristic mode so tests never touch a network LLM."""
    monkeypatch.setattr(agent_graph, "get_llm", lambda: None)
    agent_graph.reset_llm_cache()
    yield
    agent_graph.reset_llm_cache()


@pytest.fixture
def mocked_tools(monkeypatch):
    """Replace every MCP-backed capability with fast in-memory stubs."""

    async def classify(self, text):
        return {
            "verdict": "Fake",
            "band": "Fake",
            "trust_score": 12.0,
            "probability_real": 0.12,
            "model": "mock",
            "degraded": False,
        }

    async def explain(self, text):
        return {
            "tokens": [{"word": "SHOCKING", "weight": -0.4}],
            "backend": "mock",
            "degraded": False,
        }

    async def evidence(self, claim, k=3):
        return [dict(item) for item in FAKE_EVIDENCE]

    async def fact_checks(self, query, k=2):
        return []

    monkeypatch.setattr(ToolBelt, "classify", classify)
    monkeypatch.setattr(ToolBelt, "explain", explain)
    monkeypatch.setattr(ToolBelt, "evidence", evidence)
    monkeypatch.setattr(ToolBelt, "fact_checks", fact_checks)
    monkeypatch.setattr(ToolBelt, "load_mcp", lambda self: asyncio.sleep(0))


def test_investigate_with_mocked_tools(mocked_tools):
    result = investigate("Bleach cures every virus. The government hid the proof for decades.")
    assert REQUIRED_KEYS <= set(result)
    assert result["verdict"] == "Fake"
    assert result["claims"]
    assert result["citations"]
    assert result["citations"][0]["publisher"] == "Snopes"
    assert result["explanation"]


def test_claims_are_capped_at_four(mocked_tools):
    text = " ".join(f"The government confirmed fact number {i} on Tuesday." for i in range(20))
    assert len(investigate(text)["claims"]) <= 4


def test_refuting_evidence_marks_claim_refuted(mocked_tools):
    result = investigate("Bleach cures every known virus according to secret research.")
    assert any(claim["status"] == "refuted" for claim in result["claims"])


def test_citation_ids_resolve(mocked_tools):
    result = investigate("Bleach cures every known virus according to secret research.")
    valid = {c["id"] for c in result["citations"]}
    for claim in result["claims"]:
        assert set(claim["evidence_ids"]) <= valid


def test_llm_failure_degrades_not_crashes(monkeypatch, mocked_tools):
    async def boom(_prompt):
        raise RuntimeError("Gemini exploded")

    monkeypatch.setattr(agent_graph, "_ask_llm", boom)
    result = investigate("The mayor resigned on Tuesday after the audit was published.")
    assert result["degraded"] is True
    assert result["explanation"]
    assert REQUIRED_KEYS <= set(result)


def test_pipeline_failure_returns_degraded_fallback(monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr(agent_graph, "_run_pipeline", boom)
    monkeypatch.setattr(ToolBelt, "load_mcp", lambda self: asyncio.sleep(0))
    result = investigate("Some article text that should still produce a response.")
    assert result["degraded"] is True
    assert result["claims"] == []
    assert "unavailable" in result["explanation"].lower()


def test_timeout_returns_degraded_fallback(monkeypatch):
    async def slow(*_args, **_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(agent_graph, "_run_pipeline", slow)
    monkeypatch.setattr(ToolBelt, "load_mcp", lambda self: asyncio.sleep(0))
    result = investigate("Text that will time out.", timeout=0.5)
    assert result["degraded"] is True
    assert result["notes"] == ["fallback:timeout"]


def test_empty_input_short_circuits():
    result = investigate("   ")
    assert result["degraded"] is True
    assert result["claims"] == []


def test_tool_budget_is_enforced():
    belt = ToolBelt(budget=2, use_mcp=False)
    assert belt.exhausted is False
    belt._spend()
    belt._spend()
    assert belt.exhausted is True
    assert belt._spend() is False


def test_fallback_payload_shape():
    payload = _fallback_payload("some text", "timeout")
    assert REQUIRED_KEYS <= set(payload)
    assert payload["degraded"] is True
    assert payload["tool_calls"] == 0


def test_heuristic_claims_respects_limit():
    text = "One fact happened here. Two facts happened there. Three facts happened elsewhere."
    assert len(heuristic_claims(text, limit=2)) == 2


@pytest.mark.parametrize(
    ("rating", "expected"),
    [("False", "refuted"), ("True", "supported"), ("Mixture of things", "unverified")],
)
def test_heuristic_status_reads_ratings(rating, expected):
    evidence = [{"rating": rating, "text": "x", "publisher": "p", "url": "u"}]
    assert heuristic_status("a claim", evidence)["status"] == expected


def test_heuristic_status_without_evidence():
    assert heuristic_status("a claim", [])["status"] == "unverified"


@pytest.mark.asyncio
async def test_ainvestigate_is_awaitable(mocked_tools):
    result = await ainvestigate("The bridge project cost four billion dollars.")
    assert REQUIRED_KEYS <= set(result)
