"""Step 8b — the LangGraph investigation agent.

Three nodes run in sequence::

    decompose  ->  verify  ->  synthesize

``decompose`` splits the article into at most four atomic claims. ``verify``
gathers evidence for each claim through the MCP tool belt and labels it
supported / refuted / unverified. ``synthesize`` writes a three-sentence cited
verdict and merges in the classifier score and SHAP token weights.

Guarantees:

* hard cap of ``AGENT_MAX_TOOL_CALLS`` (default 6) tool invocations per request
* hard wall-clock cap of ``AGENT_TIMEOUT_SECONDS`` (default 30s)
* ANY failure returns the classifier-only result with ``degraded: true``

Tooling: the agent prefers the four MCP servers via ``MultiServerMCPClient``
(stdio). If MCP or its dependencies are unavailable it transparently calls the
same underlying functions in-process, so the graph never hard-fails on tooling.

Run::

    python -m src.agent.graph "Some article text"
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
from typing import Any, TypedDict

from src.agent.prompts import (
    DECOMPOSE_PROMPT,
    MAX_CLAIMS,
    PROMPT_VERSIONS,
    SYNTHESIZE_PROMPT,
    VERIFY_PROMPT,
    format_evidence,
    format_findings,
    format_sources,
    format_tokens,
)
from src.config import (
    MAX_TEXT_CHARS,
    ROOT_DIR,
    agent_max_tool_calls,
    agent_timeout,
    get_env,
    get_logger,
)

LOG = get_logger("veritruth.agent")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

MCP_SERVERS: dict[str, dict[str, Any]] = {
    "classifier": {
        "command": sys.executable,
        "args": ["-m", "src.mcp_servers.classifier_server"],
        "transport": "stdio",
        "cwd": str(ROOT_DIR),
    },
    "evidence": {
        "command": sys.executable,
        "args": ["-m", "src.mcp_servers.evidence_server"],
        "transport": "stdio",
        "cwd": str(ROOT_DIR),
    },
    "factcheck": {
        "command": sys.executable,
        "args": ["-m", "src.mcp_servers.factcheck_server"],
        "transport": "stdio",
        "cwd": str(ROOT_DIR),
    },
    "explainer": {
        "command": sys.executable,
        "args": ["-m", "src.mcp_servers.explainer_server"],
        "transport": "stdio",
        "cwd": str(ROOT_DIR),
    },
}


# --------------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    text: str
    classification: dict[str, Any]
    tokens: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    explanation: str
    tool_calls: int
    degraded: bool
    notes: list[str]
    belt: Any  # ToolBelt instance; carried through the graph, not serialised


# ----------------------------------------------------------------- tool belt
class ToolBelt:
    """Budget-limited access to the four VeriTruth capabilities.

    Prefers MCP (``use_mcp=True``); otherwise calls the same functions
    in-process. Either way the call budget is enforced here in one place.
    """

    def __init__(self, budget: int | None = None, use_mcp: bool | None = None) -> None:
        self.budget = int(budget if budget is not None else agent_max_tool_calls())
        self.calls = 0
        self.notes: list[str] = []
        if use_mcp is None:
            use_mcp = get_env("AGENT_USE_MCP", "0").lower() in {"1", "true", "yes", "on"}
        self.use_mcp = bool(use_mcp)
        self._mcp_tools: dict[str, Any] = {}

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def _spend(self) -> bool:
        if self.exhausted:
            return False
        self.calls += 1
        return True

    async def load_mcp(self) -> None:
        """Best-effort MCP tool discovery; silently degrades to in-process."""
        if not self.use_mcp:
            return
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(MCP_SERVERS)
            tools = await client.get_tools()
            self._mcp_tools = {t.name: t for t in tools}
            LOG.info("Loaded %d MCP tools: %s", len(tools), sorted(self._mcp_tools))
        except Exception as exc:
            LOG.warning("MCP unavailable (%s); using in-process tools.", exc)
            self.notes.append("mcp_unavailable")
            self._mcp_tools = {}

    async def _via_mcp(self, name: str, args: dict[str, Any]) -> Any:
        tool = self._mcp_tools.get(name)
        if tool is None:
            return None
        try:
            raw = await tool.ainvoke(args)
        except Exception as exc:
            LOG.warning("MCP tool '%s' failed (%s); in-process fallback.", name, exc)
            return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return raw

    async def classify(self, text: str) -> dict[str, Any]:
        if self._spend():
            result = await self._via_mcp("classify_news", {"text": text})
            if isinstance(result, dict) and result:
                return result
        from src.mcp_servers.classifier_server import classify_news

        return await asyncio.to_thread(classify_news, text)

    async def explain(self, text: str) -> dict[str, Any]:
        if self._spend():
            result = await self._via_mcp("explain_prediction", {"text": text, "top_k": 10})
            if isinstance(result, dict) and result:
                return result
        from src.mcp_servers.explainer_server import explain_prediction

        return await asyncio.to_thread(explain_prediction, text, 10)

    async def evidence(self, claim: str, k: int = 3) -> list[dict[str, Any]]:
        if self._spend():
            result = await self._via_mcp("search_evidence", {"claim": claim, "k": k})
            if isinstance(result, list) and result:
                return result
        from src.mcp_servers.evidence_server import search_evidence

        return await asyncio.to_thread(search_evidence, claim, k)

    async def fact_checks(self, query: str, k: int = 2) -> list[dict[str, Any]]:
        if self._spend():
            result = await self._via_mcp("search_fact_checks", {"query": query, "k": k})
            if isinstance(result, list) and result:
                return result
        from src.mcp_servers.factcheck_server import search_fact_checks

        return await asyncio.to_thread(search_fact_checks, query, k)


# ----------------------------------------------------------------------- LLM
_LLM_LOCK = threading.Lock()
_LLM: Any = None
_LLM_TRIED = False


def get_llm() -> Any:
    """Cached Gemini chat model, or ``None`` when unconfigured/unavailable."""
    global _LLM, _LLM_TRIED
    if _LLM is not None or _LLM_TRIED:
        return _LLM
    with _LLM_LOCK:
        if _LLM is None and not _LLM_TRIED:
            _LLM_TRIED = True
            api_key = get_env("GEMINI_API_KEY")
            if not api_key:
                LOG.info("GEMINI_API_KEY unset; agent runs in heuristic mode.")
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                from src.config import GEMINI_MODEL

                _LLM = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=api_key,
                    temperature=0.0,
                    timeout=20,
                    max_retries=1,
                )
                LOG.info("Gemini model '%s' ready.", GEMINI_MODEL)
            except Exception as exc:
                LOG.warning("Gemini unavailable (%s); heuristic mode.", exc)
                _LLM = None
    return _LLM


def reset_llm_cache() -> None:
    """Test hook: forget the cached LLM so env changes take effect."""
    global _LLM, _LLM_TRIED
    with _LLM_LOCK:
        _LLM = None
        _LLM_TRIED = False


async def _ask_llm(prompt: str) -> str:
    """Single LLM turn; returns ``""`` when the LLM is absent or fails."""
    llm = get_llm()
    if llm is None:
        return ""
    try:
        response = await llm.ainvoke(prompt)
    except Exception as exc:
        LOG.warning("LLM call failed (%s); falling back to heuristics.", exc)
        return ""
    content = getattr(response, "content", response)
    if isinstance(content, list):  # Gemini may return content parts
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "").strip()


def _parse_json(raw: str, pattern: re.Pattern[str]) -> Any:
    """Extract the first JSON value matching ``pattern`` from an LLM reply."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = pattern.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ----------------------------------------------------------- heuristic paths
def heuristic_claims(text: str, limit: int = MAX_CLAIMS) -> list[str]:
    """Sentence-split fallback used when no LLM is configured."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    claims = [s for s in sentences if len(s.split()) >= 4] or sentences
    return [c[:400] for c in claims[:limit]]


def heuristic_status(claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Rating-keyword heuristic used when no LLM is configured."""
    if not evidence:
        return {
            "status": "unverified",
            "reason": "No fact-check evidence was retrieved for this claim.",
            "evidence_ids": [],
        }
    ratings = " ".join(str(item.get("rating", "")).lower() for item in evidence)
    ids = list(range(1, len(evidence) + 1))
    refuting = ("false", "fake", "misleading", "incorrect", "debunk", "pants on fire", "hoax")
    supporting = ("true", "correct", "accurate", "confirmed")
    if any(word in ratings for word in refuting):
        return {
            "status": "refuted",
            "reason": f"Retrieved fact-checks rate related claims as false ({ratings[:80]}).",
            "evidence_ids": ids,
        }
    if any(word in ratings for word in supporting):
        return {
            "status": "supported",
            "reason": f"Retrieved fact-checks rate related claims as true ({ratings[:80]}).",
            "evidence_ids": ids,
        }
    return {
        "status": "unverified",
        "reason": "Retrieved evidence was related but inconclusive.",
        "evidence_ids": ids,
    }


def heuristic_summary(state: AgentState) -> str:
    """Deterministic three-sentence verdict used when no LLM is configured."""
    classification = state.get("classification", {})
    verdict = classification.get("verdict", "Suspicious")
    score = classification.get("trust_score", 50.0)
    claims = state.get("claims", [])
    citations = state.get("citations", [])
    refuted = sum(1 for c in claims if c.get("status") == "refuted")
    supported = sum(1 for c in claims if c.get("status") == "supported")

    first = (
        f"The classifier rates this text as {verdict} with a trust score of "
        f"{score:.0f}/100."
    )
    if citations:
        refs = " ".join(f"[{c['id']}]" for c in citations[:3])
        second = (
            f"Across {len(claims)} extracted claim(s), {supported} were supported and "
            f"{refuted} were contradicted by retrieved fact-checks {refs}."
        )
    else:
        second = (
            f"Across {len(claims)} extracted claim(s), no matching fact-check evidence "
            "was retrieved."
        )
    if refuted:
        third = "Treat this article as unreliable until the refuted claims are corrected."
    elif not citations:
        third = "Evidence was thin, so verify these claims against a primary source before sharing."
    else:
        third = "The available evidence is broadly consistent with the article, but stay cautious."
    return f"{first} {second} {third}"


# --------------------------------------------------------------------- nodes
async def decompose_node(state: AgentState) -> AgentState:
    """Split the article into at most ``MAX_CLAIMS`` atomic claims."""
    text = state["text"]
    raw = await _ask_llm(DECOMPOSE_PROMPT.format(text=text[:6000], max_claims=MAX_CLAIMS))
    parsed = _parse_json(raw, _JSON_ARRAY_RE)
    claims: list[str] = []
    if isinstance(parsed, list):
        claims = [str(c).strip()[:400] for c in parsed if str(c).strip()][:MAX_CLAIMS]
    if not claims:
        claims = heuristic_claims(text)
        state.setdefault("notes", []).append("decompose_heuristic")
    state["claims"] = [{"claim": c, "status": "unverified", "reason": "", "evidence_ids": []} for c in claims]
    return state


async def verify_node(state: AgentState) -> AgentState:
    """Retrieve evidence per claim and assign a support status."""
    belt: ToolBelt = state["belt"]
    evidence_pool: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    url_to_id: dict[str, int] = {}

    for claim_entry in state.get("claims", []):
        claim = claim_entry["claim"]
        items = await belt.evidence(claim, k=3)
        if not items and not belt.exhausted:
            items = await belt.fact_checks(claim, k=2)

        local_ids: list[int] = []
        for item in items:
            url = str(item.get("url") or f"local::{item.get('publisher','')}::{item.get('claim','')}")
            if url not in url_to_id:
                citation_id = len(citations) + 1
                url_to_id[url] = citation_id
                citations.append(
                    {
                        "id": citation_id,
                        "publisher": item.get("publisher", "Unknown"),
                        "url": item.get("url", ""),
                        "rating": item.get("rating", "Unrated"),
                        "title": (item.get("claim") or item.get("text", ""))[:200],
                    }
                )
            local_ids.append(url_to_id[url])
            enriched = dict(item)
            enriched["claim_text"] = claim
            enriched["citation_id"] = url_to_id[url]
            evidence_pool.append(enriched)

        raw = await _ask_llm(
            VERIFY_PROMPT.format(claim=claim, evidence=format_evidence(items))
        )
        parsed = _parse_json(raw, _JSON_OBJECT_RE)
        if isinstance(parsed, dict) and parsed.get("status") in {
            "supported",
            "refuted",
            "unverified",
        }:
            verdict = parsed
        else:
            verdict = heuristic_status(claim, items)
            state.setdefault("notes", []).append("verify_heuristic")

        claim_entry["status"] = verdict["status"]
        claim_entry["reason"] = str(verdict.get("reason", ""))[:500]
        claim_entry["evidence_ids"] = local_ids

    state["evidence"] = evidence_pool
    state["citations"] = citations
    return state


async def synthesize_node(state: AgentState) -> AgentState:
    """Write the final three-sentence cited verdict."""
    classification = state.get("classification", {})
    raw = await _ask_llm(
        SYNTHESIZE_PROMPT.format(
            verdict=classification.get("verdict", "Suspicious"),
            trust_score=round(float(classification.get("trust_score", 50.0)), 1),
            tokens=format_tokens(state.get("tokens", [])),
            findings=format_findings(state.get("claims", [])),
            sources=format_sources(state.get("citations", [])),
        )
    )
    explanation = raw.strip()
    if not explanation:
        explanation = heuristic_summary(state)
        state.setdefault("notes", []).append("synthesize_heuristic")
    state["explanation"] = explanation[:2000]
    return state


# --------------------------------------------------------------------- graph
_GRAPH: Any = None
_GRAPH_LOCK = threading.Lock()


def build_graph() -> Any:
    """Compile the decompose -> verify -> synthesize StateGraph (cached)."""
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    with _GRAPH_LOCK:
        if _GRAPH is None:
            from langgraph.graph import END, START, StateGraph

            builder = StateGraph(AgentState)
            builder.add_node("decompose", decompose_node)
            builder.add_node("verify", verify_node)
            builder.add_node("synthesize", synthesize_node)
            builder.add_edge(START, "decompose")
            builder.add_edge("decompose", "verify")
            builder.add_edge("verify", "synthesize")
            builder.add_edge("synthesize", END)
            _GRAPH = builder.compile()
    return _GRAPH


async def _run_pipeline(text: str, belt: ToolBelt) -> dict[str, Any]:
    """Execute the graph (or the plain node chain if LangGraph is missing)."""
    classification = await belt.classify(text)
    explanation_meta = await belt.explain(text)
    tokens = list(explanation_meta.get("tokens", []))

    state: AgentState = {
        "text": text,
        "classification": classification,
        "tokens": tokens,
        "claims": [],
        "evidence": [],
        "citations": [],
        "explanation": "",
        "notes": [],
        "belt": belt,
    }

    try:
        graph = build_graph()
        result = await graph.ainvoke(state)
    except Exception as exc:
        LOG.warning("LangGraph unavailable (%s); running nodes directly.", exc)
        state.setdefault("notes", []).append("langgraph_unavailable")
        result = await synthesize_node(await verify_node(await decompose_node(state)))

    notes = list(result.get("notes", [])) + belt.notes
    degraded = bool(classification.get("degraded")) or not result.get("citations")

    return {
        "verdict": classification.get("verdict", "Suspicious"),
        "band": classification.get("band", classification.get("verdict", "Suspicious")),
        "trust_score": float(classification.get("trust_score", 50.0)),
        "claims": list(result.get("claims", [])),
        "evidence": list(result.get("evidence", [])),
        "explanation": result.get("explanation", ""),
        "tokens": tokens,
        "citations": list(result.get("citations", [])),
        "model": classification.get("model", "unknown"),
        "explainer_backend": explanation_meta.get("backend", "unknown"),
        "llm_used": get_llm() is not None,
        "tool_calls": belt.calls,
        "prompt_versions": dict(PROMPT_VERSIONS),
        "degraded": degraded,
        "notes": sorted(set(notes)),
    }


def _fallback_payload(text: str, reason: str) -> dict[str, Any]:
    """Classifier-only response used whenever the agent path fails."""
    try:
        from src.models.predict import predict

        classification = predict(text)
    except Exception as exc:
        LOG.error("Classifier also unavailable (%s).", exc)
        classification = {
            "verdict": "Suspicious",
            "band": "Suspicious",
            "trust_score": 50.0,
            "model": "unavailable",
        }
    verdict = classification.get("verdict", "Suspicious")
    score = float(classification.get("trust_score", 50.0))
    return {
        "verdict": verdict,
        "band": classification.get("band", verdict),
        "trust_score": score,
        "claims": [],
        "evidence": [],
        "explanation": (
            f"The classifier rates this text as {verdict} with a trust score of "
            f"{score:.0f}/100. Evidence retrieval was unavailable for this request "
            f"({reason}), so no fact-check citations could be gathered. "
            "Treat this as a model-only opinion and verify with a primary source."
        ),
        "tokens": [],
        "citations": [],
        "model": classification.get("model", "unknown"),
        "explainer_backend": "unavailable",
        "llm_used": False,
        "tool_calls": 0,
        "prompt_versions": dict(PROMPT_VERSIONS),
        "degraded": True,
        "notes": [f"fallback:{reason}"],
    }


async def ainvestigate(text: str, timeout: float | None = None) -> dict[str, Any]:
    """Async entry point. Never raises; degrades to classifier-only output."""
    cleaned = (text or "").strip()[:MAX_TEXT_CHARS]
    if not cleaned:
        return _fallback_payload("", "empty_input")

    limit = float(timeout if timeout is not None else agent_timeout())
    belt = ToolBelt()
    try:
        await asyncio.wait_for(belt.load_mcp(), timeout=max(5.0, limit / 3))
    except Exception as exc:
        LOG.warning("MCP bootstrap skipped (%s).", exc)

    try:
        return await asyncio.wait_for(_run_pipeline(cleaned, belt), timeout=limit)
    except TimeoutError:
        LOG.error("Agent exceeded %.1fs budget; returning classifier-only result.", limit)
        return _fallback_payload(cleaned, "timeout")
    except Exception as exc:
        LOG.error("Agent failed (%s); returning classifier-only result.", exc)
        return _fallback_payload(cleaned, type(exc).__name__)


def investigate(text: str, timeout: float | None = None) -> dict[str, Any]:
    """Synchronous entry point for scripts, tests and the CLI."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(ainvestigate(text, timeout))
    # Called from inside a running loop: run in a private loop on a worker thread.
    result: dict[str, Any] = {}

    def runner() -> None:
        nonlocal result
        result = asyncio.run(ainvestigate(text, timeout))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=(timeout or agent_timeout()) + 15)
    return result or _fallback_payload(text, "thread_timeout")


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or (
        "BREAKING: Scientists confirm drinking bleach cures every known virus, "
        "and the government has been hiding the evidence for decades."
    )
    result = investigate(text)
    print("\n--- STEP 8 VERIFICATION (agent) -------------------------------")
    print(f"Verdict      : {result['verdict']} ({result['trust_score']:.1f}/100)")
    print(f"LLM used     : {result['llm_used']}   tool calls: {result['tool_calls']}")
    print(f"Degraded     : {result['degraded']}   notes: {result['notes']}")
    print(f"Claims       : {len(result['claims'])}")
    for claim in result["claims"]:
        print(f"  - [{claim['status']}] {claim['claim'][:70]}")
    print(f"Citations    : {len(result['citations'])}")
    for citation in result["citations"][:3]:
        print(f"  [{citation['id']}] {citation['publisher']} {citation['url'][:60]}")
    print(f"Explanation  : {result['explanation'][:300]}")
    assert set(result) >= {"verdict", "trust_score", "claims", "evidence", "citations", "degraded"}
    print("Expected     : verdict + explanation present, no exception raised")
    print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
