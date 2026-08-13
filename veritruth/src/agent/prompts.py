"""Step 8a — versioned prompts for the three agent nodes.

Prompts are versioned so that a change in wording is visible in logs and in the
API response (``prompt_versions`` in the investigate payload).
"""

from __future__ import annotations

DECOMPOSE_VERSION = "decompose/v1"
VERIFY_VERSION = "verify/v1"
SYNTHESIZE_VERSION = "synthesize/v1"

PROMPT_VERSIONS: dict[str, str] = {
    "decompose": DECOMPOSE_VERSION,
    "verify": VERIFY_VERSION,
    "synthesize": SYNTHESIZE_VERSION,
}

MAX_CLAIMS = 4

DECOMPOSE_PROMPT = """You are a fact-checking analyst. Break the news text below into \
at most {max_claims} atomic, independently checkable factual claims.

Rules:
- Each claim must be a single self-contained declarative sentence.
- Resolve pronouns and vague references using the surrounding text.
- Keep the original meaning; never add facts that are not stated or implied.
- Ignore opinion, speculation and rhetorical questions.
- If the text contains no checkable factual claim, return an empty list.

Return ONLY a JSON array of strings, with no markdown fences and no commentary.
Example: ["The mayor resigned on Tuesday.", "The bridge cost $4 billion."]

NEWS TEXT:
\"\"\"{text}\"\"\"
"""

VERIFY_PROMPT = """You are a fact-checking analyst assessing ONE claim against retrieved \
evidence.

CLAIM:
{claim}

EVIDENCE (numbered; may be incomplete, irrelevant or empty):
{evidence}

Decide a status for the claim:
- "supported"    - evidence clearly backs the claim
- "refuted"      - evidence clearly contradicts the claim
- "unverified"   - evidence is absent, off-topic or inconclusive

Return ONLY a JSON object, no markdown fences:
{{"status": "supported|refuted|unverified",
  "reason": "one sentence citing the evidence numbers you relied on",
  "evidence_ids": [1, 2]}}

Never claim support that the evidence does not actually provide. When in doubt,
choose "unverified".
"""

SYNTHESIZE_PROMPT = """You are writing the final verdict for a fake-news investigation.

CLASSIFIER RESULT:
- verdict: {verdict}
- trust score: {trust_score}/100 (higher means more likely genuine)

SUSPICIOUS LANGUAGE FLAGGED BY THE MODEL:
{tokens}

PER-CLAIM FINDINGS:
{findings}

AVAILABLE SOURCES (cite by number):
{sources}

Write EXACTLY three sentences for a general reader:
1. State the overall verdict and what the article claims.
2. Summarise what the evidence showed, citing sources as [1], [2] and so on.
3. State the practical takeaway, and explicitly say so if evidence was thin.

Do not invent sources or citation numbers that are not listed above. Do not use
markdown, headings or bullet points. Return only the three sentences.
"""


def format_evidence(items: list[dict]) -> str:
    """Render retrieved evidence as a numbered block for the verify prompt."""
    if not items:
        return "(no evidence retrieved)"
    lines = []
    for i, item in enumerate(items, 1):
        publisher = item.get("publisher") or "Unknown"
        rating = item.get("rating") or "Unrated"
        text = (item.get("text") or item.get("review") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] ({publisher} | rating: {rating}) {text[:400]}")
    return "\n".join(lines)


def format_sources(citations: list[dict]) -> str:
    """Render the deduplicated citation list for the synthesize prompt."""
    if not citations:
        return "(no sources available)"
    return "\n".join(
        f"[{c.get('id', i)}] {c.get('publisher', 'Unknown')} - {c.get('url', 'no url')}"
        for i, c in enumerate(citations, 1)
    )


def format_findings(claims: list[dict]) -> str:
    """Render per-claim verification results for the synthesize prompt."""
    if not claims:
        return "(no claims were extracted)"
    return "\n".join(
        f"- \"{c.get('claim', '')}\" -> {c.get('status', 'unverified')}: {c.get('reason', '')}"
        for c in claims
    )


def format_tokens(tokens: list[dict]) -> str:
    """Render SHAP token weights for the synthesize prompt."""
    if not tokens:
        return "(no token attributions available)"
    parts = []
    for token in tokens[:10]:
        direction = "toward fake" if float(token.get("weight", 0)) < 0 else "toward real"
        parts.append(f"{token.get('word', '')} ({direction})")
    return ", ".join(parts)
