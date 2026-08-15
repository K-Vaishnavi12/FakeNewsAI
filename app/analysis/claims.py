"""Claim extraction and claim-to-source mapping.

The user's input is split into a small number of checkable claims. Each claim is
then scored independently against every retrieved source, so the final answer can
cite exactly which article supports which statement.
"""

from __future__ import annotations

import re

from app.analysis.input_parser import ParsedInput, parse_input
from app.analysis.relevance import RELEVANT, WEAKLY_RELEVANT, score_source, source_text
from app.news.normalize import domain_of
from app.schemas import MISSING, Claim, ClaimStatus, NewsSource
from app.text_utils import content_tokens, has_negation_cue, split_sentences, truncate

MAX_CLAIMS = 5
MAX_CLAIM_CHARS = 400


def extract_claims(text: str) -> list[Claim]:
    """Split the input into at most `MAX_CLAIMS` checkable statements."""
    text = (text or "").strip()
    if not text:
        return []

    sentences = split_sentences(text)

    # A headline or a single statement is one claim.
    if len(sentences) <= 1:
        claim_text, _ = truncate(text, MAX_CLAIM_CHARS)
        return [
            Claim(
                claim_id="CLAIM-001",
                claim_text=claim_text,
                relation="UNVERIFIED",
                explanation="",
            )
        ]

    # Prefer sentences that carry checkable specifics: names, numbers, dates.
    def specificity(sentence: str) -> int:
        score = len(set(content_tokens(sentence)))
        score += 3 * len(re.findall(r"\b\d[\d,.]*\b", sentence))
        score += 2 * len(re.findall(r"\b[A-Z][a-z]{2,}", sentence))
        return score

    ranked = sorted(sentences, key=specificity, reverse=True)[:MAX_CLAIMS]
    # Restore the original reading order so the breakdown follows the article.
    ordered = [s for s in sentences if s in ranked][:MAX_CLAIMS]

    claims: list[Claim] = []
    for idx, sentence in enumerate(ordered, start=1):
        claim_text, _ = truncate(sentence.strip(), MAX_CLAIM_CHARS)
        claims.append(
            Claim(
                claim_id=f"CLAIM-{idx:03d}",
                claim_text=claim_text,
                relation="UNVERIFIED",
                explanation="",
            )
        )
    return claims


def independent_domains(sources: list[NewsSource]) -> set[str]:
    """Distinct publisher domains - a proxy for source independence."""
    domains: set[str] = set()
    for source in sources:
        if source.url and source.url != MISSING:
            domain = domain_of(source.url)
            if domain:
                domains.add(domain)
                continue
        if source.publisher and source.publisher != MISSING:
            domains.add(source.publisher.lower())
    return domains


def map_claims_to_sources(
    claims: list[Claim], sources: list[NewsSource], parsed: ParsedInput | None = None
) -> list[Claim]:
    """Attach supporting/contradicting source IDs and a status to each claim.

    Each claim is scored against every source with the same weighted relevance
    model used for the overall verdict, so a claim is only cited against an
    article that is about the same event.
    """
    if not claims:
        return claims

    if not sources:
        for claim in claims:
            claim.source_ids = []
            claim.relation = "UNVERIFIED"
            claim.explanation = (
                "No News API article was retrieved for this statement, so it "
                "could not be checked against independent reporting. Absence of "
                "a matching article is not evidence that the statement is false."
            )
        return claims

    for claim in claims:
        # Score this individual claim, reusing the parsed entities of the whole
        # clip so short claim sentences still benefit from the article context.
        claim_parsed = parse_input(claim.claim_text)
        if parsed is not None:
            claim_parsed.entities = parsed.entities or claim_parsed.entities
            claim_parsed.dates = parsed.dates or claim_parsed.dates
            claim_parsed.user_urls = parsed.user_urls
            claim_parsed.publisher = parsed.publisher

        supporting: list[str] = []
        partial: list[str] = []
        contradicting: list[str] = []

        for source in sources:
            doc = source_text(source)
            if not doc:
                continue
            breakdown = score_source(claim_parsed, source)
            score = breakdown.relevance_score
            doc_negated = has_negation_cue(doc)
            claim_negated = has_negation_cue(claim.claim_text)

            if doc_negated and not claim_negated and score >= WEAKLY_RELEVANT:
                contradicting.append(source.source_id)
            elif score >= RELEVANT:
                supporting.append(source.source_id)
            elif score >= WEAKLY_RELEVANT:
                partial.append(source.source_id)

        claim.source_ids = supporting + partial + contradicting
        claim.relation, claim.explanation = _decide_claim_status(
            supporting, partial, contradicting, sources
        )

    return claims


def _decide_claim_status(
    supporting: list[str],
    partial: list[str],
    contradicting: list[str],
    sources: list[NewsSource],
) -> tuple[ClaimStatus, str]:
    """Apply the strict evidence rules to one claim."""
    by_id = {s.source_id: s for s in sources}
    supporting_sources = [by_id[i] for i in supporting if i in by_id]
    contradicting_sources = [by_id[i] for i in contradicting if i in by_id]

    support_domains = independent_domains(supporting_sources)
    contradict_domains = independent_domains(contradicting_sources)

    if contradicting and not supporting:
        return (
            "CONTRADICTED",
            f"{len(contradicting)} retrieved article(s) covering this subject "
            f"({', '.join(contradicting)}) contain denial, correction or "
            "fact-check wording, and no retrieved article directly supports the "
            "statement.",
        )

    if supporting and contradicting:
        return (
            "PARTIALLY_SUPPORTED",
            f"Sources disagree: {', '.join(supporting)} appear to report this "
            f"statement while {', '.join(contradicting)} contain contradicting "
            "or corrective wording. Independent checking is required.",
        )

    # Rule: one article is never definitive confirmation.
    if len(support_domains) >= 2:
        return (
            "SUPPORTED",
            f"{len(support_domains)} independent publishers "
            f"({', '.join(supporting)}) report the same core statement.",
        )

    if supporting:
        return (
            "PARTIALLY_SUPPORTED",
            f"Only one independent publisher ({', '.join(supporting)}) appears "
            "to report this statement. A single source is not treated as "
            "confirmation.",
        )

    if partial:
        return (
            "PARTIALLY_SUPPORTED",
            f"{', '.join(partial)} cover a related topic but do not clearly "
            "report this specific statement.",
        )

    return (
        "UNVERIFIED",
        "None of the retrieved articles are close enough to this statement to "
        "support or contradict it. This is not evidence that the statement is "
        "false; it means the search returned no usable evidence.",
    )
