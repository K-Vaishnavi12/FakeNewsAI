"""Evidence-first status decision and score separation (A6 + A7).

Replaces the old "everything lands on 50-55" behaviour with:

* four **separate** scores that are never blended into one misleading number;
* the six-case decision table from the specification;
* a sectioned, humanised explanation.

Guiding rule, unchanged: the machine-learning model is a *writing-style* signal.
It can never on its own make a claim Supported or Contradicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.analysis.input_parser import ParsedInput
from app.news.normalize import domain_of
from app.schemas import (
    MISSING,
    FinalStatus,
    MLResult,
    NewsSource,
    SourceAgreement,
    StructuredExplanation,
    Verdict,
    VerificationScores,
)

ML_DISCLAIMER = (
    "This model evaluates language patterns learned from its training data. It "
    "does not independently verify whether the reported event happened."
)

NO_MATCH_NOTE = (
    "No matching article was retrieved from the configured News API. This does "
    "not prove that the claim is false. The article may be paywalled, recently "
    "published, unavailable to the provider, or indexed under different wording."
)

PARTIAL_TEXT_NOTE = (
    "Publisher match found. Full article verification may be limited because "
    "the source content was not fully available through the search API."
)


@dataclass
class EvidenceSummary:
    """Everything the decision table needs, computed once."""

    sources: list[NewsSource] = field(default_factory=list)
    supporting: list[NewsSource] = field(default_factory=list)
    contradicting: list[NewsSource] = field(default_factory=list)
    partial: list[NewsSource] = field(default_factory=list)
    unrelated: list[NewsSource] = field(default_factory=list)
    news_search_ok: bool = True
    news_search_error: str | None = None
    query_count: int = 0

    @property
    def relevant(self) -> list[NewsSource]:
        return self.supporting + self.contradicting

    @property
    def independent_support_domains(self) -> set[str]:
        return _domains(self.supporting)

    @property
    def independent_contradict_domains(self) -> set[str]:
        return _domains(self.contradicting)

    @property
    def conflict(self) -> bool:
        return bool(self.supporting and self.contradicting)

    @property
    def metadata_only_match(self) -> bool:
        """A relevant source exists but its full text was not retrievable."""
        return any(
            not s.full_text_available
            for s in (self.supporting + self.partial)
        )

    @property
    def best_relevance(self) -> float:
        if not self.sources:
            return 0.0
        return max(s.relevance_score for s in self.sources)


def _domains(sources: list[NewsSource]) -> set[str]:
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


def summarise(
    sources: list[NewsSource],
    news_search_ok: bool = True,
    news_search_error: str | None = None,
    query_count: int = 0,
) -> EvidenceSummary:
    summary = EvidenceSummary(
        sources=list(sources),
        news_search_ok=news_search_ok,
        news_search_error=news_search_error,
        query_count=query_count,
    )
    for source in sources:
        if source.claim_relation == "SUPPORTS":
            summary.supporting.append(source)
        elif source.claim_relation == "CONTRADICTS":
            summary.contradicting.append(source)
        elif source.claim_relation == "PARTIALLY_SUPPORTS":
            summary.partial.append(source)
        else:
            summary.unrelated.append(source)
    return summary


def compute_source_agreement(summary: EvidenceSummary) -> SourceAgreement:
    if not (summary.supporting or summary.contradicting or summary.partial):
        return "NONE"
    if summary.conflict:
        return "LOW"
    if summary.supporting:
        count = len(summary.independent_support_domains)
        if count >= 3:
            return "HIGH"
        if count == 2:
            return "MEDIUM"
        return "LOW"
    if summary.contradicting:
        return "HIGH" if len(summary.independent_contradict_domains) >= 2 else "LOW"
    return "LOW"


def _scale(value: float, low: int, high: int) -> int:
    """Map a 0-1 quality value into an integer confidence band."""
    value = min(max(value, 0.0), 1.0)
    return int(round(low + (high - low) * value))


def decide_status(
    summary: EvidenceSummary, ml: MLResult
) -> tuple[FinalStatus, int, str]:
    """The six-case decision table. Returns `(status, confidence, rationale)`."""

    # --- CASE 5: the search could not be performed at all -----------------
    if not summary.news_search_ok:
        return (
            "Unable to Verify",
            min(45, 50),
            "The news search could not be completed, so no independent evidence "
            "was available. This is not evidence that the claim is false.",
        )

    # --- CASE 6: the ensemble contradicts itself --------------------------
    ml_unreliable = ml.available and not ml.models_agree

    # --- CASE 4: contradicted -------------------------------------------
    if summary.contradicting and not summary.supporting:
        independent = len(summary.independent_contradict_domains)
        quality = max(s.relevance_score for s in summary.contradicting)
        if independent >= 2:
            return (
                "Contradicted by Retrieved Evidence",
                _scale(quality, 70, 95),
                f"{independent} independent publishers cover this story with "
                "denial, correction or fact-check wording, and no retrieved "
                "article reports the claim as stated.",
            )
        return (
            "Needs Verification",
            _scale(quality, 40, 55),
            "One retrieved article appears to contradict the claim, but a "
            "single source is not treated as conclusive.",
        )

    # --- Conflict: supporting and contradicting evidence together ---------
    if summary.conflict:
        return (
            "Needs Verification",
            50,
            "The retrieved sources disagree: some report the claim while others "
            "contain contradicting or corrective wording.",
        )

    # --- CASE 1 / CASE 2: supported --------------------------------------
    if summary.supporting:
        independent = len(summary.independent_support_domains)
        quality = max(s.relevance_score for s in summary.supporting)
        agreement_bonus = min(len(summary.supporting), 4) / 4.0
        blended = 0.7 * quality + 0.3 * agreement_bonus

        if independent >= 2 and not ml_unreliable:
            return (
                "Supported by Retrieved Evidence",
                _scale(blended, 75, 95),
                f"{independent} independent publishers report the same event, "
                "people and place as the submitted claim, with no retrieved "
                "article contradicting it.",
            )

        # One independent publisher, or metadata-only match.
        note = (
            "One independent publisher reports this claim."
            if independent == 1
            else "The retrieved sources are not independent of each other."
        )
        if summary.metadata_only_match:
            note += " " + PARTIAL_TEXT_NOTE
        return (
            "Partially Supported",
            _scale(blended, 55, 74),
            note,
        )

    # --- CASE 2 (weak): only partially relevant coverage -------------------
    if summary.partial:
        quality = max(s.relevance_score for s in summary.partial)
        if summary.metadata_only_match:
            return (
                "Partially Supported",
                _scale(quality, 55, 68),
                "Source metadata matching this story was found, but the full "
                "article text was not available through the search API, so the "
                "specific claim could not be confirmed word for word.",
            )
        return (
            "Partially Supported",
            _scale(quality, 50, 66),
            "The retrieved articles cover the same topic and are shown below as "
            "related evidence, but they do not confirm the specific claim word "
            "for word.",
        )

    # --- CASE 6: models disagree and there is nothing to fall back on -----
    if ml_unreliable:
        return (
            "Needs Verification",
            min(45, 55),
            "The machine-learning ensemble members disagreed with each other, "
            "and no relevant article was retrieved to settle the question.",
        )

    # --- CASE 3: ML has an opinion, evidence has none ---------------------
    if summary.sources:
        return (
            "Needs Verification",
            52,
            "The news search returned related articles, but none of them report "
            "the same specific event closely enough to count as confirmation. "
            "They are still listed below with their links so you can judge them "
            "yourself. " + NO_MATCH_NOTE,
        )

    return (
        "Needs Verification",
        45,
        NO_MATCH_NOTE,
    )


def status_to_verdict(status: FinalStatus) -> Verdict:
    """Map the evidence-first status onto the legacy three-value verdict."""
    if status == "Supported by Retrieved Evidence":
        return "Likely Real"
    if status == "Contradicted by Retrieved Evidence":
        return "Likely Fake"
    return "Needs Verification"


def compute_scores(
    summary: EvidenceSummary, ml: MLResult, status: FinalStatus, confidence: int
) -> VerificationScores:
    """Four separate scores. They are deliberately not blended together."""
    relevant = summary.supporting + summary.contradicting + summary.partial
    evidence_relevance = (
        int(round(max(s.relevance_score for s in relevant) * 100)) if relevant else 0
    )

    agreement = compute_source_agreement(summary)
    agreement_score = {"HIGH": 90, "MEDIUM": 70, "LOW": 40, "NONE": 0}[agreement]

    return VerificationScores(
        ml_style_signal=ml.confidence if ml.available else 0,
        ml_style_direction=ml.prediction,
        evidence_relevance=evidence_relevance,
        source_agreement_score=agreement_score,
        verification_confidence=confidence,
        final_status=status,
        relevant_source_count=len(summary.supporting) + len(summary.contradicting),
        independent_publisher_count=len(
            summary.independent_support_domains | summary.independent_contradict_domains
        ),
        ml_disclaimer=ML_DISCLAIMER,
    )


def build_structured_explanation(
    parsed: ParsedInput,
    summary: EvidenceSummary,
    ml: MLResult,
    status: FinalStatus,
    confidence: int,
    rationale: str,
) -> StructuredExplanation:
    """The sectioned, humanised explanation required by A7."""

    # --- What the system found -------------------------------------------
    found = rationale
    if summary.supporting:
        cited = ", ".join(s.source_id for s in summary.supporting[:4])
        found += f" Supporting sources: {cited}."
    if summary.contradicting:
        cited = ", ".join(s.source_id for s in summary.contradicting[:4])
        found += f" Contradicting sources: {cited}."
    if summary.metadata_only_match:
        found += " " + PARTIAL_TEXT_NOTE

    # --- ML signal --------------------------------------------------------
    if ml.available:
        ml_line = (
            f"{ml.confidence}% {ml.prediction}-style signal"
            f"{'' if ml.models_agree else ' (ensemble members disagreed)'}"
        )
    else:
        ml_line = "The writing-style model was not available for this request."

    # --- Source search ----------------------------------------------------
    if not summary.news_search_ok:
        search_line = (
            f"{summary.query_count} search quer"
            f"{'y was' if summary.query_count == 1 else 'ies were'} prepared, but "
            "the news search service could not be reached."
        )
    else:
        relevant_count = len(summary.supporting) + len(summary.contradicting)
        search_line = (
            f"{summary.query_count} focused quer"
            f"{'y' if summary.query_count == 1 else 'ies'} were run against the "
            f"News API. {len(summary.sources)} article(s) were retrieved, of which "
            f"{relevant_count} were judged relevant to the same event and "
            f"{len(summary.partial)} were only partially related."
        )

    # --- Next step --------------------------------------------------------
    if status == "Supported by Retrieved Evidence":
        next_step = (
            "Open the source links below and read the original reporting "
            "yourself before relying on this claim."
        )
    elif status == "Contradicted by Retrieved Evidence":
        next_step = (
            "Do not share this claim. Read the contradicting articles listed "
            "below in full and look for a published correction or fact-check."
        )
    elif status == "Unable to Verify":
        next_step = (
            "Try again shortly, then verify manually against the original "
            "publisher and at least two independent reputable outlets."
        )
    else:
        next_step = (
            "Open the original publisher link if available and compare the claim "
            "with an official statement or at least two independent reputable "
            "sources."
        )

    if parsed.user_urls:
        next_step = (
            "Open the URL you supplied and confirm it actually contains this "
            "claim, then " + next_step[0].lower() + next_step[1:]
        )

    return StructuredExplanation(
        verdict=status,
        verification_confidence=confidence,
        what_the_system_found=found.strip(),
        ml_text_pattern_signal=ml_line,
        important_limitation=ML_DISCLAIMER,
        source_search=search_line,
        recommended_next_step=next_step,
    )


def render_plain_explanation(explanation: StructuredExplanation) -> str:
    """Flatten the sectioned explanation for consumers that want one string."""
    return (
        f"{explanation.what_the_system_found} "
        f"Writing-style model signal: {explanation.ml_text_pattern_signal}. "
        f"{explanation.important_limitation} "
        f"{explanation.source_search}"
    ).strip()
