"""Deterministic analysis assembly and guardrails.

The decision table itself now lives in `app.analysis.verification`. This module:

1. assembles a complete `FinalAnalysis` from the evidence-first status; and
2. enforces the evidence rules on whatever the LLM returns, so a model cannot
   invent source IDs, over-claim confidence, cite itself, or promote an
   unrelated article into supporting evidence.
"""

from __future__ import annotations

from app.analysis.input_parser import ParsedInput
from app.analysis.verification import (
    EvidenceSummary,
    build_structured_explanation,
    compute_scores,
    compute_source_agreement,
    decide_status,
    render_plain_explanation,
    status_to_verdict,
    summarise,
)
from app.schemas import (
    Claim,
    ClaimBreakdownItem,
    FinalAnalysis,
    MLAssessment,
    MLResult,
    SourceAssessmentItem,
    StructuredExplanation,
    VerificationScores,
)
from app.text_utils import truncate

ML_INTERPRETATION = "ML output is a signal, not proof."

# Kept for backward compatibility with existing imports/tests.
WEAK_EVIDENCE_CONFIDENCE_CAP = 55
EvidenceState = EvidenceSummary


def summarise_evidence(
    sources, news_search_ok: bool = True, news_search_error: str | None = None,
    query_count: int = 0,
) -> EvidenceSummary:
    return summarise(sources, news_search_ok, news_search_error, query_count)


def build_source_assessment(
    parsed: ParsedInput, summary: EvidenceSummary
) -> list[SourceAssessmentItem]:
    """One auditable row per retrieved source."""
    from app.analysis.relevance import classify_relation, score_source

    items: list[SourceAssessmentItem] = []
    for source in summary.sources:
        used = source.claim_relation in {
            "SUPPORTS",
            "CONTRADICTS",
            "PARTIALLY_SUPPORTS",
        }
        source.used_in_final_answer = used
        breakdown = score_source(parsed, source)
        _relation, _status, reason = classify_relation(parsed, source, breakdown)
        if source.availability_note:
            reason = f"{reason} {source.availability_note}"
        items.append(
            SourceAssessmentItem(
                source_id=source.source_id,
                publisher=source.publisher,
                title=source.title,
                url=source.url,
                relation=source.claim_relation,
                evidence_status=source.evidence_status,
                reason=reason,
                used_in_final_answer=used,
            )
        )
    return items


def build_limitations(summary: EvidenceSummary, ml: MLResult) -> list[str]:
    limitations: list[str] = []

    if not summary.news_search_ok and summary.news_search_error:
        limitations.append(summary.news_search_error)
    elif not summary.sources:
        limitations.append(
            "No relevant News API sources were found. This is not evidence that "
            "the claim is false."
        )
    elif not (summary.supporting or summary.contradicting):
        limitations.append(
            "No retrieved article was relevant enough to the same event to be "
            "used as evidence. This is not evidence that the claim is false."
        )

    if summary.metadata_only_match:
        limitations.append(
            "At least one relevant result returned metadata only. Full article "
            "text was not available through the search API, so the wording of "
            "the claim could not be checked against the full source."
        )

    if not ml.available:
        limitations.append(
            "The writing-style model was unavailable for this request, so no "
            "model signal contributed to the result."
        )
    else:
        limitations.append(
            "The writing-style model was trained to recognise language patterns, "
            "not factual truth. It cannot confirm that an event happened."
        )
        if not ml.models_agree:
            limitations.append(
                "The ensemble members disagreed with each other, which indicates "
                "the style signal is unreliable for this input."
            )

    limitations.append(
        "News API coverage is partial and its article text is often truncated, "
        "so a genuine story may be missing from the results."
    )
    if summary.unrelated:
        limitations.append(
            f"{len(summary.unrelated)} retrieved article(s) were judged unrelated "
            "and were not used as evidence."
        )
    return limitations


def build_deterministic_analysis(
    parsed: ParsedInput,
    ml: MLResult,
    claims: list[Claim],
    summary: EvidenceSummary,
    reason: str | None = None,
) -> tuple[FinalAnalysis, VerificationScores, StructuredExplanation]:
    """A complete analysis produced without any LLM involvement."""
    status, confidence, rationale = decide_status(summary, ml)
    if reason:
        rationale = f"{rationale} {reason}"

    scores = compute_scores(summary, ml, status, confidence)
    explanation = build_structured_explanation(
        parsed, summary, ml, status, confidence, rationale
    )

    summary_text, _ = truncate(
        (parsed.headline or parsed.raw_text).strip().replace("\n", " "), 180
    )

    analysis = FinalAnalysis(
        verdict=status_to_verdict(status),
        confidence=confidence,
        headline_summary=summary_text or "No text was submitted.",
        plain_language_explanation=render_plain_explanation(explanation),
        ml_assessment=MLAssessment(
            model_name=ml.model_name,
            prediction=ml.prediction,
            confidence=ml.confidence,
            interpretation=ML_INTERPRETATION,
        ),
        claim_breakdown=[
            ClaimBreakdownItem(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                status=c.relation,
                explanation=c.explanation,
                source_ids=list(c.source_ids),
            )
            for c in claims
        ],
        source_assessment=build_source_assessment(parsed, summary),
        source_agreement=compute_source_agreement(summary),
        recommended_action=explanation.recommended_next_step,
        limitations=build_limitations(summary, ml),
        generated_by="DETERMINISTIC_FALLBACK",
    )
    return analysis, scores, explanation


# --- Guardrails on LLM output ----------------------------------------------


def enforce_guardrails(
    analysis: FinalAnalysis,
    summary: EvidenceSummary,
    ml: MLResult,
    claims: list[Claim],
    parsed: ParsedInput,
) -> tuple[FinalAnalysis, list[str]]:
    """Clamp an LLM-produced analysis back inside the evidence rules."""
    warnings: list[str] = []
    valid_ids = {s.source_id for s in summary.sources}

    # 0. `used_in_final_answer` is measured, never asserted by the model.
    for source in summary.sources:
        source.used_in_final_answer = source.claim_relation in {
            "SUPPORTS",
            "CONTRADICTS",
            "PARTIALLY_SUPPORTS",
        }

    # 1. No invented source IDs anywhere.
    for item in analysis.claim_breakdown:
        kept = [sid for sid in item.source_ids if sid in valid_ids]
        if len(kept) != len(item.source_ids):
            warnings.append(
                f"Removed source IDs cited for {item.claim_id or 'a claim'} that "
                "were not part of the retrieved evidence."
            )
        item.source_ids = kept

    # 2. The measured relation always wins over the model's proposal.
    rebuilt = build_source_assessment(parsed, summary)
    llm_by_id = {i.source_id: i for i in analysis.source_assessment}
    for item in rebuilt:
        llm_item = llm_by_id.get(item.source_id)
        if llm_item and llm_item.reason:
            item.reason = llm_item.reason
        if llm_item and llm_item.relation != item.relation:
            warnings.append(
                f"{item.source_id}: the explanation model proposed relation "
                f"'{llm_item.relation}' but the measured relation is "
                f"'{item.relation}'. The measured value was kept."
            )
    extra = set(llm_by_id) - valid_ids
    if extra:
        warnings.append(
            "The explanation model referenced source IDs that do not exist "
            f"({', '.join(sorted(extra))}). They were discarded."
        )
    analysis.source_assessment = rebuilt

    # 3. Measured source agreement is authoritative.
    analysis.source_agreement = compute_source_agreement(summary)

    # 4/5. Status, verdict and confidence may not exceed the evidence.
    status, allowed_confidence, rationale = decide_status(summary, ml)
    allowed_verdict = status_to_verdict(status)

    if analysis.verdict != allowed_verdict:
        if allowed_verdict == "Needs Verification":
            warnings.append(
                f"The explanation model proposed '{analysis.verdict}', but the "
                f"evidence supports only '{status}'. The verdict was corrected."
            )
            analysis.verdict = "Needs Verification"
            analysis.plain_language_explanation = (
                f"{analysis.plain_language_explanation.strip()} Note: {rationale}"
            ).strip()
        elif analysis.verdict == "Needs Verification":
            # The model was more cautious than required - acceptable.
            pass
        else:
            warnings.append(
                f"The explanation model proposed '{analysis.verdict}', which "
                f"conflicts with the measured status '{status}'. The measured "
                "value was kept."
            )
            analysis.verdict = allowed_verdict

    analysis.confidence = max(0, min(100, int(analysis.confidence)))
    if analysis.confidence == 0:
        analysis.confidence = allowed_confidence
    # The evidence sets the ceiling; a small tolerance lets the model nuance it.
    analysis.confidence = min(analysis.confidence, allowed_confidence + 5)
    # Hard safety ceilings keyed off the measured status, not the legacy verdict.
    # "Partially Supported" also maps to the "Needs Verification" verdict, so
    # capping on the verdict alone would wrongly flatten a genuine partial match.
    if status == "Needs Verification":
        analysis.confidence = min(analysis.confidence, 55)
    elif status == "Unable to Verify":
        analysis.confidence = min(analysis.confidence, 50)

    # 6. The ML block always carries the measured numbers and the caveat.
    analysis.ml_assessment = MLAssessment(
        model_name=ml.model_name,
        prediction=ml.prediction,
        confidence=ml.confidence,
        interpretation=ML_INTERPRETATION,
    )

    # 7. Claim breakdown uses measured statuses and citations.
    if claims:
        llm_claims = {i.claim_id: i for i in analysis.claim_breakdown if i.claim_id}
        reconciled: list[ClaimBreakdownItem] = []
        for claim in claims:
            llm_item = llm_claims.get(claim.claim_id)
            explanation = claim.explanation
            if llm_item and llm_item.explanation.strip():
                explanation = llm_item.explanation.strip()
                if set(llm_item.source_ids) != set(claim.source_ids):
                    explanation = f"{explanation} {claim.explanation}".strip()
            if llm_item and llm_item.status != claim.relation:
                warnings.append(
                    f"{claim.claim_id}: the explanation model proposed status "
                    f"'{llm_item.status}' but the measured status is "
                    f"'{claim.relation}'. The measured value was kept."
                )
            reconciled.append(
                ClaimBreakdownItem(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    status=claim.relation,
                    explanation=explanation,
                    source_ids=list(claim.source_ids),
                )
            )
        analysis.claim_breakdown = reconciled

    # 8. Limitations must always be present.
    for limit in build_limitations(summary, ml):
        if limit not in analysis.limitations:
            analysis.limitations.append(limit)

    if not analysis.recommended_action.strip():
        explanation = build_structured_explanation(
            parsed, summary, ml, status, analysis.confidence, rationale
        )
        analysis.recommended_action = explanation.recommended_next_step

    return analysis, warnings
