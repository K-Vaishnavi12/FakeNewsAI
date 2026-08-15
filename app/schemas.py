"""Pydantic schemas for the public API contract.

These mirror the response format required by the specification exactly, so the
frontend and the tests can rely on stable field names.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Controlled vocabularies ------------------------------------------------

Verdict = Literal["Likely Real", "Likely Fake", "Needs Verification"]
MLPrediction = Literal["REAL", "FAKE", "UNKNOWN"]
ClaimRelation = Literal[
    "SUPPORTS", "CONTRADICTS", "PARTIALLY_SUPPORTS", "UNRELATED", "UNKNOWN"
]
ClaimStatus = Literal[
    "SUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED", "UNVERIFIED"
]
EvidenceStatus = Literal["RELEVANT", "WEAK", "CONTRADICTORY", "UNRELATED", "UNKNOWN"]
SourceAgreement = Literal["HIGH", "MEDIUM", "LOW", "NONE"]
InputType = Literal["HEADLINE", "ARTICLE_CLIP", "FULL_ARTICLE", "UNKNOWN"]
SourceType = Literal[
    "USER_SUBMITTED_TEXT", "NEWS_API_RESULT", "MODEL_OUTPUT", "AI_EXPLANATION"
]

# Evidence-first status vocabulary (A6). `verdict` above is retained and derived
# from this, so existing consumers keep working.
FinalStatus = Literal[
    "Supported by Retrieved Evidence",
    "Partially Supported",
    "Needs Verification",
    "Unable to Verify",
    "Contradicted by Retrieved Evidence",
]

RelevanceBand = Literal[
    "STRONGLY_RELEVANT", "RELEVANT", "WEAKLY_RELEVANT", "UNRELATED", "UNKNOWN"
]

MISSING = "Not provided by the source."


# --- Request ----------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Headline or article text supplied by the user.")
    max_sources: int = Field(default=10, ge=1, le=50)


# --- News search ------------------------------------------------------------


class SearchQuery(BaseModel):
    query_id: str
    query_text: str
    query_type: str
    articles_found: int = 0


class RelevanceComponents(BaseModel):
    """Transparent breakdown of how a source's relevance was computed."""

    title_similarity: float = 0.0
    claim_similarity: float = 0.0
    entity_overlap: float = 0.0
    date_overlap: float = 0.0
    event_overlap: float = 0.0
    source_metadata_match: float = 0.0
    relevance_score: float = 0.0
    band: RelevanceBand = "UNRELATED"
    notes: list[str] = Field(default_factory=list)


class NewsSource(BaseModel):
    """Normalized News API record. Missing metadata is never fabricated."""

    source_id: str
    source_type: SourceType = "NEWS_API_RESULT"
    publisher: str = MISSING
    title: str = MISSING
    description: str = MISSING
    content: str = MISSING
    url: str = MISSING
    author: str = MISSING
    published_at: str = MISSING
    retrieval_query: str = MISSING
    retrieval_query_id: str = MISSING
    retrieved_at: str = MISSING
    text_similarity: float = 0.0
    relevance_score: float = 0.0
    claim_relation: ClaimRelation = "UNKNOWN"
    evidence_status: EvidenceStatus = "UNKNOWN"
    used_in_final_answer: bool = False
    # A4: the provider often returns metadata without the full article body
    # (paywalls, licensing). That is never treated as evidence of falsity.
    full_text_available: bool = False
    availability_note: str = ""
    relevance_components: RelevanceComponents = Field(
        default_factory=RelevanceComponents
    )
    # Heuristic only - explicitly not a fact-check verdict.
    source_quality_hint: str = "Source quality hint — not a fact-check verdict."


class NewsSearchResult(BaseModel):
    ok: bool = True
    queries: list[SearchQuery] = Field(default_factory=list)
    articles: list[NewsSource] = Field(default_factory=list)
    error: str | None = None


# --- ML ---------------------------------------------------------------------


class ModelVote(BaseModel):
    model_name: str
    prediction: MLPrediction
    confidence: int = Field(ge=0, le=100)


class TokenAttribution(BaseModel):
    """One word and its signed contribution to the linear model's decision."""

    word: str
    weight: float


class MLResult(BaseModel):
    model_name: str = "VeritasCheck Ensemble"
    prediction: MLPrediction = "UNKNOWN"
    confidence: int = Field(default=0, ge=0, le=100)
    votes: list[ModelVote] = Field(default_factory=list)
    models_agree: bool = True
    interpretation: str = "ML output is a signal, not proof."
    available: bool = True
    note: str | None = None
    token_attributions: list[TokenAttribution] = Field(default_factory=list)
    attribution_method: str = (
        "Logistic Regression coefficient attribution (TF-IDF features)"
    )


# --- Claims -----------------------------------------------------------------


class Claim(BaseModel):
    claim_id: str
    claim_text: str
    source_ids: list[str] = Field(default_factory=list)
    relation: ClaimStatus = "UNVERIFIED"
    explanation: str = ""


# --- Final analysis (LLM or deterministic fallback) --------------------------


class MLAssessment(BaseModel):
    model_name: str = ""
    prediction: MLPrediction = "UNKNOWN"
    confidence: int = Field(default=0, ge=0, le=100)
    interpretation: str = "ML output is a signal, not proof."


class ClaimBreakdownItem(BaseModel):
    claim_id: str = ""
    claim_text: str = ""
    status: ClaimStatus = "UNVERIFIED"
    explanation: str = ""
    source_ids: list[str] = Field(default_factory=list)


class SourceAssessmentItem(BaseModel):
    source_id: str = ""
    publisher: str = MISSING
    title: str = MISSING
    url: str = MISSING
    relation: ClaimRelation = "UNKNOWN"
    evidence_status: EvidenceStatus = "UNKNOWN"
    reason: str = ""
    used_in_final_answer: bool = False


class FinalAnalysis(BaseModel):
    verdict: Verdict = "Needs Verification"
    confidence: int = Field(default=0, ge=0, le=100)
    headline_summary: str = ""
    plain_language_explanation: str = ""
    ml_assessment: MLAssessment = Field(default_factory=MLAssessment)
    claim_breakdown: list[ClaimBreakdownItem] = Field(default_factory=list)
    source_assessment: list[SourceAssessmentItem] = Field(default_factory=list)
    source_agreement: SourceAgreement = "NONE"
    recommended_action: str = ""
    limitations: list[str] = Field(default_factory=list)
    # Provenance of the explanation itself.
    generated_by: Literal["NVIDIA_LLM", "DETERMINISTIC_FALLBACK"] = (
        "DETERMINISTIC_FALLBACK"
    )


class VerificationScores(BaseModel):
    """A6: four separate scores instead of one misleading number.

    `verification_confidence` is confidence in the *verification outcome*, not a
    probability that the claim is true. It is deliberately never called a
    "truth score".
    """

    ml_style_signal: int = Field(default=0, ge=0, le=100)
    ml_style_direction: MLPrediction = "UNKNOWN"
    evidence_relevance: int = Field(default=0, ge=0, le=100)
    source_agreement_score: int = Field(default=0, ge=0, le=100)
    verification_confidence: int = Field(default=0, ge=0, le=100)
    final_status: FinalStatus = "Needs Verification"
    relevant_source_count: int = 0
    independent_publisher_count: int = 0
    ml_disclaimer: str = (
        "This model evaluates language patterns learned from its training data. "
        "It does not independently verify whether the reported event happened."
    )


class StructuredExplanation(BaseModel):
    """A7: the humanised, sectioned explanation."""

    verdict: FinalStatus = "Needs Verification"
    verification_confidence: int = Field(default=0, ge=0, le=100)
    what_the_system_found: str = ""
    ml_text_pattern_signal: str = ""
    important_limitation: str = ""
    source_search: str = ""
    recommended_next_step: str = ""


class ParsedInputRecord(BaseModel):
    """A1: normalized view of the pasted clip."""

    raw_text: str = ""
    headline: str = ""
    body: str = ""
    publisher: str = ""
    publisher_inferred: bool = False
    publisher_note: str = ""
    author: str = ""
    published_date: str = ""
    user_urls: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    claim_sentences: list[str] = Field(default_factory=list)
    event_terms: list[str] = Field(default_factory=list)
    input_type: InputType = "UNKNOWN"


# --- Provenance -------------------------------------------------------------


class UserSubmittedSource(BaseModel):
    source_id: str = "USER-001"
    source_type: SourceType = "USER_SUBMITTED_TEXT"
    text: str = ""
    truncated: bool = False
    character_count: int = 0
    input_type: InputType = "UNKNOWN"
    user_supplied_urls: list[str] = Field(default_factory=list)
    notice: str = (
        "This text was supplied by the user. It was not independently verified "
        "and may be incomplete or edited."
    )


class ModelSource(BaseModel):
    source_id: str = "MODEL-001"
    source_type: SourceType = "MODEL_OUTPUT"
    description: str = "Ensemble ML classification signal, not external evidence."


class AIExplanationSource(BaseModel):
    source_id: str = "AI-001"
    source_type: SourceType = "AI_EXPLANATION"
    description: str = "NVIDIA-generated interpretation of supplied evidence."


class SourceProvenance(BaseModel):
    user_submitted_source: UserSubmittedSource = Field(
        default_factory=UserSubmittedSource
    )
    news_api_sources: list[NewsSource] = Field(default_factory=list)
    model_source: ModelSource = Field(default_factory=ModelSource)
    ai_explanation_source: AIExplanationSource = Field(
        default_factory=AIExplanationSource
    )


class UserInput(BaseModel):
    text: str = ""
    character_count: int = 0
    input_type: InputType = "UNKNOWN"
    user_supplied_urls: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    request_id: str
    analyzed_at: str
    user_input: UserInput
    parsed_input: ParsedInputRecord = Field(default_factory=ParsedInputRecord)
    final_analysis: FinalAnalysis
    verification_scores: VerificationScores = Field(default_factory=VerificationScores)
    structured_explanation: StructuredExplanation = Field(
        default_factory=StructuredExplanation
    )
    claims: list[Claim] = Field(default_factory=list)
    ml_result: MLResult
    news_search: NewsSearchResult
    source_provenance: SourceProvenance
    system_warnings: list[str] = Field(default_factory=list)

    def model_dump_public(self) -> dict[str, Any]:
        """Serialisation helper - no secret ever enters these models."""
        return self.model_dump(mode="json")
