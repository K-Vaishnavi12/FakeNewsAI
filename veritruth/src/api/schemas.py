"""Step 9a — Pydantic v2 request/response schemas for the VeriTruth API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import MAX_TEXT_CHARS

Band = Literal["Real", "Suspicious", "Fake"]
ClaimStatus = Literal["supported", "refuted", "unverified"]


class TextRequest(BaseModel):
    """Any endpoint that takes a block of news text."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="News headline or article body to analyse.",
        examples=["Scientists confirm the moon landing footage was filmed in a studio."],
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ExplainRequest(TextRequest):
    top_k: int = Field(10, ge=1, le=25, description="Number of tokens to return.")


class PredictResponse(BaseModel):
    verdict: Band
    band: Band
    trust_score: float = Field(..., ge=0.0, le=100.0)
    probability_real: float = Field(..., ge=0.0, le=1.0)
    model_name: str = Field(..., alias="model")
    degraded: bool = False

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class TokenWeight(BaseModel):
    word: str
    weight: float


class ExplainResponse(BaseModel):
    tokens: list[TokenWeight] = Field(default_factory=list)
    backend: str = "unknown"
    degraded: bool = False


class Claim(BaseModel):
    claim: str
    status: ClaimStatus = "unverified"
    reason: str = ""
    evidence_ids: list[int] = Field(default_factory=list)


class Evidence(BaseModel):
    text: str = ""
    claim: str = ""
    claim_text: str = ""
    rating: str = "Unrated"
    publisher: str = "Unknown"
    url: str = ""
    score: float = 0.0
    citation_id: int = 0


class Citation(BaseModel):
    id: int
    publisher: str = "Unknown"
    url: str = ""
    rating: str = "Unrated"
    title: str = ""


class InvestigateResponse(BaseModel):
    verdict: Band
    band: Band
    trust_score: float = Field(..., ge=0.0, le=100.0)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    explanation: str = ""
    tokens: list[TokenWeight] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    model_name: str = Field("unknown", alias="model")
    explainer_backend: str = "unknown"
    llm_used: bool = False
    tool_calls: int = 0
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    degraded: bool = False
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    predicted_verdict: Band
    user_verdict: Band
    trust_score: float = Field(50.0, ge=0.0, le=100.0)
    comment: str = Field("", max_length=2000)


class FeedbackResponse(BaseModel):
    id: int
    stored: bool = True
    total: int = 0


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_name: str = Field(..., alias="model")
    model_loaded: bool = False
    evidence_chunks: int = 0
    llm_configured: bool = False
    factcheck_api_configured: bool = False

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    path: str = ""
