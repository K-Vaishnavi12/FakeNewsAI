"""End-to-end analysis pipeline.

    input validation and cleaning
        -> ML ensemble
        -> query generation -> News API -> normalization
        -> similarity / claim mapping / verdict
        -> NVIDIA explanation (guardrailed, with deterministic fallback)
        -> provenance assembly

Every stage fails soft: a broken stage adds a system warning and the pipeline
continues, because a partial but honest answer is more useful than an error.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.analysis.claims import extract_claims, map_claims_to_sources
from app.analysis.provenance import build_provenance, build_user_input
from app.analysis.input_parser import parse_input
from app.analysis.relevance import score_sources
from app.analysis.verdict import (
    build_deterministic_analysis,
    compute_source_agreement,
    enforce_guardrails,
    summarise_evidence,
)
from app.config import Settings, get_settings
from app.llm.nvidia_client import NvidiaClient
from app.llm.prompts import SYSTEM_PROMPT, build_user_message
from app.ml.predictor import EnsemblePredictor, get_predictor
from app.news.adapter import NewsAPIAdapter
from app.news.normalize import normalize_articles
from app.news.query_generator import generate_queries_from_parsed
from app.schemas import (
    AnalyzeResponse,
    Claim,
    MLResult,
    NewsSearchResult,
    NewsSource,
    ParsedInputRecord,
)
from app.text_utils import (
    clean_text,
    detect_injection,
    is_probably_non_english,
    truncate,
)

MIN_MEANINGFUL_CHARS = 15


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnalysisPipeline:
    """Orchestrates the full analysis. Dependencies are injectable for tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        predictor: EnsemblePredictor | None = None,
        news_adapter: NewsAPIAdapter | None = None,
        nvidia_client: NvidiaClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.predictor = predictor or get_predictor()
        self.news_adapter = news_adapter or NewsAPIAdapter(self.settings)
        self.nvidia_client = nvidia_client or NvidiaClient(self.settings)

    # --- stages -----------------------------------------------------------

    def _validate(self, raw_text: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        text = clean_text(raw_text or "")

        if len(text) > self.settings.max_input_chars:
            text, _ = truncate(text, self.settings.max_input_chars)
            warnings.append(
                f"The submitted text exceeded {self.settings.max_input_chars} "
                "characters and was truncated before analysis."
            )

        # Prompt-injection attempts are reported but the text is still analysed.
        warnings.extend(detect_injection(text))

        if text and len(text) < MIN_MEANINGFUL_CHARS:
            warnings.append(
                "The submitted text is very short, so entity extraction, search "
                "and classification are all unreliable for this input."
            )

        if text and is_probably_non_english(text):
            warnings.append(
                "The text appears to be mostly non-English. The News API search "
                "is restricted to English-language articles and the ML models "
                "were trained on English text, so both signals are unreliable "
                "for this input."
            )

        return text, warnings

    def _run_ml(self, text: str) -> tuple[MLResult, list[str]]:
        warnings: list[str] = []
        try:
            result = self.predictor.predict(text)
        except Exception:  # noqa: BLE001 - never let the classifier break a request
            warnings.append(
                "The machine-learning ensemble failed to run for this input. "
                "The verdict is based on retrieved evidence alone."
            )
            return (
                MLResult(
                    prediction="UNKNOWN",
                    confidence=0,
                    available=False,
                    note="The ensemble raised an error while scoring this input.",
                ),
                warnings,
            )
        if result.note and not result.available:
            warnings.append(result.note)
        return result, warnings

    def _run_news_search(
        self, parsed, max_sources: int
    ) -> tuple[NewsSearchResult, list[str]]:
        warnings: list[str] = []
        queries = generate_queries_from_parsed(parsed)

        if not queries:
            return (
                NewsSearchResult(ok=True, queries=[], articles=[], error=None),
                warnings,
            )

        try:
            records, error, _code = self.news_adapter.search_many(queries)
        except Exception:  # noqa: BLE001 - belt and braces
            return (
                NewsSearchResult(
                    ok=False,
                    queries=queries,
                    articles=[],
                    error=(
                        "The news search service failed unexpectedly. This "
                        "result should be treated as unverified."
                    ),
                ),
                warnings,
            )

        if not records and error:
            return (
                NewsSearchResult(ok=False, queries=queries, articles=[], error=error),
                warnings,
            )

        if error:
            warnings.append(error)

        sources: list[NewsSource] = normalize_articles(
            records, retrieved_at=_now_iso(), max_sources=max_sources
        )

        # Record how many articles each query contributed.
        counts: dict[str, int] = {}
        for query_id, _query_text, _article in records:
            counts[query_id] = counts.get(query_id, 0) + 1
        for query in queries:
            query.articles_found = counts.get(query.query_id, 0)

        return (
            NewsSearchResult(ok=True, queries=queries, articles=sources, error=None),
            warnings,
        )

    def _explain(
        self,
        parsed,
        ml: MLResult,
        claims: list[Claim],
        summary,
        news_search: NewsSearchResult,
    ):
        """Try NVIDIA, then guardrail it; otherwise fall back deterministically."""
        warnings: list[str] = []
        agreement = compute_source_agreement(summary)

        fallback, scores, structured = build_deterministic_analysis(
            parsed, ml, claims, summary
        )

        user_message = build_user_message(
            user_text=parsed.raw_text,
            ml=ml,
            claims=claims,
            sources=news_search.articles,
            source_agreement=agreement,
            news_search_ok=news_search.ok,
            news_search_error=news_search.error,
            scores=scores,
            structured=structured,
        )

        analysis, error = self.nvidia_client.analyze(SYSTEM_PROMPT, user_message)

        if analysis is None:
            if error:
                warnings.append(error)
            return fallback, scores, structured, warnings

        analysis, guardrail_warnings = enforce_guardrails(
            analysis, summary, ml, claims, parsed
        )
        warnings.extend(guardrail_warnings)

        # Keep the measured scores; only the prose comes from the model.
        scores.verification_confidence = analysis.confidence
        structured.verification_confidence = analysis.confidence
        if analysis.plain_language_explanation.strip():
            structured.what_the_system_found = analysis.plain_language_explanation.strip()
        else:
            analysis.plain_language_explanation = fallback.plain_language_explanation
            warnings.append(
                "The AI explanation was empty, so the rule-based explanation is "
                "shown instead."
            )
        return analysis, scores, structured, warnings

    # --- entry point ------------------------------------------------------

    def analyze(self, raw_text: str, max_sources: int = 10) -> AnalyzeResponse:
        request_id = f"REQ-{uuid.uuid4().hex[:12]}"
        analyzed_at = _now_iso()
        system_warnings: list[str] = []

        text, validation_warnings = self._validate(raw_text)
        system_warnings.extend(validation_warnings)

        user_input = build_user_input(text)
        parsed = parse_input(text)

        publisher_note = ""
        if parsed.publisher_inferred and not parsed.user_urls:
            publisher_note = (
                "Publisher name inferred from the text only. The original "
                "article URL was not supplied."
            )
            system_warnings.append(publisher_note)

        parsed_record = ParsedInputRecord(
            **parsed.as_dict(), publisher_note=publisher_note
        )

        # --- empty input short-circuit ------------------------------------
        if not text.strip():
            ml = MLResult(
                prediction="UNKNOWN",
                confidence=0,
                available=False,
                note="No text was supplied.",
            )
            summary = summarise_evidence([], news_search_ok=True)
            empty_analysis, scores, structured = build_deterministic_analysis(
                parsed, ml, [], summary,
                reason="No text was submitted, so verification could not be completed.",
            )
            empty_analysis.headline_summary = "No text was submitted."
            system_warnings.append(
                "No text was submitted, so verification could not be completed."
            )
            return AnalyzeResponse(
                request_id=request_id,
                analyzed_at=analyzed_at,
                user_input=user_input,
                parsed_input=parsed_record,
                final_analysis=empty_analysis,
                verification_scores=scores,
                structured_explanation=structured,
                claims=[],
                ml_result=ml,
                news_search=NewsSearchResult(ok=True, queries=[], articles=[]),
                source_provenance=build_provenance("", []),
                system_warnings=system_warnings,
            )

        # --- ML (writing-style signal only) -------------------------------
        ml, ml_warnings = self._run_ml(text)
        system_warnings.extend(ml_warnings)

        # --- news search --------------------------------------------------
        news_search, news_warnings = self._run_news_search(parsed, max_sources)
        system_warnings.extend(news_warnings)
        if not news_search.ok and news_search.error:
            system_warnings.append(news_search.error)

        # --- weighted relevance scoring, claims, evidence summary ---------
        sources = list(news_search.articles)
        score_sources(parsed, sources)
        news_search.articles = sources

        claims = map_claims_to_sources(extract_claims(text), sources, parsed)

        summary = summarise_evidence(
            sources,
            news_search_ok=news_search.ok,
            news_search_error=news_search.error,
            query_count=len(news_search.queries),
        )

        # --- explanation --------------------------------------------------
        final_analysis, scores, structured, explain_warnings = self._explain(
            parsed, ml, claims, summary, news_search
        )
        system_warnings.extend(explain_warnings)

        provenance = build_provenance(text, sources)

        return AnalyzeResponse(
            request_id=request_id,
            analyzed_at=analyzed_at,
            user_input=user_input,
            parsed_input=parsed_record,
            final_analysis=final_analysis,  # type: ignore[arg-type]
            verification_scores=scores,
            structured_explanation=structured,
            claims=claims,
            ml_result=ml,
            news_search=news_search,
            source_provenance=provenance,
            system_warnings=system_warnings,
        )


_pipeline: AnalysisPipeline | None = None


def get_pipeline() -> AnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None
