"""Shared fixtures and test doubles.

No test in this suite performs a real network call or loads a real model
artifact. Every external dependency is injected into `AnalysisPipeline`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.pipeline import AnalysisPipeline  # noqa: E402
from app.schemas import MLResult, ModelVote  # noqa: E402


# --- test doubles -----------------------------------------------------------


class FakePredictor:
    """Stands in for the ML ensemble."""

    def __init__(self, result: MLResult | None = None) -> None:
        self.result = result or MLResult(
            model_name="VeritasCheck Ensemble",
            prediction="REAL",
            confidence=80,
            votes=[
                ModelVote(
                    model_name="TF-IDF + Logistic Regression",
                    prediction="REAL",
                    confidence=80,
                )
            ],
            models_agree=True,
            available=True,
        )

    def predict(self, text: str) -> MLResult:
        return self.result

    @property
    def available(self) -> bool:
        return self.result.available

    @property
    def metadata(self) -> dict:
        return {}


class FakeNewsAdapter:
    """Stands in for the News API adapter at the `search_many` boundary."""

    def __init__(
        self,
        articles: list[dict] | None = None,
        error: str | None = None,
        error_code: str | None = None,
        query_text: str = "test query",
    ) -> None:
        self.articles = articles or []
        self.error = error
        self.error_code = error_code
        self.query_text = query_text
        self.calls: list = []

    def search_many(self, queries):
        self.calls.append(queries)
        if self.error and not self.articles:
            return [], self.error, self.error_code
        query_id = queries[0].query_id if queries else "QUERY-001"
        records = [(query_id, self.query_text, a) for a in self.articles]
        return records, self.error, self.error_code


class FakeNvidiaClient:
    """Stands in for the NVIDIA client.

    `mode` controls the failure being simulated:
      - "unavailable": behaves as if the key is missing
      - "malformed":   returns non-JSON text
      - "empty":       returns an empty completion
      - "payload":     returns the supplied JSON payload
    """

    def __init__(self, mode: str = "unavailable", payload: dict | None = None) -> None:
        self.mode = mode
        self.payload = payload
        self.last_user_message: str | None = None

    def analyze(self, system_prompt: str, user_message: str):
        self.last_user_message = user_message

        if self.mode == "payload" and self.payload is not None:
            from app.llm.nvidia_client import coerce_analysis

            return coerce_analysis(self.payload), None
        if self.mode == "malformed":
            return None, (
                "The AI explanation service returned malformed JSON, so a "
                "rule-based explanation was generated instead."
            )
        if self.mode == "empty":
            return None, (
                "The AI explanation service returned an empty response, so a "
                "rule-based explanation was generated instead."
            )
        return None, (
            "The AI explanation service is not configured, so a rule-based "
            "explanation was generated instead."
        )


# --- article factory --------------------------------------------------------


def make_article(
    title: str,
    description: str = "",
    url: str = "https://example.com/story",
    publisher: str = "Example News",
    author: str | None = "A. Reporter",
    published_at: str | None = "2024-05-01T10:00:00Z",
    content: str | None = None,
    include_source: bool = True,
) -> dict:
    """Build a raw NewsAPI-shaped payload."""
    article: dict = {
        "title": title,
        "description": description,
        "content": content if content is not None else description,
        "url": url,
        "author": author,
        "publishedAt": published_at,
    }
    if include_source:
        article["source"] = {"id": None, "name": publisher}
    return article


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings with both integrations deliberately unconfigured."""
    return Settings(
        news_api_key="",
        nvidia_api_key="",
        model_dir=tmp_path / "artifacts",
        max_input_chars=20000,
    )


@pytest.fixture
def build_pipeline(settings):
    """Factory for a pipeline with injected doubles."""

    def _build(
        articles: list[dict] | None = None,
        news_error: str | None = None,
        news_error_code: str | None = None,
        ml_result: MLResult | None = None,
        nvidia_mode: str = "unavailable",
        nvidia_payload: dict | None = None,
    ) -> AnalysisPipeline:
        return AnalysisPipeline(
            settings=settings,
            predictor=FakePredictor(ml_result),
            news_adapter=FakeNewsAdapter(
                articles=articles, error=news_error, error_code=news_error_code
            ),
            nvidia_client=FakeNvidiaClient(nvidia_mode, nvidia_payload),
        )

    return _build
