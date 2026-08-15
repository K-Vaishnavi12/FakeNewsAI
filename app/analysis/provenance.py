"""Assembly of the Source Provenance block.

This is the transparency core of the application: it states, for every piece of
information in the result, exactly where that information came from and what
kind of source it is.

Four source types are kept strictly distinct:

* ``USER_SUBMITTED_TEXT`` - pasted by the user, unverified.
* ``NEWS_API_RESULT``     - retrieved external article (the only real evidence).
* ``MODEL_OUTPUT``        - the ML classifier signal; never proof.
* ``AI_EXPLANATION``      - the NVIDIA interpretation; never an independent source.
"""

from __future__ import annotations

from app.schemas import (
    AIExplanationSource,
    ModelSource,
    NewsSource,
    SourceProvenance,
    UserInput,
    UserSubmittedSource,
)
from app.text_utils import classify_input_type, extract_urls, truncate

MAX_DISPLAY_CHARS = 4000

NEWS_NOTICE = (
    "News API results are related-source evidence. They are not automatically "
    "proof of truth or falsity."
)
NO_SOURCES_NOTICE = (
    "No relevant News API sources were found. The result is based only on the "
    "ML signal and cannot be treated as independently verified."
)
SEARCH_FAILED_NOTICE = (
    "The News API request failed or was rate-limited. Please try again later. "
    "This result should be treated as unverified."
)


def build_user_input(text: str) -> UserInput:
    text = text or ""
    return UserInput(
        text=text,
        character_count=len(text),
        input_type=classify_input_type(text),
        user_supplied_urls=extract_urls(text),
    )


def build_user_source(text: str) -> UserSubmittedSource:
    """The 'User-Submitted Article or Clip' record.

    The URL list is described as *supplied by the user* - we never assert that
    such a URL is genuine or that the page behind it supports the claim.
    """
    text = text or ""
    display, was_truncated = truncate(text, MAX_DISPLAY_CHARS)
    return UserSubmittedSource(
        source_id="USER-001",
        source_type="USER_SUBMITTED_TEXT",
        text=display,
        truncated=was_truncated,
        character_count=len(text),
        input_type=classify_input_type(text),
        user_supplied_urls=extract_urls(text),
    )


def build_provenance(text: str, sources: list[NewsSource]) -> SourceProvenance:
    return SourceProvenance(
        user_submitted_source=build_user_source(text),
        news_api_sources=list(sources),
        model_source=ModelSource(),
        ai_explanation_source=AIExplanationSource(),
    )
