"""Focused search-query generation (A2).

Three targeted queries are built from the parsed clip instead of throwing the
whole article body at the News API:

    QUERY-001  HEADLINE            - the headline or strongest claim sentence
    QUERY-002  ENTITY_EVENT        - main person / organisation / place + event
    QUERY-003  DISTINCTIVE_PHRASE  - 6-12 distinctive content words

Clickbait words, filler adjectives, repeated tokens and stray punctuation are
stripped, because they pull the search towards unrelated tabloid coverage.
"""

from __future__ import annotations

from app.analysis.input_parser import ParsedInput, clean_query_terms, parse_input
from app.schemas import SearchQuery
from app.text_utils import content_tokens

MAX_QUERIES = 3
MAX_QUERY_CHARS = 240
MIN_PHRASE_WORDS = 6
MAX_PHRASE_WORDS = 12


def _trim(query: str) -> str:
    if len(query) <= MAX_QUERY_CHARS:
        return query
    cut = query[:MAX_QUERY_CHARS]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).strip()


def _headline_query(parsed: ParsedInput) -> str:
    """Cleaned headline, or the strongest claim sentence."""
    base = parsed.headline or (
        parsed.claim_sentences[0] if parsed.claim_sentences else parsed.raw_text
    )
    terms = clean_query_terms(base.split())
    # Keep entity casing but cap the length so the query stays focused.
    return _trim(" ".join(terms[:14]))


def _entity_event_query(parsed: ParsedInput) -> str:
    """Main entities plus the event verb(s)."""
    entities = parsed.entities[:3]
    events = parsed.event_terms[:2]

    # Fill with distinctive content words if the clip named no entities.
    if not entities:
        entity_words = clean_query_terms(content_tokens(parsed.raw_text))[:4]
    else:
        entity_words = entities

    parts = clean_query_terms(list(entity_words) + list(events))
    if parsed.dates:
        parts.append(parsed.dates[0])
    return _trim(" ".join(parts[:10]))


def _distinctive_phrase_query(parsed: ParsedInput) -> str:
    """6-12 distinctive content words drawn from the best claim sentence."""
    source_sentence = (
        parsed.claim_sentences[0] if parsed.claim_sentences else parsed.raw_text
    )
    entity_words = {w.lower() for e in parsed.entities[:2] for w in e.split()}

    tokens = clean_query_terms(content_tokens(source_sentence))
    distinctive = [t for t in tokens if t.lower() not in entity_words]

    if len(distinctive) < MIN_PHRASE_WORDS:
        distinctive = clean_query_terms(content_tokens(parsed.raw_text))

    return _trim(" ".join(distinctive[:MAX_PHRASE_WORDS]))


def generate_queries_from_parsed(
    parsed: ParsedInput, max_queries: int = MAX_QUERIES
) -> list[SearchQuery]:
    """Build the typed query set from an already-parsed clip."""
    if not parsed.raw_text.strip():
        return []

    candidates = [
        ("HEADLINE", _headline_query(parsed)),
        ("ENTITY_EVENT", _entity_event_query(parsed)),
        ("DISTINCTIVE_PHRASE", _distinctive_phrase_query(parsed)),
    ]

    seen: set[str] = set()
    queries: list[SearchQuery] = []
    for query_type, query_text in candidates:
        key = query_text.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(
            SearchQuery(
                query_id=f"QUERY-{len(queries) + 1:03d}",
                query_text=query_text,
                query_type=query_type,
            )
        )
        if len(queries) >= max_queries:
            break

    # Guarantee at least one query for very short input.
    if not queries:
        fallback = " ".join(content_tokens(parsed.raw_text)[:6]) or parsed.raw_text
        queries.append(
            SearchQuery(
                query_id="QUERY-001",
                query_text=_trim(fallback),
                query_type="DISTINCTIVE_PHRASE",
            )
        )
    return queries


def generate_queries(text: str, max_queries: int = MAX_QUERIES) -> list[SearchQuery]:
    """Backwards-compatible entry point that parses the text first."""
    return generate_queries_from_parsed(parse_input(text), max_queries=max_queries)
