"""Multi-component relevance scoring (A3).

Replaces the single whole-document TF-IDF cosine that caused correct matches to
be under-scored. NewsAPI truncates article `content`, so comparing a full pasted
article against a short description systematically depressed similarity: a
headline reporting the *same event* scored 0.33 against a 0.42 threshold.

The combined score weights the signals that actually indicate "same event":

    relevance = 0.30 * title_similarity
              + 0.25 * claim_similarity
              + 0.20 * entity_overlap
              + 0.10 * date_overlap
              + 0.10 * event_overlap
              + 0.05 * source_metadata_match

Every component is normalised to 0.0-1.0.

Bands:  >=0.75 strongly relevant | 0.55-0.74 relevant
        0.35-0.54 weakly relevant | <0.35 unrelated
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.analysis.input_parser import EVENT_HINTS, ParsedInput
from app.news.normalize import canonical_url, domain_of
from app.schemas import MISSING, ClaimRelation, EvidenceStatus, NewsSource
from app.text_utils import content_tokens, extract_dates, has_negation_cue

STRONGLY_RELEVANT = 0.62
RELEVANT = 0.42
WEAKLY_RELEVANT = 0.20

WEIGHTS = {
    "title": 0.30,
    "claim": 0.25,
    "entity": 0.20,
    "date": 0.10,
    "event": 0.10,
    "source": 0.05,
}

TRUNCATION_MARKER = re.compile(r"\[\+\d+\s*chars\]")


@dataclass
class RelevanceBreakdown:
    """Per-source score components, kept for transparency in the UI."""

    title_similarity: float = 0.0
    claim_similarity: float = 0.0
    entity_overlap: float = 0.0
    date_overlap: float = 0.0
    event_overlap: float = 0.0
    source_metadata_match: float = 0.0
    relevance_score: float = 0.0
    band: str = "UNRELATED"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "title_similarity": self.title_similarity,
            "claim_similarity": self.claim_similarity,
            "entity_overlap": self.entity_overlap,
            "date_overlap": self.date_overlap,
            "event_overlap": self.event_overlap,
            "source_metadata_match": self.source_metadata_match,
            "relevance_score": self.relevance_score,
            "band": self.band,
            "notes": self.notes,
        }


def _clean(text: str) -> str:
    if not text or text == MISSING:
        return ""
    return TRUNCATION_MARKER.sub(" ", text).strip()


def source_text(source: NewsSource) -> str:
    parts = [
        _clean(p) for p in (source.title, source.description, source.content)
    ]
    return " ".join(p for p in parts if p).strip()


def _pairwise_cosine(query: str, documents: list[str]) -> list[float]:
    """Cosine similarity of `query` against each document (word + char n-grams)."""
    if not query.strip():
        return [0.0] * len(documents)
    if not any(d.strip() for d in documents):
        return [0.0] * len(documents)

    corpus = [query] + documents
    try:
        word = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        ).fit_transform(corpus)
        word_scores = cosine_similarity(word[0:1], word[1:]).ravel()
    except ValueError:
        word_scores = np.zeros(len(documents))

    try:
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1
        ).fit_transform(corpus)
        char_scores = cosine_similarity(char[0:1], char[1:]).ravel()
    except ValueError:
        char_scores = np.zeros(len(documents))

    return [
        float(min(max(0.6 * float(w) + 0.4 * float(c), 0.0), 1.0))
        for w, c in zip(word_scores, char_scores)
    ]


def _entity_overlap(parsed: ParsedInput, doc: str) -> float:
    """Fraction of the claim's named entities that reappear in the document."""
    if not parsed.entities or not doc:
        return 0.0
    lowered = doc.lower()
    hits = 0.0
    for entity in parsed.entities:
        key = entity.lower()
        if key in lowered:
            hits += 1.0
            continue
        words = [w for w in key.split() if len(w) > 3]
        if words and sum(1 for w in words if w in lowered) >= max(1, len(words) // 2):
            hits += 0.5
    return round(min(hits / len(parsed.entities), 1.0), 4)


def _date_overlap(parsed: ParsedInput, source: NewsSource, doc: str) -> float:
    """Explicit date match, else proximity of the publication date."""
    if parsed.dates:
        doc_dates = set(extract_dates(doc))
        if source.published_at and source.published_at != MISSING:
            doc_dates |= set(extract_dates(source.published_at))
        if doc_dates:
            matched = sum(1 for d in parsed.dates if d in doc_dates)
            if matched:
                return round(min(matched / len(parsed.dates), 1.0), 4)
        # Year-level agreement is a weaker but still useful signal.
        claim_years = {d for d in parsed.dates if re.fullmatch(r"(?:19|20)\d{2}", d)}
        if claim_years and source.published_at != MISSING:
            if any(y in source.published_at for y in claim_years):
                return 0.6
        return 0.0

    # No date in the claim: a present publication date is neutral-positive,
    # because the article is at least locatable in time.
    if source.published_at and source.published_at != MISSING:
        return 0.5
    return 0.0


def _event_overlap(parsed: ParsedInput, doc: str) -> float:
    """Do the claim and the article describe the same kind of happening?"""
    if not doc:
        return 0.0
    lowered = doc.lower()

    if parsed.event_terms:
        hits = sum(1 for term in parsed.event_terms if term in lowered)
        event_score = hits / len(parsed.event_terms)
    else:
        event_score = 0.0

    # Back off to content-word overlap when no explicit event verb was found.
    claim_kw = set(content_tokens(parsed.raw_text)) - EVENT_HINTS
    doc_kw = set(content_tokens(doc))
    keyword_score = len(claim_kw & doc_kw) / len(claim_kw) if claim_kw else 0.0

    return round(min(max(event_score, keyword_score * 0.9), 1.0), 4)


def _source_metadata_match(parsed: ParsedInput, source: NewsSource) -> tuple[float, list[str]]:
    """URL match by the user beats everything; publisher mention is weak."""
    notes: list[str] = []

    if parsed.user_urls and source.url and source.url != MISSING:
        source_canonical = canonical_url(source.url)
        for url in parsed.user_urls:
            if canonical_url(url) == source_canonical:
                notes.append(
                    "This is the same URL the user supplied in the pasted text."
                )
                return 1.0, notes

    if parsed.publisher and source.publisher and source.publisher != MISSING:
        publisher_key = parsed.publisher.lower()
        if publisher_key in source.publisher.lower():
            notes.append(
                f"Publisher named in the pasted text ('{parsed.publisher}') matches "
                "this result's publisher."
            )
            return 0.85, notes
        domain = domain_of(source.url) if source.url != MISSING else ""
        compact = publisher_key.replace("the ", "").replace(" ", "")
        if domain and compact and compact[:8] in domain.replace(".", ""):
            notes.append(
                "Publisher named in the pasted text appears to match this "
                "result's domain."
            )
            return 0.7, notes

    return 0.0, notes


def score_source(parsed: ParsedInput, source: NewsSource) -> RelevanceBreakdown:
    """Compute the full relevance breakdown for one retrieved article."""
    doc = source_text(source)
    breakdown = RelevanceBreakdown()

    if not doc:
        breakdown.notes.append(
            "The source provided no readable title, description or content, so "
            "its relationship to the claim could not be assessed."
        )
        breakdown.band = "UNKNOWN"
        return breakdown

    title = _clean(source.title)
    query_headline = parsed.headline or parsed.raw_text

    breakdown.title_similarity = round(
        _pairwise_cosine(query_headline, [title or doc])[0], 4
    )

    if parsed.claim_sentences:
        claim_scores = [
            _pairwise_cosine(sentence, [doc])[0] for sentence in parsed.claim_sentences
        ]
        breakdown.claim_similarity = round(max(claim_scores), 4)
    else:
        breakdown.claim_similarity = round(_pairwise_cosine(parsed.raw_text, [doc])[0], 4)

    breakdown.entity_overlap = _entity_overlap(parsed, doc)
    breakdown.date_overlap = _date_overlap(parsed, source, doc)
    breakdown.event_overlap = _event_overlap(parsed, doc)
    source_score, notes = _source_metadata_match(parsed, source)
    breakdown.source_metadata_match = source_score
    breakdown.notes.extend(notes)

    relevance = (
        WEIGHTS["title"] * breakdown.title_similarity
        + WEIGHTS["claim"] * breakdown.claim_similarity
        + WEIGHTS["entity"] * breakdown.entity_overlap
        + WEIGHTS["date"] * breakdown.date_overlap
        + WEIGHTS["event"] * breakdown.event_overlap
        + WEIGHTS["source"] * breakdown.source_metadata_match
    )
    breakdown.relevance_score = round(min(max(relevance, 0.0), 1.0), 4)

    if breakdown.relevance_score >= STRONGLY_RELEVANT:
        breakdown.band = "STRONGLY_RELEVANT"
    elif breakdown.relevance_score >= RELEVANT:
        breakdown.band = "RELEVANT"
    elif breakdown.relevance_score >= WEAKLY_RELEVANT:
        breakdown.band = "WEAKLY_RELEVANT"
    else:
        breakdown.band = "UNRELATED"

    return breakdown


def classify_relation(
    parsed: ParsedInput, source: NewsSource, breakdown: RelevanceBreakdown
) -> tuple[ClaimRelation, EvidenceStatus, str]:
    """Map a relevance breakdown onto a claim relation, with a stated reason."""
    doc = source_text(source)
    if not doc:
        return (
            "UNKNOWN",
            "UNKNOWN",
            "The source provided no readable text, so its relationship to the "
            "claim could not be assessed.",
        )

    score = breakdown.relevance_score
    detail = (
        f"relevance {score:.2f} "
        f"(title {breakdown.title_similarity:.2f}, "
        f"claim {breakdown.claim_similarity:.2f}, "
        f"entities {breakdown.entity_overlap:.2f}, "
        f"event {breakdown.event_overlap:.2f})"
    )

    claim_negated = has_negation_cue(parsed.raw_text)
    doc_negated = has_negation_cue(doc)

    # A denial or fact-check cue on the article side only counts when the
    # article is genuinely about the same story.
    if doc_negated and not claim_negated and score >= WEAKLY_RELEVANT:
        return (
            "CONTRADICTS",
            "CONTRADICTORY",
            f"This article covers the same subject ({detail}) but contains "
            "denial, correction or fact-check wording, which points against the "
            "claim as stated.",
        )

    if score >= RELEVANT:
        return (
            "SUPPORTS",
            "RELEVANT",
            f"This article reports the same event, people and place as the "
            f"submitted claim ({detail}).",
        )

    if score >= WEAKLY_RELEVANT:
        return (
            "PARTIALLY_SUPPORTS",
            "WEAK",
            f"This article is about a related story but does not clearly report "
            f"the specific claim submitted ({detail}).",
        )

    return (
        "UNRELATED",
        "UNRELATED",
        f"Unrelated result — not used as evidence ({detail}).",
    )


def score_sources(
    parsed: ParsedInput, sources: list[NewsSource]
) -> dict[str, RelevanceBreakdown]:
    """Score, classify and re-rank every source. Mutates the sources in place."""
    if not sources:
        return {}

    breakdowns: dict[str, RelevanceBreakdown] = {}
    for source in sources:
        breakdown = score_source(parsed, source)
        relation, status, _reason = classify_relation(parsed, source, breakdown)

        source.text_similarity = breakdown.claim_similarity
        source.relevance_score = breakdown.relevance_score
        source.claim_relation = relation
        source.evidence_status = status
        breakdowns[id(source)] = breakdown

    sources.sort(key=lambda s: s.relevance_score, reverse=True)

    # Re-ID after ranking so NEWS-001 is always the most relevant source.
    remapped: dict[str, RelevanceBreakdown] = {}
    for index, source in enumerate(sources, start=1):
        source.source_id = f"NEWS-{index:03d}"
        remapped[source.source_id] = breakdowns[id(source)]
    return remapped
