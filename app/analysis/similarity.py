"""Similarity and relevance scoring between the user's claim and each source.

Two complementary signals are combined:

* `text_similarity` - TF-IDF cosine similarity over character and word n-grams.
  Robust to paraphrase and to the short, truncated text the News API returns.
* `relevance_score` - an entity/date/number overlap score that checks whether the
  article is actually about the *same* event, people, place and time, rather than
  merely using similar words.

A high cosine with a low entity overlap is exactly the "related but different
coverage" case the specification requires us to distinguish, and it is scored
down accordingly.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import MISSING, ClaimRelation, EvidenceStatus, NewsSource
from app.text_utils import (
    content_tokens,
    extract_capitalised_phrases,
    extract_dates,
    has_negation_cue,
)

# Decision thresholds. Deliberately conservative: it is better to return
# "Needs Verification" than to over-claim support.
SUPPORT_SIMILARITY = 0.42
PARTIAL_SIMILARITY = 0.22
UNRELATED_SIMILARITY = 0.12
ENTITY_OVERLAP_FLOOR = 0.25


def source_text(source: NewsSource) -> str:
    """Concatenate the usable text of a source, skipping missing fields."""
    parts = [
        p
        for p in (source.title, source.description, source.content)
        if p and p != MISSING
    ]
    # Strip the NewsAPI truncation marker so it does not pollute the vectorizer.
    joined = " ".join(parts)
    return re.sub(r"\[\+\d+\s*chars\]", " ", joined).strip()


def cosine_similarities(claim_text: str, documents: list[str]) -> list[float]:
    """TF-IDF cosine similarity of `claim_text` against each document."""
    usable = [d for d in documents if d.strip()]
    if not claim_text.strip() or not usable:
        return [0.0] * len(documents)

    corpus = [claim_text] + documents
    scores: list[float] = [0.0] * len(documents)

    try:
        word_vec = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        word_matrix = word_vec.fit_transform(corpus)
        word_scores = cosine_similarity(word_matrix[0:1], word_matrix[1:]).ravel()
    except ValueError:
        word_scores = np.zeros(len(documents))

    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1
        )
        char_matrix = char_vec.fit_transform(corpus)
        char_scores = cosine_similarity(char_matrix[0:1], char_matrix[1:]).ravel()
    except ValueError:
        char_scores = np.zeros(len(documents))

    for idx in range(len(documents)):
        if not documents[idx].strip():
            scores[idx] = 0.0
            continue
        combined = 0.65 * float(word_scores[idx]) + 0.35 * float(char_scores[idx])
        scores[idx] = round(min(max(combined, 0.0), 1.0), 4)
    return scores


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d[\d,.]*\b", text))


def entity_overlap(claim_text: str, doc_text: str) -> float:
    """How much of the claim's specific detail reappears in the document."""
    if not claim_text.strip() or not doc_text.strip():
        return 0.0

    claim_entities = {e.lower() for e in extract_capitalised_phrases(claim_text)}
    doc_lower = doc_text.lower()

    # Entities: count a match if the whole phrase or any distinctive word hits.
    entity_hits = 0
    for entity in claim_entities:
        if entity in doc_lower:
            entity_hits += 1
            continue
        words = [w for w in entity.split() if len(w) > 3]
        if words and any(w in doc_lower for w in words):
            entity_hits += 0.5
    entity_score = entity_hits / len(claim_entities) if claim_entities else 0.0

    # Keyword overlap as a fallback signal for entity-free claims.
    claim_kw = set(content_tokens(claim_text))
    doc_kw = set(content_tokens(doc_text))
    keyword_score = (
        len(claim_kw & doc_kw) / len(claim_kw) if claim_kw else 0.0
    )

    claim_dates = set(extract_dates(claim_text))
    date_score = (
        len(claim_dates & set(extract_dates(doc_text))) / len(claim_dates)
        if claim_dates
        else 0.0
    )

    claim_nums = _numbers(claim_text)
    number_score = (
        len(claim_nums & _numbers(doc_text)) / len(claim_nums) if claim_nums else 0.0
    )

    # Weight whichever specific signals actually exist in the claim.
    components = [(entity_score, 0.45), (keyword_score, 0.35)]
    if claim_dates:
        components.append((date_score, 0.1))
    if claim_nums:
        components.append((number_score, 0.1))

    total_weight = sum(w for _, w in components)
    score = sum(v * w for v, w in components) / total_weight
    return round(min(max(score, 0.0), 1.0), 4)


def classify_relation(
    claim_text: str,
    doc_text: str,
    similarity: float,
    overlap: float,
) -> tuple[ClaimRelation, EvidenceStatus, str]:
    """Decide how one source relates to the claim, with a stated reason."""
    if not doc_text.strip():
        return (
            "UNKNOWN",
            "UNKNOWN",
            "The source provided no readable title, description or content, so "
            "its relationship to the claim could not be assessed.",
        )

    claim_negated = has_negation_cue(claim_text)
    doc_negated = has_negation_cue(doc_text)
    # A denial/debunk cue on the article side, but not on the claim side, is the
    # signature of a contradiction or fact-check.
    contradiction_signal = doc_negated and not claim_negated

    if similarity < UNRELATED_SIMILARITY and overlap < ENTITY_OVERLAP_FLOOR:
        return (
            "UNRELATED",
            "UNRELATED",
            f"Low text similarity ({similarity:.2f}) and low overlap of names, "
            f"places and figures ({overlap:.2f}); this article appears to be "
            "about a different subject.",
        )

    if contradiction_signal and (similarity >= PARTIAL_SIMILARITY or overlap >= 0.4):
        return (
            "CONTRADICTS",
            "CONTRADICTORY",
            f"The article covers the same subject (similarity {similarity:.2f}, "
            f"detail overlap {overlap:.2f}) but contains denial or correction "
            "wording, which points against the claim as stated.",
        )

    if similarity >= SUPPORT_SIMILARITY and overlap >= ENTITY_OVERLAP_FLOOR:
        return (
            "SUPPORTS",
            "RELEVANT",
            f"High text similarity ({similarity:.2f}) together with matching "
            f"names, places or figures ({overlap:.2f}) indicates this article "
            "reports the same core claim.",
        )

    if similarity >= PARTIAL_SIMILARITY or overlap >= 0.45:
        if overlap < ENTITY_OVERLAP_FLOOR:
            return (
                "PARTIALLY_SUPPORTS",
                "WEAK",
                f"Wording is similar ({similarity:.2f}) but the specific names, "
                f"places, dates or figures largely do not match ({overlap:.2f}), "
                "so this may be related coverage of a different event.",
            )
        return (
            "PARTIALLY_SUPPORTS",
            "WEAK",
            f"The article touches on the same topic (similarity {similarity:.2f}, "
            f"detail overlap {overlap:.2f}) but does not clearly report the "
            "specific claim submitted.",
        )

    return (
        "UNRELATED",
        "UNRELATED",
        f"Similarity ({similarity:.2f}) and detail overlap ({overlap:.2f}) are "
        "both too low to treat this article as evidence about the claim.",
    )


def score_sources(claim_text: str, sources: list[NewsSource]) -> list[NewsSource]:
    """Populate similarity, relevance, relation and evidence status in place."""
    if not sources:
        return sources

    documents = [source_text(s) for s in sources]
    similarities = cosine_similarities(claim_text, documents)

    for source, doc, similarity in zip(sources, documents, similarities):
        overlap = entity_overlap(claim_text, doc)
        relation, status, _reason = classify_relation(
            claim_text, doc, similarity, overlap
        )
        source.text_similarity = similarity
        # Relevance blends "is it about the same thing" with raw similarity.
        source.relevance_score = round(0.6 * overlap + 0.4 * similarity, 4)
        source.claim_relation = relation
        source.evidence_status = status

    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    # Re-ID after sorting so NEWS-001 is always the most relevant source.
    for idx, source in enumerate(sources, start=1):
        source.source_id = f"NEWS-{idx:03d}"
    return sources


def relation_reason(claim_text: str, source: NewsSource) -> str:
    """Recompute the human-readable reason for a scored source."""
    doc = source_text(source)
    overlap = entity_overlap(claim_text, doc)
    _relation, _status, reason = classify_relation(
        claim_text, doc, source.text_similarity, overlap
    )
    return reason
