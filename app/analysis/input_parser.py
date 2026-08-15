"""Structured parsing of a pasted news clip (A1).

Turns raw pasted text into a normalized object that the query generator and the
relevance scorer can both use. Everything here is heuristic and *conservative*:
a publisher is only ever reported as "inferred from the text", never asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import InputType
from app.text_utils import (
    clean_text,
    classify_input_type,
    content_tokens,
    extract_capitalised_phrases,
    extract_dates,
    extract_urls,
    split_sentences,
)

# Publishers whose names commonly appear inside pasted clips. Matching one of
# these only ever produces an *inferred* attribution.
KNOWN_PUBLISHERS = [
    "The New York Times", "New York Times", "The Washington Post",
    "Washington Post", "The Guardian", "Reuters", "Associated Press",
    "The Wall Street Journal", "Wall Street Journal", "Bloomberg",
    "Financial Times", "BBC News", "BBC", "CNN", "NBC News", "CBS News",
    "ABC News", "NPR", "Al Jazeera", "The Hindu", "The Times of India",
    "Hindustan Times", "Indian Express", "The Economic Times", "Deutsche Welle",
    "Politico", "Axios", "The Atlantic", "Time", "Newsweek", "Forbes",
    "The Telegraph", "The Independent", "Sky News", "CNBC", "MarketWatch",
]

BYLINE_PATTERN = re.compile(
    r"^\s*(?:by|written by|reported by)\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})",
    re.IGNORECASE | re.MULTILINE,
)

DATELINE_PATTERN = re.compile(
    r"\b(?:published|updated|posted)(?:\s+on)?\s*:?\s*"
    r"((?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:January|February|March|"
    r"April|May|June|July|August|September|October|November|December)\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")

# Words that add noise to a search query.
NOISE_WORDS = {
    "breaking", "shocking", "urgent", "exclusive", "viral", "unbelievable",
    "must", "read", "watch", "believe", "wow", "omg", "alert", "update",
    "just", "now", "today", "latest", "news", "report", "reports", "reported",
    "says", "said", "according", "amid", "ahead", "video", "photos", "live",
}

# Verb-ish tokens that usually denote the event itself.
EVENT_HINTS = {
    "announced", "announce", "approved", "approve", "launched", "launch",
    "discovered", "discover", "signed", "sign", "banned", "ban", "arrested",
    "resigned", "resign", "elected", "elect", "killed", "died", "won", "lost",
    "raised", "cut", "cuts", "ruled", "ruling", "filed", "sued", "acquired",
    "merged", "recalled", "issued", "issue", "released", "passed", "rejected",
    "confirmed", "denied", "warned", "declared", "unveiled", "opened", "closed",
    "voted", "agreed", "reached", "found", "reports", "decision", "deal",
    "strike", "attack", "protest", "outbreak", "earthquake", "flood", "storm",
}


@dataclass
class ParsedInput:
    """Normalized representation of the user's pasted clip."""

    raw_text: str = ""
    headline: str = ""
    body: str = ""
    publisher: str = ""
    publisher_inferred: bool = False
    author: str = ""
    published_date: str = ""
    user_urls: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    claim_sentences: list[str] = field(default_factory=list)
    event_terms: list[str] = field(default_factory=list)
    input_type: InputType = "UNKNOWN"

    def as_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "headline": self.headline,
            "body": self.body,
            "publisher": self.publisher,
            "publisher_inferred": self.publisher_inferred,
            "author": self.author,
            "published_date": self.published_date,
            "user_urls": self.user_urls,
            "entities": self.entities,
            "dates": self.dates,
            "numbers": self.numbers,
            "claim_sentences": self.claim_sentences,
            "event_terms": self.event_terms,
            "input_type": self.input_type,
        }


def _detect_publisher(text: str) -> tuple[str, bool]:
    """Return `(publisher, inferred)`. Inferred is always True here.

    The publisher is only ever *mentioned* inside pasted text, so it can never
    be treated as verified provenance.
    """
    head = text[:600]
    for name in KNOWN_PUBLISHERS:
        if re.search(rf"\b{re.escape(name)}\b", head, re.IGNORECASE):
            return name, True
    return "", False


def _detect_headline(text: str, sentences: list[str]) -> tuple[str, str]:
    """Split a clip into `(headline, body)`.

    A first line that is short, has no terminal full stop and is not a byline
    behaves like a headline.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "", ""

    first = lines[0]
    looks_like_headline = (
        len(first) <= 160
        and len(lines) > 1
        and not first.endswith(".")
        and not BYLINE_PATTERN.match(first)
        and len(first.split()) >= 3
    )
    if looks_like_headline:
        return first, "\n".join(lines[1:]).strip()

    # No explicit headline line: use the first substantive sentence as the
    # main claim, and keep the whole text as the body.
    if sentences:
        return sentences[0], text
    return first, text


def _score_claim_sentence(sentence: str) -> int:
    """Rank sentences by how checkable they are."""
    score = len(set(content_tokens(sentence)))
    score += 3 * len(NUMBER_PATTERN.findall(sentence))
    score += 2 * len(extract_capitalised_phrases(sentence))
    lowered = sentence.lower()
    score += 2 * sum(1 for w in EVENT_HINTS if w in lowered)
    return score


def parse_input(raw_text: str) -> ParsedInput:
    """Build the normalized input object from pasted text."""
    text = clean_text(raw_text or "")
    if not text:
        return ParsedInput(raw_text="", input_type="UNKNOWN")

    sentences = split_sentences(text)
    headline, body = _detect_headline(text, sentences)
    publisher, inferred = _detect_publisher(text)

    author_match = BYLINE_PATTERN.search(text)
    author = author_match.group(1).strip() if author_match else ""

    date_match = DATELINE_PATTERN.search(text)
    published_date = date_match.group(1).strip() if date_match else ""

    entities = extract_capitalised_phrases(text)
    # Longer entity phrases are far more discriminative.
    entities.sort(key=lambda e: (-len(e.split()), -len(e)))

    lowered = text.lower()
    event_terms = [w for w in EVENT_HINTS if w in lowered]

    claim_sentences = sorted(sentences, key=_score_claim_sentence, reverse=True)[:5]
    if headline and headline not in claim_sentences:
        claim_sentences.insert(0, headline)

    return ParsedInput(
        raw_text=text,
        headline=headline,
        body=body,
        publisher=publisher,
        publisher_inferred=inferred,
        author=author,
        published_date=published_date,
        user_urls=extract_urls(text),
        entities=entities[:12],
        dates=extract_dates(text),
        numbers=NUMBER_PATTERN.findall(text)[:12],
        claim_sentences=claim_sentences[:5],
        event_terms=event_terms[:8],
        input_type=classify_input_type(text),
    )


def clean_query_terms(terms: list[str]) -> list[str]:
    """Drop clickbait/noise words and duplicates while preserving order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for term in terms:
        key = term.lower().strip(" \"'.,;:!?")
        if not key or key in seen or key in NOISE_WORDS:
            continue
        if len(key) < 3 and not key.isdigit():
            continue
        seen.add(key)
        cleaned.append(term.strip(" \"'.,;:!?"))
    return cleaned
