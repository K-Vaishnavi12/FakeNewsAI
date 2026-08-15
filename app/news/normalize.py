"""Normalization of raw News API payloads into auditable source records.

Guarantees:
* Missing metadata is rendered as "Not provided by the source." - never invented.
* Articles are de-duplicated by *canonical* URL (tracking parameters stripped).
* The original, untouched article URL is preserved for display.
* The query that retrieved each article is stored on the record.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.schemas import MISSING, NewsSource
from app.text_utils import clean_text, truncate

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "cmpid",
    "icid", "ito", "at_medium", "at_campaign", "sh", "smid", "partner",
}

MAX_DESCRIPTION_CHARS = 600
MAX_CONTENT_CHARS = 1200

# Purely a formatting/heuristic aid. This is NOT a reliability rating and is
# always displayed with the "not a fact-check verdict" caveat.
KNOWN_WIRE_DOMAINS = {
    "reuters.com", "apnews.com", "afp.com", "bbc.co.uk", "bbc.com",
    "pti.in", "aninews.in", "bloomberg.com", "ft.com", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "aljazeera.com", "dw.com", "cnbc.com", "thehindu.com",
    "indianexpress.com", "hindustantimes.com", "economictimes.indiatimes.com",
}

QUALITY_HINT_LABEL = "Source quality hint — not a fact-check verdict."


def canonical_url(url: str) -> str:
    """Normalise a URL for de-duplication purposes only."""
    if not url or not isinstance(url, str):
        return ""
    raw = url.strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    except ValueError:
        return raw.lower()

    scheme = "https"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if netloc.endswith(":80") or netloc.endswith(":443"):
        netloc = netloc.rsplit(":", 1)[0]

    path = parts.path.rstrip("/") or "/"

    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    return urlunsplit((scheme, netloc, path, query, ""))


def domain_of(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlsplit(url if "://" in url else f"http://{url}").netloc.lower()
    except ValueError:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.split(":")[0]


def _text_or_missing(value, limit: int | None = None) -> str:
    """Return cleaned text, or the explicit 'missing' sentinel."""
    if value is None:
        return MISSING
    if not isinstance(value, str):
        return MISSING
    cleaned = clean_text(value)
    # NewsAPI truncation marker, e.g. "... [+2130 chars]" - keep but note it.
    if not cleaned or cleaned.lower() in {"null", "none", "[removed]"}:
        return MISSING
    if limit:
        cleaned, _ = truncate(cleaned, limit)
    return cleaned


def _publisher_of(article: dict) -> str:
    source = article.get("source")
    if isinstance(source, dict):
        name = _text_or_missing(source.get("name"))
        if name != MISSING:
            return name
        source_id = _text_or_missing(source.get("id"))
        if source_id != MISSING:
            return source_id
    elif isinstance(source, str):
        return _text_or_missing(source)
    # Last resort: derive from the URL domain, and say so.
    domain = domain_of(article.get("url") or "")
    if domain:
        return f"{domain} (derived from URL)"
    return MISSING


def quality_hint(url: str) -> str:
    """A neutral, clearly-labelled formatting hint. Never a truth score."""
    domain = domain_of(url)
    if not domain:
        return f"{QUALITY_HINT_LABEL} Domain unknown."
    if domain in KNOWN_WIRE_DOMAINS:
        return (
            f"{QUALITY_HINT_LABEL} '{domain}' is on a list of widely-cited "
            "news organisations."
        )
    return (
        f"{QUALITY_HINT_LABEL} '{domain}' is not on the built-in list of "
        "widely-cited outlets; this says nothing about its accuracy."
    )


def _full_text_status(article: dict, content: str, description: str) -> tuple[bool, str]:
    """A4: decide whether the provider returned the whole article body.

    NewsAPI truncates `content` with a "[+1234 chars]" marker, and paywalled or
    licence-restricted articles often return metadata only. Neither case is
    evidence that a claim is false, so the note is deliberately neutral.
    """
    raw_content = article.get("content")
    truncated = bool(
        isinstance(raw_content, str) and re.search(r"\[\+\d+\s*chars\]", raw_content)
    )
    has_content = content != MISSING and len(content) > 400 and not truncated

    if has_content:
        return True, ""

    if content != MISSING or description != MISSING:
        return False, (
            "Relevant source metadata found; full article text was not "
            "available through the API."
        )

    return False, (
        "Only headline-level metadata was returned by the search API for this "
        "result."
    )


def normalize_article(
    article: dict,
    source_id: str,
    query_text: str,
    retrieved_at: str,
    query_id: str = "",
) -> NewsSource:
    """Convert one raw payload into a `NewsSource`."""
    url = article.get("url")
    url_value = _text_or_missing(url)

    description = _text_or_missing(article.get("description"), MAX_DESCRIPTION_CHARS)
    content = _text_or_missing(article.get("content"), MAX_CONTENT_CHARS)
    full_text_available, availability_note = _full_text_status(
        article, content, description
    )

    return NewsSource(
        source_id=source_id,
        source_type="NEWS_API_RESULT",
        publisher=_publisher_of(article),
        title=_text_or_missing(article.get("title")),
        description=description,
        content=content,
        url=url_value,
        author=_text_or_missing(article.get("author")),
        published_at=_text_or_missing(article.get("publishedAt")),
        retrieval_query=query_text or MISSING,
        retrieval_query_id=query_id or MISSING,
        retrieved_at=retrieved_at or MISSING,
        full_text_available=full_text_available,
        availability_note=availability_note,
        source_quality_hint=quality_hint(url_value if url_value != MISSING else ""),
    )


def _dedupe_key(article: dict) -> str:
    """Canonical URL if present, else a title+publisher signature."""
    url = article.get("url")
    canonical = canonical_url(url) if isinstance(url, str) else ""
    if canonical:
        return f"url::{canonical}"
    title = (article.get("title") or "").strip().lower()
    publisher = ""
    source = article.get("source")
    if isinstance(source, dict):
        publisher = (source.get("name") or "").strip().lower()
    if title:
        return f"title::{publisher}::{title}"
    return ""


def normalize_articles(
    records: list[tuple[str, str, dict]],
    retrieved_at: str,
    max_sources: int = 10,
) -> list[NewsSource]:
    """Normalise, de-duplicate and ID a batch of `(query_id, query_text, raw)`.

    The *first* occurrence wins, so the article keeps the query that found it
    first. Records with no usable identity (no URL and no title) are dropped.
    """
    seen: set[str] = set()
    sources: list[NewsSource] = []

    for query_id, query_text, article in records:
        if not isinstance(article, dict):
            continue
        key = _dedupe_key(article)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)

        source_id = f"NEWS-{len(sources) + 1:03d}"
        sources.append(
            normalize_article(article, source_id, query_text, retrieved_at, query_id)
        )
        if len(sources) >= max_sources:
            break

    return sources
