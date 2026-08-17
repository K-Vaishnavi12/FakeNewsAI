"""News retrieval: NewsAPI.org with a keyless Google News RSS fallback.

Results are memoised in a short-lived TTL cache (see :mod:`.cache`) keyed by
the full query parameters, so repeated identical claims do not re-hit the
network or burn NewsAPI quota.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests

from .cache import TTLCache
from .config import settings
from .logging_config import get_logger

logger = get_logger(__name__)

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

_REQUEST_TIMEOUT = 15  # seconds

# Module-level cache shared by all callers in this process.
_news_cache = TTLCache(
    ttl_seconds=settings.NEWS_CACHE_TTL_SECONDS,
    max_entries=settings.NEWS_CACHE_MAX_ENTRIES,
)


def _cache_key(endpoint: str, query: str, page_size: int,
               domains: Optional[List[str]]) -> tuple:
    """Build a hashable cache key from the full request parameters.

    Includes the endpoint so NewsAPI and RSS results never collide.
    """
    domain_part = tuple(sorted(domains)) if domains else ()
    return (endpoint, query.strip().lower(), int(page_size), domain_part)


def _fetch_google_news_rss(query: str, page_size: int = 10) -> list:
    """Fetch articles from Google News RSS (free, no API key required).

    Args:
        query: Search phrase.
        page_size: Maximum number of articles to return.

    Returns:
        A list of article dicts shaped like NewsAPI's, so downstream code can
        treat both sources identically. Returns ``[]`` on any failure.
    """
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}

    key = _cache_key(GOOGLE_NEWS_RSS, query, page_size, None)
    cached = _news_cache.get(key)
    if cached is not None:
        logger.debug("RSS cache hit for query=%r", query)
        return cached

    try:
        resp = requests.get(GOOGLE_NEWS_RSS, params=params,
                            timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        # Network/HTTP failure is expected and non-fatal, but must be visible.
        logger.warning("Google News RSS request failed for query=%r",
                       query, exc_info=True)
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        logger.warning("Google News RSS returned malformed XML for query=%r",
                       query, exc_info=True)
        return []

    articles = []
    for item in root.iter("item"):
        if len(articles) >= page_size:
            break

        title_el = item.find("title")
        link_el = item.find("link")
        pub_date_el = item.find("pubDate")

        # Google News titles are usually "Headline - Source Name".
        raw_title = title_el.text if title_el is not None else ""
        source_name = ""
        if " - " in raw_title:
            title, source_name = (p.strip() for p in raw_title.rsplit(" - ", 1))
        else:
            title = raw_title

        description_el = item.find("description")
        desc_text = ""
        if description_el is not None and description_el.text:
            # Descriptions are HTML fragments; strip tags before storing.
            desc_text = re.sub(r"<[^>]+>", "", description_el.text).strip()

        articles.append({
            "title": title,
            "source": {"name": source_name or "Google News"},
            "url": link_el.text if link_el is not None else "",
            "description": desc_text or title,
            "content": desc_text,
            "publishedAt": pub_date_el.text if pub_date_el is not None else "",
        })

    _news_cache.set(key, articles)
    return articles


def _fetch_newsapi(query: str, page_size: int,
                   domains: Optional[List[str]]) -> list:
    """Query NewsAPI.org. Returns ``[]`` if unconfigured or on any failure."""
    api_key = settings.NEWSAPI_KEY
    if not api_key:
        # Expected in dev: we simply fall through to the keyless RSS source.
        logger.debug("NEWSAPI_KEY not set; skipping NewsAPI.")
        return []

    key = _cache_key(NEWSAPI_ENDPOINT, query, page_size, domains)
    cached = _news_cache.get(key)
    if cached is not None:
        logger.debug("NewsAPI cache hit for query=%r", query)
        return cached

    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    if domains:
        params["domains"] = ",".join(domains)

    try:
        resp = requests.get(NEWSAPI_ENDPOINT, params=params,
                            timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except requests.RequestException:
        # Never log `params` -- it contains the API key.
        logger.warning("NewsAPI request failed for query=%r", query,
                       exc_info=True)
        return []
    except ValueError:
        logger.warning("NewsAPI returned non-JSON for query=%r", query,
                       exc_info=True)
        return []

    _news_cache.set(key, articles)
    return articles


def search_news(query: str, domains: Optional[List[str]] = None,
                page_size: int = 10) -> list:
    """Search news articles, preferring NewsAPI and falling back to RSS.

    Args:
        query: Search phrase derived from the user's claim.
        domains: Optional NewsAPI domain filter.
        page_size: Maximum number of articles to return.

    Returns:
        A list of NewsAPI-shaped article dicts (possibly empty).
    """
    if not query or not query.strip():
        return []

    articles = _fetch_newsapi(query, page_size, domains)

    # NewsAPI free tier frequently returns zero rows; RSS is the safety net.
    if not articles:
        articles = _fetch_google_news_rss(query, page_size=page_size)

    logger.info("search_news(query=%r) -> %d article(s)", query, len(articles))
    return articles


def clear_news_cache() -> None:
    """Drop all cached news results. Exposed primarily for tests."""
    _news_cache.clear()
