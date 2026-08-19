"""Source credibility scoring for live-news corroboration.

The old approach was a single binary flag: ``TRUSTED_SOURCES`` matched a
handful of publisher substrings and anything else was simply not trusted. That
is both too coarse (all non-trusted outlets weigh zero) and too loose (a random
"Time News Blog" matched because the substring ``times`` lives inside it).

This module replaces it with a graded, curated credibility score:

* Tier per source: ``factchecker`` (1.0), ``high`` (0.95), ``medium`` (0.75),
  ``low`` (0.35), ``unknown`` (0.5).
* Matching considers the article's URL *domain* and its *source name*, so a
  spoofed display name is still caught by the real URL host.
* Known fact-checker organisations are enriched from ``server/data/org_stats.csv``
  when that file is present (it ships with the datasets).
* Unknown sources default to a neutral 0.5 rather than 0, so corroboration is
  not silently killed by a publisher we have no data for.

All lookups are pure string ops on static data -- no network calls.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List
from urllib.parse import urlparse

from .logging_config import get_logger

logger = get_logger(__name__)

# Tier labels and their numeric scores. The score is what the corroboration
# logic thresholds on; the label is what gets shown in the UI/API.
TIER_SCORES = {
    "factchecker": 1.0,
    "high": 0.95,
    "medium": 0.75,
    "low": 0.35,
    "unknown": 0.5,
}

# Scores at or above this count a source as "credible enough" to upgrade a
# keyword overlap into a strong corroborating match.
CREDIBLE_THRESHOLD = 0.7

# Major news agencies, flagship national/international newspapers and
# authoritative institutional publishers. Matched by exact name or domain.
_HIGH_CREDIBILITY = frozenset({
    # names
    "reuters", "associated press", "ap", "ap news", "bbc", "bbc news",
    "the new york times", "new york times", "nytimes", "the washington post",
    "washington post", "the guardian", "guardian", "bloomberg", "cnn",
    "npr", "the economist", "economist", "the wall street journal",
    "wall street journal", "wsj", "financial times", "ft", "abc news",
    "nbc news", "cbs news", "pbs", "politico", "axios", "propublica",
    "the atlantic", "time", "usa today", "los angeles times",
    "chicago tribune", "wired", "the verge", "techcrunch", "arstechnica",
    "al jazeera", "dw", "deutsche welle", "france 24", "sky news",
    "the hill", "military.com", "nature", "science", "scientific american",
    "new scientist", "smithsonian", "live science", "phys.org", "nasa",
    "cnbc",
    # domains
    "reuters.com", "apnews.com", "ap.org", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "guardian.co.uk", "bloomberg.com", "cnn.com", "npr.org",
    "economist.com", "wsj.com", "ft.com", "abcnews.go.com",
    "nbcnews.com", "cbsnews.com", "pbs.org", "politico.com",
    "axios.com", "propublica.org", "theatlantic.com", "time.com",
    "usatoday.com", "latimes.com", "chicagotribune.com", "wired.com",
    "theverge.com", "techcrunch.com", "arstechnica.com",
    "aljazeera.com", "dw.com", "france24.com", "news.sky.com",
    "thehill.com", "military.com", "nature.com", "science.org",
    "scientificamerican.com", "newscientist.com", "smithsonianmag.com",
    "livescience.com", "phys.org", "nasa.gov", "cnbc.com",
})

# Dedicated fact-checking organisations. A fact-checker confirming a claim is
# the single strongest signal the pipeline has, so they sit at score 1.0.
_FACT_CHECKERS = frozenset({
    "snopes", "snopes.com", "politifact", "politifact.com", "factcheck.org",
    "factcheck", "afp fact check", "afp", "reuters fact check",
    "ap fact check", "full fact", "fullfact.org", "africa check",
    "africacheck.org", "boom", "boomlive", "check your fact",
    "checkyourfact.com", "dfrac", "dfrac.org", "dubawa", "dubawa.org",
    "lead stories", "alt news", "the quint", "first draft", "factly",
    "factly.in", "mediawise", "pesacheck", "logically", "newsmobile",
    "teyt", "vishvas news", "storyful", "the wire", "ghana fact",
    "factcheck.afp.com", "factchecker", "fact check",
})

# Established-but-softer outlets and mainstream aggregators. They are
# trustworthy enough to count toward corroboration, but score below the top
# agencies.
_MEDIUM_CREDIBILITY = frozenset({
    "the independent", "independent.co.uk", "the telegraph",
    "telegraph.co.uk", "huffpost", "huffington post", "huffpost.com",
    "buzzfeed news", "buzzfeednews.com", "vox", "vox.com", "slate",
    "salon", "business insider", "businessinsider.com", "marketwatch",
    "yahoo news", "google news", "msn", "msn.com", "the register",
    "gizmodo", "engadget", "macrumors", "daily mail", "dailymail",
    "mirror", "the sun", "new york post", "nypost", "the daily beast",
    "the intercept", "mother jones", "theweek", "quartz", "qz.com",
    "fortune", "fortune.com", "fast company", "mashable", "venturebeat",
    "news.google.com", "news.google.co.uk",
})

# Sources with a documented record of misinformation, heavy editorial bias or
# fabricated content. These actively *reduce* a corroboration match's value.
_LOW_CREDIBILITY = frozenset({
    "infowars", "infowars.com", "naturalnews", "naturalnews.com",
    "zerohedge", "zerohedge.com", "breitbart", "breitbart.com",
    "the gateway pundit", "gatewaypundit", "gatewaypundit.com",
    "worldtruth.tv", "beforeitsnews", "yournewswire", "prisonplanet",
    "globalresearch", "wnd", "worldnetdaily", "wnd.com",
    "conservative daily", "patribotics", "activist post",
})

_UNKNOWN_LABEL = "unknown"

# Domains that merely *wrap* other publishers' content (Google News RSS links
# all point at news.google.com). They carry no signal about the publisher
# itself, so they neither corroborate nor contradict a name match.
_WRAPPER_DOMAINS = frozenset({
    "news.google.com", "news.google.co.uk", "news.google.de",
    "news.google.fr", "news.google.in", "news.google.ca",
    "feedproxy.google.com", "feedburner.google.com",
})

# Sorted longest-first so substring matching prefers the most specific token
# ("the new york times" before "times") and cannot fire on tiny fragments.
_ALL_TOKENS: List[str] = sorted(
    _FACT_CHECKERS | _HIGH_CREDIBILITY | _MEDIUM_CREDIBILITY
    | _LOW_CREDIBILITY,
    key=len,
    reverse=True,
)

_csv_fact_checkers: frozenset = frozenset()
_csv_loaded = False


def _load_fact_checkers_from_csv() -> frozenset:
    """Read organisation names from ``org_stats.csv``, if it is present.

    The FACTors-derived ``org_stats.csv`` ships with the datasets and lists
    real fact-checking bodies. Any organisation found there is promoted to the
    fact-checker tier. Absence of the file is expected (datasets are
    gitignored) and simply yields no extra entries.
    """
    global _csv_fact_checkers, _csv_loaded
    if _csv_loaded:
        return _csv_fact_checkers
    _csv_loaded = True

    try:
        import pandas as pd

        from .paths import DATA_DIR
    except Exception:
        return _csv_fact_checkers

    path = os.path.join(DATA_DIR, "org_stats.csv")
    if not os.path.isfile(path):
        return _csv_fact_checkers

    try:
        df = pd.read_csv(path)
        names = {
            str(v).strip().lower()
            for v in df["organisation"].dropna().tolist()
            if str(v).strip() and len(str(v).strip()) > 1
        }
        _csv_fact_checkers = frozenset(names)
        logger.info("Enriched credibility with %d fact-checkers from org_stats.csv",
                    len(_csv_fact_checkers))
    except Exception:
        logger.warning("Could not parse org_stats.csv for fact-checker names",
                       exc_info=True)

    return _csv_fact_checkers


def _normalize(value) -> str:
    """Lowercase and collapse whitespace for matching."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _domain_from_url(url) -> str:
    """Return the bare host of ``url`` (www. stripped), or ``''``."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def get_source_credibility(source_name=None, url=None) -> dict:
    """Score a news source on a 0-1 credibility scale.

    Args:
        source_name: Publisher display name (string or ``{'name': str}``).
        url: Article URL, used for domain matching. Optional but recommended.

    Returns:
        A dict with ``score`` (float 0-1), ``tier``, ``label``,
        ``matched_on`` and ``is_fact_checker``.
    """
    if isinstance(source_name, dict):
        source_name = source_name.get("name") or ""
    name = _normalize(source_name)
    domain = _domain_from_url(url)
    candidates = [c for c in (name, domain) if c]

    _load_fact_checkers_from_csv()
    tiers = [
        ("factchecker", _FACT_CHECKERS | _csv_fact_checkers),
        ("high", _HIGH_CREDIBILITY),
        ("medium", _MEDIUM_CREDIBILITY),
        ("low", _LOW_CREDIBILITY),
    ]

    # 1. Exact match on either the source name or the URL domain. Exact is
    #    authoritative: "Reuters" + reuters.com is unambiguously Reuters.
    for candidate in candidates:
        for tier, lookup in tiers:
            if candidate in lookup:
                return _result(tier, candidate, name)

    # 2. Substring match on the name only. A publisher calling itself
    #    "Reuters Breaking News" is probably Reuters -- but a random blog with
    #    "times" in its title is not "The Times". Require the URL domain to
    #    corroborate the brand: it must exactly match a token of the same
    #    tier, or be absent / a known wrapper (no evidence either way).
    #    Anything else is treated as a spoofed display name and demoted.
    for tier, lookup in tiers:
        for token in _ALL_TOKENS:
            if token in lookup and _contains_word(name, token):
                if domain and domain not in lookup \
                        and domain not in _WRAPPER_DOMAINS:
                    return _result(_UNKNOWN_LABEL, None, name)
                return _result(tier, token, name)

    return _result(_UNKNOWN_LABEL, None, name)


def _contains_word(text: str, token: str) -> bool:
    """True if ``token`` appears in ``text`` on word boundaries."""
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


def _result(tier: str, matched_on, name: str) -> dict:
    return {
        "score": TIER_SCORES[tier],
        "tier": tier,
        "label": tier.capitalize(),
        "matched_on": matched_on,
        "is_fact_checker": tier == "factchecker",
        "source_name": name or None,
    }