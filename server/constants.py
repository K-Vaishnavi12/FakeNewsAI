"""Single source of truth for the lexicons shared by the agent and the
ML prediction layer.

Previously these lists were duplicated (and had drifted out of sync) between
``agent.py`` and ``ml/predict.py``. Centralising them means a change to, say,
the trusted-source list takes effect everywhere at once.

All entries are lowercase; callers are expected to lowercase their input
before matching.
"""

# Publishers treated as authoritative when scoring news corroboration.
# Matched as substrings against a source name, so keep entries distinctive.
TRUSTED_SOURCES = frozenset({
    "reuters", "ap", "apnews", "associated press", "bbc", "cnn", "nasa",
    "space", "phys.org", "nature", "science", "smithsonian", "bloomberg",
    "times", "guardian", "npr", "the hill", "military.com", "live science",
    "forbes", "washington post", "nbc", "cbs", "abc", "pbs", "techcrunch",
    "wired",
})

# Attribution / sourcing language typical of professional reporting.
# Presence of these nudges the stylometric score toward "real".
JOURNALISTIC_CUES = frozenset({
    "confirmed", "announced", "reported", "stated", "ruled", "published",
    "spokesman", "spokesperson", "officials", "authorities", "agency",
    "department", "court order", "investigators", "according to",
    "researchers", "scientists", "study published", "press release",
    "white house", "pentagon", "supreme court", "reuters",
    "associated press", "bbc", "nasa",
})

# Clickbait / conspiracy vocabulary. Presence nudges the score toward "fake".
SENSATIONALIST_CUES = frozenset({
    "shocking", "leaked", "secret plot", "deep state", "conspiracy", "cabal",
    "unbelievable", "you won't believe", "miracle cure",
    "they don't want you to know", "exposed", "insane", "disturbing",
    "explodes", "humiliated", "destroys",
})

# Words that, if they surface as top model signals, justify an explicit
# "clickbait vocabulary" red flag in the final report.
CLICKBAIT_SIGNAL_WORDS = frozenset({
    "video", "shocking", "breaking", "leaked", "secret", "cabal", "bunker",
})

# Used only by the heuristic fallback when the joblib model is unavailable.
FALLBACK_REAL_INDICATORS = frozenset({
    "reuters", "associated press", "apnews", "bbc", "cnn", "nytimes",
    "washington post", "bloomberg", "official", "statement", "announced",
})

FALLBACK_FAKE_INDICATORS = frozenset({
    "shocking", "conspiracy", "illuminati", "hoax", "secret plot", "deepfake",
    "miracle cure", "unbelievable", "fake", "debunked",
})

# Terms stripped when turning a user claim into a news search query.
SEARCH_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "could", "may", "might", "must", "can",
    "that", "which", "who", "whom", "this", "these", "those", "it", "its",
    "they", "them", "their", "breaking", "news", "just", "alert", "exclusive",
    "report", "reports", "unbelievable", "shocking", "claim", "claims",
    "latest", "today", "yesterday",
})

# Lightweight stopword set for token-overlap corroboration scoring.
OVERLAP_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "news", "breaking",
})

# Acronyms that are legitimately all-caps and must not count as "shouting".
ALLOWED_ACRONYMS = frozenset({
    "USA", "NASA", "UN", "FBI", "CIA", "EU", "WHO", "UK", "US",
})

# Canonical label ordering used when persisting a model bundle.
# Index == the integer class label produced by the training pipeline.
CLASS_LABELS = ("fake", "real")
