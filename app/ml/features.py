"""Hand-engineered stylometric features for the tree-based ensemble member.

These are cheap, dependency-free signals about *writing style*: capitalisation,
punctuation intensity, sensational vocabulary, hedging and attribution markers.
They complement the TF-IDF models, which look at lexical content instead.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "urgent", "breaking", "exposed", "secret",
    "secretly", "hoax", "conspiracy", "outrageous", "insane", "destroyed",
    "slams", "blasts", "furious", "panicking", "wake", "alert", "warning",
    "leaked", "banned", "silenced", "coverup", "scandal", "miracle",
    "terrifying", "horrifying", "share", "viral", "truth", "lies", "lie",
}

ATTRIBUTION_WORDS = {
    "said", "according", "reported", "stated", "confirmed", "announced",
    "told", "spokesperson", "statement", "published", "data", "study",
    "survey", "report", "official", "officials", "researchers", "analysis",
}

HEDGE_WORDS = {
    "may", "might", "could", "reportedly", "allegedly", "appears",
    "suggests", "estimated", "approximately", "preliminary",
}

ABSOLUTE_WORDS = {
    "always", "never", "everyone", "nobody", "all", "every", "completely",
    "totally", "entirely", "guaranteed", "proven", "definitely", "100%",
}

FEATURE_NAMES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "uppercase_ratio",
    "all_caps_word_ratio",
    "exclamation_count",
    "question_count",
    "exclamation_ratio",
    "digit_ratio",
    "punctuation_ratio",
    "quote_count",
    "sensational_ratio",
    "attribution_ratio",
    "hedge_ratio",
    "absolute_ratio",
    "repeated_punctuation",
    "title_case_ratio",
    "has_url",
]


def extract_features(text: str) -> list[float]:
    """Return a fixed-length numeric feature vector for one document."""
    text = text or ""
    chars = len(text)
    words = re.findall(r"[\w']+", text)
    word_count = len(words)

    if chars == 0 or word_count == 0:
        return [0.0] * len(FEATURE_NAMES)

    letters = [c for c in text if c.isalpha()]
    upper_letters = sum(1 for c in letters if c.isupper())
    all_caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    lowered = [w.lower() for w in words]

    exclamations = text.count("!")
    questions = text.count("?")
    digits = sum(1 for c in text if c.isdigit())
    punctuation = sum(1 for c in text if c in ".,;:!?-—\"'()")
    quotes = text.count('"') + text.count("“") + text.count("”")
    repeated_punct = len(re.findall(r"[!?]{2,}", text))
    title_case_words = sum(
        1 for w in words if len(w) > 2 and w[0].isupper() and w[1:].islower()
    )
    has_url = 1.0 if re.search(r"https?://", text) else 0.0

    def ratio(vocab: set[str]) -> float:
        return sum(1 for w in lowered if w in vocab) / word_count

    return [
        float(chars),
        float(word_count),
        sum(len(w) for w in words) / word_count,
        (upper_letters / len(letters)) if letters else 0.0,
        all_caps_words / word_count,
        float(exclamations),
        float(questions),
        exclamations / word_count,
        digits / chars,
        punctuation / chars,
        float(quotes),
        ratio(SENSATIONAL_WORDS),
        ratio(ATTRIBUTION_WORDS),
        ratio(HEDGE_WORDS),
        ratio(ABSOLUTE_WORDS),
        float(repeated_punct),
        title_case_words / word_count,
        has_url,
    ]


class StyleFeatureExtractor(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper so the extractor can live inside a Pipeline."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        return np.asarray([extract_features(doc) for doc in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
