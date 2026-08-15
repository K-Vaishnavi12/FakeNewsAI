"""Real per-token attribution from the linear ensemble members.

The TruthLens interface has a "Model Attribution & Keywords" panel. Rather than
filling it with placeholder numbers, this module derives *genuine* attributions
from the trained Logistic Regression pipeline:

    contribution(token) = tfidf_value(token) * coefficient(token)

A positive contribution pushes the model towards REAL, a negative one towards
FAKE. This is a faithful, exact decomposition for a linear model — the sum of
all token contributions plus the intercept is the decision score.

This is **not** SHAP. It is linear-model coefficient attribution, and it is
labelled as such wherever it is displayed, so the interface does not overstate
what the system actually computes.
"""

from __future__ import annotations

import re

import numpy as np

MAX_TOKENS = 60


def _pipeline_parts(model):
    """Return `(vectorizer, linear_estimator)` or `None` if unsupported."""
    try:
        vectorizer = model.named_steps.get("tfidf")
        estimator = model.named_steps.get("clf")
    except AttributeError:
        return None
    if vectorizer is None or estimator is None:
        return None
    if not hasattr(estimator, "coef_"):
        return None
    return vectorizer, estimator


def token_attributions(model, text: str, max_tokens: int = MAX_TOKENS) -> list[dict]:
    """Attribution weights for the words of `text`, in original reading order.

    Returns a list of ``{"word": str, "weight": float}`` where weight is scaled
    to roughly [-1, 1]. Returns ``[]`` if the model does not expose linear
    coefficients, so callers can degrade gracefully.
    """
    if not text or not text.strip():
        return []

    parts = _pipeline_parts(model)
    if parts is None:
        return []
    vectorizer, estimator = parts

    try:
        matrix = vectorizer.transform([text])
        vocabulary = vectorizer.vocabulary_
        coefficients = np.ravel(estimator.coef_)
    except Exception:  # noqa: BLE001 - attribution must never break a request
        return []

    # Contribution of each *feature* present in this document.
    contributions: dict[str, float] = {}
    row = matrix.tocoo()
    for index, value in zip(row.col, row.data):
        if index >= len(coefficients):
            continue
        contributions[index] = float(value) * float(coefficients[index])

    if not contributions:
        return []

    # Map feature indices back to their n-gram strings.
    index_to_term = {idx: term for term, idx in vocabulary.items()}

    # Distribute each n-gram's contribution over the words it contains, so the
    # UI can highlight individual words in the user's original text.
    word_scores: dict[str, float] = {}
    for index, contribution in contributions.items():
        term = index_to_term.get(index)
        if not term:
            continue
        words = term.split()
        share = contribution / len(words)
        for word in words:
            word_scores[word] = word_scores.get(word, 0.0) + share

    if not word_scores:
        return []

    # Normalise to [-1, 1] using the largest magnitude present.
    largest = max(abs(v) for v in word_scores.values())
    if largest <= 0:
        return []

    # Walk the original text so word order and casing are preserved.
    results: list[dict] = []
    for match in re.finditer(r"[\w'-]+", text):
        original = match.group(0)
        score = word_scores.get(original.lower())
        if score is None:
            # Word was dropped by the vectorizer (stop word, min_df, etc.).
            results.append({"word": original, "weight": 0.0})
        else:
            results.append(
                {"word": original, "weight": round(float(score / largest), 4)}
            )
        if len(results) >= max_tokens:
            break

    return results
