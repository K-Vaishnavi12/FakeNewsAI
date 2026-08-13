"""Step 5 — token-level explanations: SHAP first, LIME as automatic fallback.

``explain(text) -> list[{"word": str, "weight": float}]`` returns the top-10
tokens by absolute contribution. Weight sign convention:

* **negative** weight -> token pushes the prediction toward **Fake** (red in UI)
* **positive** weight -> token pushes the prediction toward **Real**

Three strategies are attempted in order; the first that works is cached:

1. ``shap.Explainer`` over the calibrated ``predict_proba`` (model-agnostic).
2. ``lime.lime_text.LimeTextExplainer`` on the same function.
3. A deterministic leave-one-out occlusion analysis (pure numpy, always works).

Run::

    python -m src.explain.shap_explainer "text to explain"
"""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from src.config import MAX_TEXT_CHARS, get_env, get_logger

LOG = get_logger("veritruth.explain")

TOP_K = 10
_MAX_EXPLAIN_CHARS = 1200
_MAX_TOKENS = 120
_TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

_LOCK = threading.Lock()
_BACKEND: str | None = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _prep(text: str) -> str:
    return (text or "").strip()[: min(_MAX_EXPLAIN_CHARS, MAX_TEXT_CHARS)]


def _predict_fn() -> Callable[[list[str]], np.ndarray]:
    """P(REAL) as a 1-D array — the function both SHAP and LIME wrap."""
    from src.models.predict import predict_proba

    def fn(texts: list[str]) -> np.ndarray:
        return np.asarray(predict_proba(list(texts)), dtype=float).ravel()

    return fn


def _predict_fn_2d() -> Callable[[list[str]], np.ndarray]:
    """[P(FAKE), P(REAL)] — LIME requires a 2-column probability matrix."""
    base = _predict_fn()

    def fn(texts: list[str]) -> np.ndarray:
        p = base(list(texts))
        return np.column_stack([1.0 - p, p])

    return fn


def _rank(pairs: list[tuple[str, float]], k: int = TOP_K) -> list[dict[str, Any]]:
    """Merge duplicate tokens, sort by |weight|, round for JSON friendliness."""
    merged: dict[str, float] = {}
    for word, weight in pairs:
        if not word or not word.strip():
            continue
        key = word.strip()
        if not np.isfinite(weight):
            continue
        merged[key] = merged.get(key, 0.0) + float(weight)
    ranked = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [{"word": w, "weight": round(float(v), 6)} for w, v in ranked]


# --------------------------------------------------------------------- SHAP
def _explain_shap(text: str, top_k: int) -> list[dict[str, Any]]:
    import shap

    fn = _predict_fn()
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(fn, masker, algorithm="permutation", silent=True)
    values = explainer([text], max_evals=200, batch_size=8)
    row = values[0]
    tokens = [str(t) for t in np.asarray(row.data).ravel().tolist()]
    weights = np.asarray(row.values, dtype=float).ravel().tolist()
    if not tokens or len(tokens) != len(weights):
        raise ValueError("SHAP returned mismatched tokens/values")
    return _rank(list(zip(tokens, weights, strict=False)), top_k)


# --------------------------------------------------------------------- LIME
def _explain_lime(text: str, top_k: int) -> list[dict[str, Any]]:
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["Fake", "Real"], random_state=42, bow=True)
    exp = explainer.explain_instance(
        text,
        _predict_fn_2d(),
        num_features=top_k,
        num_samples=300,
        labels=(1,),
    )
    return _rank([(w, float(v)) for w, v in exp.as_list(label=1)], top_k)


# ---------------------------------------------------------------- occlusion
def _explain_occlusion(text: str, top_k: int) -> list[dict[str, Any]]:
    """Leave-one-out: how much does P(REAL) drop when a token is removed?

    Zero external dependencies, deterministic, and bounded to ``_MAX_TOKENS``
    model calls — the guaranteed-available last resort.
    """
    fn = _predict_fn()
    tokens = _tokenize(text)[:_MAX_TOKENS]
    if not tokens:
        return []
    base = float(fn([text])[0])
    variants = []
    for i in range(len(tokens)):
        variants.append(" ".join(tokens[:i] + tokens[i + 1 :]))
    probas = fn(variants)
    # removing a token that supported "Real" lowers P(real) -> positive weight
    weights = [base - float(p) for p in np.asarray(probas).ravel().tolist()]
    return _rank(list(zip(tokens, weights, strict=False)), top_k)


_STRATEGIES: list[tuple[str, Callable[[str, int], list[dict[str, Any]]]]] = [
    ("shap", _explain_shap),
    ("lime", _explain_lime),
    ("occlusion", _explain_occlusion),
]


def explain(text: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Top-``top_k`` contributing tokens. Never raises; may return ``[]``."""
    global _BACKEND
    cleaned = _prep(text)
    if len(_tokenize(cleaned)) < 2:
        return []

    preferred = get_env("EXPLAINER_BACKEND", "").lower()
    order = list(_STRATEGIES)
    if preferred in {name for name, _ in _STRATEGIES}:
        order.sort(key=lambda item: item[0] != preferred)
    elif _BACKEND:
        order.sort(key=lambda item: item[0] != _BACKEND)

    for name, strategy in order:
        try:
            result = strategy(cleaned, top_k)
            if result:
                with _LOCK:
                    _BACKEND = name
                return result
            LOG.warning("Explainer '%s' returned nothing; trying next.", name)
        except Exception as exc:
            LOG.warning("Explainer '%s' failed (%s); trying next.", name, exc)
    LOG.error("All explainers failed for input of length %s.", len(cleaned))
    return []


def active_backend() -> str:
    """Name of the explainer that last succeeded (``unknown`` before first call)."""
    return _BACKEND or "unknown"


def explain_with_meta(text: str, top_k: int = TOP_K) -> dict[str, Any]:
    tokens = explain(text, top_k)
    return {"tokens": tokens, "backend": active_backend(), "degraded": not tokens}


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or (
        "SHOCKING: secret government cure exposed by anonymous whistleblower, "
        "doctors furious about this miracle claim."
    )
    result = explain_with_meta(text)
    print("\n--- STEP 5 VERIFICATION (explainability) ----------------------")
    print(f"Backend used : {result['backend']}")
    print(f"Tokens       : {len(result['tokens'])} (expected <= {TOP_K})")
    for item in result["tokens"]:
        arrow = "-> Fake" if item["weight"] < 0 else "-> Real"
        print(f"  {item['word']:<20} {item['weight']:+.5f}  {arrow}")
    print("Expected     : <=10 tokens, weights signed, no exception raised")
    print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
