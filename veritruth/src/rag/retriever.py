"""Step 6c — evidence retrieval over the ChromaDB fact-check collection.

``retrieve_evidence(claim, k=3)`` returns a list of dicts::

    {"text": ..., "publisher": ..., "url": ..., "rating": ..., "score": 0-1, "claim": ...}

If ChromaDB is unavailable or empty, a deterministic keyword-overlap search over
the on-disk corpus is used instead, so retrieval never returns an error to the
agent — only, at worst, weaker evidence.

Run::

    python -m src.rag.retriever "vaccines cause autism"
"""

from __future__ import annotations

import re
import sys
import threading
from typing import Any

from src.config import get_logger

LOG = get_logger("veritruth.rag.retriever")

DEFAULT_K = 3
MAX_K = 10
_WORD_RE = re.compile(r"\b\w{3,}\b", re.UNICODE)
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
    "has", "have", "had", "not", "but", "you", "your", "his", "her", "its",
    "they", "them", "their", "will", "would", "can", "could", "about", "said",
    "says", "who", "what", "when", "where", "why", "how", "all", "any", "been",
}

_LOCK = threading.Lock()
_COLLECTION: Any = None
_COLLECTION_TRIED = False


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")} - _STOPWORDS


def _get_collection():
    """Cached Chroma collection handle, or ``None`` if unavailable/empty."""
    global _COLLECTION, _COLLECTION_TRIED
    if _COLLECTION is not None or _COLLECTION_TRIED:
        return _COLLECTION
    with _LOCK:
        if _COLLECTION is None and not _COLLECTION_TRIED:
            _COLLECTION_TRIED = True
            try:
                from src.rag.ingest_vectors import get_collection

                collection = get_collection()
                if collection.count() == 0:
                    LOG.warning("Collection empty; ingesting corpus now.")
                    from src.rag.ingest_vectors import ingest

                    ingest()
                    collection = get_collection()
                _COLLECTION = collection if collection.count() > 0 else None
            except Exception as exc:
                LOG.warning("ChromaDB retrieval unavailable (%s); keyword fallback.", exc)
                _COLLECTION = None
    return _COLLECTION


def _format(text: str, meta: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "text": (text or "").strip()[:1200],
        "claim": str(meta.get("claim", ""))[:500],
        "rating": str(meta.get("rating", "Unrated")),
        "publisher": str(meta.get("publisher", "Unknown")),
        "url": str(meta.get("url", "")),
        "score": round(float(max(0.0, min(1.0, score))), 4),
    }


def _keyword_fallback(claim: str, k: int) -> list[dict[str, Any]]:
    """Jaccard-style overlap ranking over the JSON corpus."""
    try:
        from src.rag.build_corpus import load_corpus

        records = load_corpus()
    except Exception as exc:
        LOG.error("Corpus unavailable for fallback retrieval (%s).", exc)
        return []

    query = _tokens(claim)
    if not query:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        body = f"Claim: {record.get('claim','')}\nRating: {record.get('rating','')}\nFact-check: {record.get('review','')}"
        overlap = query & _tokens(body)
        if not overlap:
            continue
        score = len(overlap) / max(len(query), 1)
        scored.append((score, _format(body, record, score)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:k]]


def retrieve_evidence(claim: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Top-``k`` fact-check evidence chunks for a claim. Never raises."""
    claim = (claim or "").strip()
    if not claim:
        return []
    k = max(1, min(int(k or DEFAULT_K), MAX_K))

    collection = _get_collection()
    if collection is not None:
        try:
            res = collection.query(
                query_texts=[claim],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            results: list[dict[str, Any]] = []
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = float(dists[i]) if i < len(dists) else 1.0
                results.append(_format(doc, meta or {}, 1.0 - dist))
            if results:
                return results
            LOG.warning("Vector search returned nothing; keyword fallback.")
        except Exception as exc:
            LOG.warning("Vector search failed (%s); keyword fallback.", exc)

    return _keyword_fallback(claim, k)


def evidence_count() -> int:
    """Number of indexed chunks (0 when the vector store is unavailable)."""
    collection = _get_collection()
    if collection is None:
        try:
            from src.rag.build_corpus import load_corpus

            return len(load_corpus())
        except Exception:
            return 0
    try:
        return int(collection.count())
    except Exception:
        return 0


def main() -> None:
    claim = " ".join(sys.argv[1:]).strip() or "vaccines cause autism in children"
    results = retrieve_evidence(claim, k=3)
    print("\n--- STEP 6c VERIFICATION (retriever) --------------------------")
    print(f"Query          : {claim}")
    print(f"Indexed chunks : {evidence_count()}")
    print(f"Results        : {len(results)}")
    for i, item in enumerate(results, 1):
        print(f"  [{i}] score={item['score']:.3f} | {item['publisher']} | {item['rating']}")
        print(f"      {item['text'][:100]}...")
        print(f"      {item['url']}")
    print("Expected       : >=1 result with publisher and URL populated")
    print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
