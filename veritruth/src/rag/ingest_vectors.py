"""Step 6b — chunk, embed and upsert the fact-check corpus into ChromaDB.

* splitter : LangChain ``RecursiveCharacterTextSplitter`` (256 chunk, 32 overlap)
* embedder : ``sentence-transformers/all-MiniLM-L6-v2``
* store    : ``chromadb.PersistentClient(path="vectordb/")``, collection ``factchecks``

Every external dependency has a graceful fallback: if LangChain is missing we
use an equivalent local splitter; if sentence-transformers cannot be loaded we
fall back to Chroma's bundled default embedding function.

Run::

    python -m src.rag.ingest_vectors
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

from src.config import (
    CHROMA_COLLECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    VECTORDB_DIR,
    ensure_dirs,
    get_logger,
)

LOG = get_logger("veritruth.rag.ingest_vectors")

# 256 "tokens" ~= 1024 characters; the splitter works in characters.
CHARS_PER_TOKEN = 4
CHUNK_CHARS = CHUNK_SIZE * CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * CHARS_PER_TOKEN
UPSERT_BATCH = 128


def _simple_split(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware character splitter used when LangChain is unavailable."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(text):
        window = text[start : start + size]
        if start + size < len(text):
            cut = max(window.rfind(". "), window.rfind("\n"), window.rfind(" "))
            if cut > size // 2:
                window = window[: cut + 1]
        chunks.append(window.strip())
        start += max(len(window), step) if len(window) >= step else step
    return [c for c in chunks if c]


def get_splitter():
    """LangChain splitter when importable, else a local equivalent."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CHARS,
            chunk_overlap=CHUNK_OVERLAP_CHARS,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    except Exception:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_CHARS,
                chunk_overlap=CHUNK_OVERLAP_CHARS,
            )
        except Exception as exc:
            LOG.warning("LangChain splitter unavailable (%s); using local splitter.", exc)
            return None


def split_text(text: str, splitter: Any = None) -> list[str]:
    if splitter is not None:
        try:
            return [c for c in splitter.split_text(text) if c and c.strip()]
        except Exception as exc:
            LOG.warning("Splitter failed (%s); using local splitter.", exc)
    return _simple_split(text, CHUNK_CHARS, CHUNK_OVERLAP_CHARS)


def get_embedding_function():
    """SentenceTransformer embeddings; ``None`` lets Chroma use its default."""
    try:
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    except Exception as exc:
        LOG.warning(
            "SentenceTransformer embeddings unavailable (%s); using Chroma default.", exc
        )
        return None


def get_client():
    import chromadb

    ensure_dirs()
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTORDB_DIR))


def get_collection(client=None, embedding_function=None):
    """Fetch-or-create the ``factchecks`` collection."""
    client = client or get_client()
    embedding_function = embedding_function or get_embedding_function()
    kwargs: dict[str, Any] = {
        "name": CHROMA_COLLECTION,
        "metadata": {"hnsw:space": "cosine"},
    }
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    try:
        return client.get_or_create_collection(**kwargs)
    except Exception as exc:
        LOG.warning("Collection create with embeddings failed (%s); retrying plain.", exc)
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )


def _doc_id(record: dict[str, Any], index: int) -> str:
    basis = f"{record.get('url', '')}|{record.get('claim', '')}|{index}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:24]


def build_documents(records: list[dict[str, Any]]) -> tuple[list[str], list[dict], list[str]]:
    """Flatten fact-check records into embeddable chunks with metadata."""
    splitter = get_splitter()
    docs: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []

    for record in records:
        claim = str(record.get("claim", "")).strip()
        review = str(record.get("review", "")).strip()
        rating = str(record.get("rating", "Unrated")).strip()
        publisher = str(record.get("publisher", "Unknown")).strip()
        url = str(record.get("url", "")).strip()

        body = f"Claim: {claim}\nRating: {rating}\nFact-check: {review}".strip()
        if len(body) < 10:
            continue

        for chunk_index, chunk in enumerate(split_text(body, splitter)):
            docs.append(chunk)
            metas.append(
                {
                    "claim": claim[:500],
                    "rating": rating[:100],
                    "publisher": publisher[:200],
                    "url": url[:500],
                    "source": str(record.get("source", "unknown"))[:100],
                }
            )
            ids.append(_doc_id(record, len(ids) + chunk_index))
    return docs, metas, ids


def ingest(reset: bool = False) -> dict[str, Any]:
    """Embed the corpus into ChromaDB. Returns a status dict; never raises."""
    from src.rag.build_corpus import load_corpus

    try:
        client = get_client()
    except Exception as exc:
        LOG.error("ChromaDB unavailable (%s).", exc)
        return {"ok": False, "reason": str(exc), "count": 0}

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            LOG.info("Deleted existing collection '%s'.", CHROMA_COLLECTION)
        except Exception:
            pass

    try:
        collection = get_collection(client)
    except Exception as exc:
        LOG.error("Could not open collection (%s).", exc)
        return {"ok": False, "reason": str(exc), "count": 0}

    records = load_corpus()
    docs, metas, ids = build_documents(records)
    if not docs:
        return {"ok": False, "reason": "no documents produced", "count": 0}

    written = 0
    for start in range(0, len(docs), UPSERT_BATCH):
        stop = start + UPSERT_BATCH
        try:
            collection.upsert(
                documents=docs[start:stop],
                metadatas=metas[start:stop],
                ids=ids[start:stop],
            )
            written += len(docs[start:stop])
        except Exception as exc:
            LOG.warning("Upsert batch %s-%s failed (%s); continuing.", start, stop, exc)

    try:
        total = collection.count()
    except Exception:
        total = written

    return {
        "ok": written > 0,
        "reason": "ingested",
        "count": total,
        "chunks_written": written,
        "records": len(records),
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Ingest fact-checks into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Drop the collection first")
    ns = parser.parse_args()

    result = ingest(reset=ns.reset)
    print("\n--- STEP 6b VERIFICATION (vector ingest) ----------------------")
    print(f"Status          : {'OK' if result['ok'] else 'FAILED: ' + result['reason']}")
    print(f"Source records  : {result.get('records', 0)}")
    print(f"Chunks written  : {result.get('chunks_written', 0)}")
    print(f"Collection count: {result.get('count', 0)}  (collection='{CHROMA_COLLECTION}')")
    print(f"Persist dir     : {VECTORDB_DIR}")
    print("Expected        : count > 0 and vectordb/ contains chroma.sqlite3")
    print("---------------------------------------------------------------\n")
    return result


if __name__ == "__main__":
    main()
