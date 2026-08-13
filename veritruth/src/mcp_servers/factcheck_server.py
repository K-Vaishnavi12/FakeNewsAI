"""MCP server #3 — live Google Fact Check Tools API lookup.

Requires ``GOOGLE_FACTCHECK_API_KEY``. When the key is missing or the request
fails, the tool degrades to the locally-cached fact-check corpus instead of
erroring, and flags the result with ``source: "local_corpus"``.

Run standalone::

    python -m src.mcp_servers.factcheck_server
"""


from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config import get_env
from src.mcp_servers._common import clamp_k, clean_text, get_logger

LOG = get_logger("veritruth.mcp.factcheck")

mcp = FastMCP("veritruth-factcheck")


def _from_local(query: str, k: int) -> list[dict[str, Any]]:
    """Keyword search over the cached corpus when the live API is unusable."""
    try:
        from src.rag.retriever import retrieve_evidence

        hits = retrieve_evidence(query, k=k)
    except Exception as exc:
        LOG.error("local fact-check fallback failed: %s", exc)
        return []
    return [
        {
            "claim": hit.get("claim") or hit.get("text", "")[:300],
            "rating": hit.get("rating", "Unrated"),
            "publisher": hit.get("publisher", "Unknown"),
            "url": hit.get("url", ""),
            "review": hit.get("text", ""),
            "source": "local_corpus",
        }
        for hit in hits
    ]


@mcp.tool()
def search_fact_checks(query: str, k: int = 3) -> list[dict]:
    """Search published fact-checks for a claim via the Google Fact Check API.

    Returns up to `k` matching fact-check reviews with the publisher's textual
    rating (e.g. "False", "Mostly true") and a link to the full review. Falls
    back to the local fact-check corpus when the live API is unavailable.
    """
    cleaned = clean_text(query, 2000)
    if not cleaned:
        return []
    k = clamp_k(k)

    if not get_env("GOOGLE_FACTCHECK_API_KEY"):
        LOG.info("GOOGLE_FACTCHECK_API_KEY unset; using local corpus.")
        return _from_local(cleaned, k)

    try:
        from src.rag.build_corpus import fetch_google_factchecks

        records = fetch_google_factchecks(cleaned, page_size=min(k * 2, 20), max_pages=1)
    except Exception as exc:
        LOG.error("live fact-check lookup failed (%s); using local corpus.", exc)
        records = []

    if not records:
        return _from_local(cleaned, k)

    return [
        {
            "claim": rec.get("claim", ""),
            "rating": rec.get("rating", "Unrated"),
            "publisher": rec.get("publisher", "Unknown"),
            "url": rec.get("url", ""),
            "review": rec.get("review", ""),
            "source": "google_factcheck_api",
        }
        for rec in records[:k]
    ]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
