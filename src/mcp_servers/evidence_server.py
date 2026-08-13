"""MCP server #2 — evidence retrieval (ChromaDB RAG).

Run standalone::

    python -m src.mcp_servers.evidence_server
"""


from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp_servers._common import clamp_k, clean_text, get_logger

LOG = get_logger("veritruth.mcp.evidence")

mcp = FastMCP("veritruth-evidence")


@mcp.tool()
def search_evidence(claim: str, k: int = 3) -> list[dict]:
    """Retrieve fact-check evidence chunks for a claim.

    Performs semantic search over the local ChromaDB `factchecks` collection and
    returns up to `k` passages, each with its publisher, rating and source URL.
    """
    cleaned = clean_text(claim, 2000)
    if not cleaned:
        return []
    try:
        from src.rag.retriever import retrieve_evidence

        return list(retrieve_evidence(cleaned, k=clamp_k(k)))
    except Exception as exc:
        LOG.error("search_evidence failed: %s", exc)
        return []


@mcp.tool()
def evidence_stats() -> dict[str, Any]:
    """Report how many fact-check chunks are currently indexed."""
    try:
        from src.rag.retriever import evidence_count

        count = int(evidence_count())
        return {"indexed_chunks": count, "available": count > 0}
    except Exception as exc:
        LOG.error("evidence_stats failed: %s", exc)
        return {"indexed_chunks": 0, "available": False, "error": str(exc)[:300]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
