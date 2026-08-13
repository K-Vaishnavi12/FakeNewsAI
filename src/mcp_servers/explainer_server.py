"""MCP server #4 — SHAP/LIME token attributions.

Run standalone::

    python -m src.mcp_servers.explainer_server
"""


from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp_servers._common import clean_text, get_logger

LOG = get_logger("veritruth.mcp.explainer")

mcp = FastMCP("veritruth-explainer")


@mcp.tool()
def explain_prediction(text: str, top_k: int = 10) -> dict[str, Any]:
    """Explain why the classifier scored this text the way it did.

    Returns the most influential tokens with signed weights: a negative weight
    pushes the prediction toward Fake, a positive weight toward Real.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {"tokens": [], "backend": "none", "degraded": True, "error": "empty input"}
    try:
        k = max(1, min(int(top_k), 25))
    except (TypeError, ValueError):
        k = 10
    try:
        from src.explain.shap_explainer import explain_with_meta

        return explain_with_meta(cleaned, top_k=k)
    except Exception as exc:
        LOG.error("explain_prediction failed: %s", exc)
        return {"tokens": [], "backend": "none", "degraded": True, "error": str(exc)[:300]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
