"""MCP server #1 — classifier.

Exposes the calibrated fake-news classifier as an MCP tool over stdio.

Run standalone::

    python -m src.mcp_servers.classifier_server
"""


from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp_servers._common import clean_text, get_logger

LOG = get_logger("veritruth.mcp.classifier")

mcp = FastMCP("veritruth-classifier")


@mcp.tool()
def classify_news(text: str) -> dict[str, Any]:
    """Classify a news headline or article as Real / Suspicious / Fake.

    Returns the verdict, a calibrated trust score from 0 (certainly fake) to
    100 (certainly real), the raw P(real), and the backing model name.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "verdict": "Suspicious",
            "band": "Suspicious",
            "trust_score": 50.0,
            "probability_real": 0.5,
            "model": "none",
            "degraded": True,
            "error": "empty input",
        }
    try:
        from src.models.predict import predict

        result = dict(predict(cleaned))
        result.setdefault("degraded", False)
        return result
    except Exception as exc:  # never propagate out of a tool
        LOG.error("classify_news failed: %s", exc)
        return {
            "verdict": "Suspicious",
            "band": "Suspicious",
            "trust_score": 50.0,
            "probability_real": 0.5,
            "model": "unavailable",
            "degraded": True,
            "error": str(exc)[:300],
        }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
