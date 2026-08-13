"""Shared helpers for the four VeriTruth MCP servers.

Each server is a standalone FastMCP process spoken to over stdio. They must be
importable without side effects (so tests can call the tool functions directly)
and must never raise out of a tool — a tool that raises kills the agent turn, so
every tool returns a structured error payload instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When launched as ``python src/mcp_servers/foo.py`` the repo root is not on
# sys.path. Insert it so ``import src.*`` works regardless of launch style.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import MAX_TEXT_CHARS, get_logger  # noqa: E402

__all__ = ["MAX_TEXT_CHARS", "clean_text", "clamp_k", "get_logger"]


def clean_text(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """Coerce any input to a bounded, stripped string."""
    if text is None:
        return ""
    return str(text).strip()[:limit]


def clamp_k(k: int, default: int = 3, maximum: int = 10) -> int:
    """Coerce a caller-supplied ``k`` into ``[1, maximum]``."""
    try:
        value = int(k)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))
