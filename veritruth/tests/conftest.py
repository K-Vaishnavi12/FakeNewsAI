"""Shared pytest fixtures.

Tests must never depend on network access, a Gemini key, or a trained model, so
the environment is pinned to fully-offline defaults before any src module is
imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Offline, deterministic, fast: no MCP subprocesses, no LLM, no remote APIs.
os.environ.setdefault("AGENT_USE_MCP", "0")
os.environ.setdefault("AGENT_TIMEOUT_SECONDS", "60")
os.environ.setdefault("EXPLAINER_BACKEND", "occlusion")
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_FACTCHECK_API_KEY", None)


@pytest.fixture(scope="session")
def sample_texts() -> dict[str, str]:
    return {
        "real": (
            "The central bank kept its benchmark interest rate unchanged at 6.5 percent "
            "on Friday, citing steady inflation and resilient domestic demand."
        ),
        "fake": (
            "SHOCKING!!! Doctors are furious about this one weird trick the government "
            "has hidden for 40 years that cures every disease overnight. SHARE NOW!!!"
        ),
        "unicode": "Le président a déclaré que l'économie est stable. 政府发表声明。🙂",
    }


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with the real lifespan executed once per session."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client
