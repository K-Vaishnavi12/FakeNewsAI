"""Centralised configuration.

All secrets are read from environment variables (optionally loaded from a local
`.env` file). Nothing in this module is ever serialised to an API response, so
keys cannot leak to the frontend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `override=False` means a real exported environment variable always wins over
# the .env file, which is what CI and container deployments expect.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    # Treat the placeholder values from .env.example as "not configured".
    if not raw or raw.startswith("your_"):
        return default
    return raw


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    # News API
    news_api_key: str = ""
    news_api_url: str = "https://newsapi.org/v2/everything"
    news_api_timeout_seconds: float = 8.0
    news_api_page_size: int = 10

    # NVIDIA
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-8b-instruct"
    nvidia_timeout_seconds: float = 60.0
    nvidia_temperature: float = 0.2
    nvidia_max_tokens: int = 2048

    # Application
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    max_input_chars: int = 20000
    backend_url: str = "http://127.0.0.1:8000"

    @property
    def news_api_configured(self) -> bool:
        return bool(self.news_api_key)

    @property
    def nvidia_configured(self) -> bool:
        return bool(self.nvidia_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_dir = _get_str("MODEL_DIR", "artifacts")
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    return Settings(
        news_api_key=_get_str("NEWS_API_KEY"),
        news_api_url=_get_str("NEWS_API_URL", "https://newsapi.org/v2/everything"),
        news_api_timeout_seconds=_get_float("NEWS_API_TIMEOUT_SECONDS", 8.0),
        news_api_page_size=_get_int("NEWS_API_PAGE_SIZE", 10),
        nvidia_api_key=_get_str("NVIDIA_API_KEY"),
        nvidia_base_url=_get_str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        nvidia_model=_get_str("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
        nvidia_timeout_seconds=_get_float("NVIDIA_TIMEOUT_SECONDS", 60.0),
        nvidia_temperature=_get_float("NVIDIA_TEMPERATURE", 0.2),
        nvidia_max_tokens=_get_int("NVIDIA_MAX_TOKENS", 2048),
        model_dir=model_path,
        max_input_chars=_get_int("MAX_INPUT_CHARS", 20000),
        backend_url=_get_str("BACKEND_URL", "http://127.0.0.1:8000"),
    )


def reset_settings_cache() -> None:
    """Used by tests that patch environment variables."""
    get_settings.cache_clear()
