"""Central configuration, loaded from environment variables only.

Security note: this module MUST NOT contain default values for any secret.
A missing secret should degrade functionality (e.g. fall back to the keyless
Google News RSS feed), never silently use a hardcoded credential.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable ('1', 'true', 'yes', 'on')."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back on bad input."""
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: str) -> list:
    """Parse a comma-separated environment variable into a stripped list."""
    raw = os.getenv(name) or default
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """Runtime settings for the FakeNewsAI backend."""

    # --- Secrets -----------------------------------------------------------
    # No default. If unset, news_fetcher falls back to Google News RSS.
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

    # --- Model -------------------------------------------------------------
    MODEL_MODE = os.getenv("MODEL_MODE", "local")

    # --- LLM synthesis (opt-in) -------------------------------------------
    # Disabled by default: the rule-based synthesis engine in agent.py is the
    # supported path. Enabling this requires an instruction-tuned model that
    # can actually follow the JSON contract in agent.py's prompt.
    ENABLE_LLM = _env_bool("ENABLE_LLM", False)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf")
    # Instruction-tuned default (gpt2 is a base model and cannot follow the
    # prompt or emit JSON, which made the old LLM path silently useless).
    LOCAL_HF_MODEL = os.getenv("LOCAL_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    # --- HTTP / security ---------------------------------------------------
    # Explicit allow-list; never '*'. Vite dev server defaults included.
    CORS_ALLOWED_ORIGINS = _env_list(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    # Hard cap on analysed text. Protects against unbounded CPU/LLM cost.
    MAX_INPUT_CHARS = _env_int("MAX_INPUT_CHARS", 10_000)
    # page_size bounds for any endpoint accepting pagination.
    MIN_PAGE_SIZE = 1
    MAX_PAGE_SIZE = _env_int("MAX_PAGE_SIZE", 50)
    DEFAULT_PAGE_SIZE = 5

    # Flask-Limiter rules (per client IP).
    RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "200 per hour")
    RATE_LIMIT_ANALYZE = os.getenv("RATE_LIMIT_ANALYZE", "20 per minute;300 per hour")
    RATE_LIMIT_TRAIN = os.getenv("RATE_LIMIT_TRAIN", "2 per hour")
    # Set to a redis:// URI in production so limits are shared across workers.
    RATE_LIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")

    # Training endpoint is destructive; keep it off unless explicitly enabled.
    ENABLE_TRAIN_ENDPOINT = _env_bool("ENABLE_TRAIN_ENDPOINT", False)

    # --- Caching -----------------------------------------------------------
    NEWS_CACHE_TTL_SECONDS = _env_int("NEWS_CACHE_TTL_SECONDS", 600)  # 10 min
    NEWS_CACHE_MAX_ENTRIES = _env_int("NEWS_CACHE_MAX_ENTRIES", 256)

    # --- Logging -----------------------------------------------------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
