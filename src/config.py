"""Central configuration, path resolution, logging and seeding for VeriTruth.

Every module imports from here so that paths stay cross-platform (pathlib) and
all secrets are read from the environment (never hardcoded).
"""

from __future__ import annotations

import logging
import os
import random
from functools import cache
from pathlib import Path

import numpy as np

try:  # python-dotenv is optional at runtime; absence must not crash anything.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - only hit when dotenv missing

    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False


SEED = 42

# repo root = .../veritruth  (this file lives at veritruth/src/config.py)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")


def _path_from_env(var: str, default: str) -> Path:
    raw = os.getenv(var, default)
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p)


DATA_DIR: Path = _path_from_env("VERITRUTH_DATA_DIR", "data")
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = _path_from_env("VERITRUTH_MODEL_DIR", "models")
VECTORDB_DIR: Path = _path_from_env("VERITRUTH_VECTORDB_DIR", "vectordb")
FEEDBACK_DB_PATH: Path = _path_from_env("VERITRUTH_DB_PATH", "data/feedback.sqlite3")

BASELINE_PATH: Path = MODEL_DIR / "baseline.joblib"
THRESHOLD_PATH: Path = MODEL_DIR / "threshold.json"
TRANSFORMER_DIR: Path = MODEL_DIR / "distilbert"
CORPUS_PATH: Path = DATA_DIR / "factcheck_corpus.json"

CHROMA_COLLECTION = "factchecks"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TRANSFORMER_MODEL = os.getenv("TRANSFORMER_MODEL", "distilbert-base-uncased")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Trust-score bands (0-100, higher = more trustworthy / more likely Real).
BAND_REAL_MIN = 70.0
BAND_SUSPICIOUS_MIN = 40.0

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

MAX_TEXT_CHARS = 20_000
TRANSFORMER_MAX_LEN = 256
CPU_BATCH_SIZE = 8
GPU_BATCH_SIZE = 16

HTTP_TIMEOUT = 15.0


def get_env(name: str, default: str = "") -> str:
    """Read an env var, treating whitespace-only values as unset."""
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def agent_timeout() -> float:
    try:
        return float(get_env("AGENT_TIMEOUT_SECONDS", "30") or 30)
    except ValueError:
        return 30.0


def agent_max_tool_calls() -> int:
    try:
        return int(get_env("AGENT_MAX_TOOL_CALLS", "6") or 6)
    except ValueError:
        return 6


def force_baseline() -> bool:
    return get_env("FORCE_BASELINE", "0").lower() in {"1", "true", "yes", "on"}


def ensure_dirs() -> None:
    """Create every directory the project writes to."""
    for directory in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MODEL_DIR,
        VECTORDB_DIR,
        FEEDBACK_DB_PATH.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Make every stochastic component reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch optional / partially installed
        pass


_LOGGING_READY = False


@cache
def get_logger(name: str = "veritruth") -> logging.Logger:
    """Structured-ish console logger, configured exactly once."""
    global _LOGGING_READY
    if not _LOGGING_READY:
        logging.basicConfig(
            level=get_env("LOG_LEVEL", "INFO").upper() or "INFO",
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        _LOGGING_READY = True
    return logging.getLogger(name)


def band_for_score(score: float) -> str:
    """Map a 0-100 trust score to a verdict band."""
    if score >= BAND_REAL_MIN:
        return "Real"
    if score >= BAND_SUSPICIOUS_MIN:
        return "Suspicious"
    return "Fake"


ensure_dirs()
