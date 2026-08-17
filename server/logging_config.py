"""Logging setup for the FakeNewsAI backend.

Replaces the ad-hoc ``print()`` calls that were scattered across the server.
Call :func:`setup_logging` exactly once at process start (``app.py`` /
``cli.py`` / ``__main__.py``); every other module should simply do::

    from .logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once, idempotently.

    Args:
        level: Logging level name, e.g. ``"DEBUG"``/``"INFO"``/``"WARNING"``.
            Unrecognised values fall back to ``INFO``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = getattr(logging, str(level).upper(), logging.INFO)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved)
    root.handlers = [handler]

    # These libraries are extremely chatty at INFO and drown out our logs.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
