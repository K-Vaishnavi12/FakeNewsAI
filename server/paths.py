"""Filesystem sandboxing for anything reachable from an HTTP request.

The ``/api/train_local`` endpoint used to accept raw ``csv_path`` and
``out_path`` strings straight from the request body and hand them to
``pandas.read_csv`` / ``joblib.dump``. That is an arbitrary file read *and*
an arbitrary file write, and since ``joblib.dump`` writes a pickle, an
attacker-chosen output path is effectively a remote code execution primitive.

The fix has two layers:

1. :func:`resolve_within` -- a hard containment check. Any path that escapes
   its allowed base directory (via ``..``, an absolute path, or a symlink)
   is rejected.
2. :func:`resolve_dataset` / :func:`resolve_model_output` -- the preferred
   API. Callers pass a *logical name* (``"welfake"``, ``"isot"``) rather than
   a path, so untrusted input never touches the filesystem layer at all.
"""

import os
import re

from .logging_config import get_logger

logger = get_logger(__name__)

# Base directories. Everything reachable from HTTP must live under one of these.
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SERVER_DIR)
# Datasets live in server/data/. They are gitignored and fetched separately via
# `python -m server.datasets sync` -- see server/data/README.md.
#
# This constant is the single source of truth for the location. Defining it once
# here is what stops the training pipeline, the sync tool and the docs from
# disagreeing; a previous revision pointed this at <repo>/Data while .gitignore
# and the README still said server/data/, so anyone following the README put the
# CSVs somewhere the loader never looked.
DATA_DIR = os.path.join(SERVER_DIR, "data")
MODELS_DIR = os.path.join(SERVER_DIR, "ml", "models")

# Logical dataset name -> filename inside DATA_DIR.
# Using a fixed registry means the API surface is an enum, not a path.
#
# NOTE: PolitiFact_fake_news_content.csv is deliberately NOT listed. Despite
# the filename its rows carry ids like "Real_1-Webpage" -- it holds the *real*
# article set, not the fake one. Registering it would let a caller train on
# ~120 real articles labelled as fake, poisoning the model. The Drive folder
# ships no correct PolitiFact fake set; add one before re-enabling this.
# (BuzzFeed_fake_news_content.csv is fine -- its ids are "Fake_1-Webpage".)
DATASET_REGISTRY = {
    "welfake": "WELFake_Dataset.csv",
    "isot_fake": "Fake.csv",
    "isot_true": "True.csv",
    "buzzfeed_fake": "BuzzFeed_fake_news_content.csv",
    "buzzfeed_real": "BuzzFeed_real_news_content.csv",
}

# Model artefact names must be a bare, safe filename ending in .joblib.
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}\.joblib$")


class UnsafePathError(ValueError):
    """Raised when a caller-supplied path escapes its allowed base directory."""


def resolve_within(base_dir: str, candidate: str) -> str:
    """Resolve ``candidate`` and guarantee it stays inside ``base_dir``.

    Args:
        base_dir: Allowed root directory.
        candidate: Untrusted relative path.

    Returns:
        The absolute, real (symlink-resolved) path.

    Raises:
        UnsafePathError: If ``candidate`` is absolute, empty, or resolves
            outside ``base_dir``.
    """
    if not candidate or not str(candidate).strip():
        raise UnsafePathError("Empty path is not allowed.")

    candidate = str(candidate).strip()

    # Reject absolute paths and Windows drive/UNC prefixes outright.
    if os.path.isabs(candidate) or re.match(r"^[A-Za-z]:", candidate) \
            or candidate.startswith("\\\\"):
        raise UnsafePathError("Absolute paths are not allowed.")

    base_real = os.path.realpath(base_dir)
    target_real = os.path.realpath(os.path.join(base_real, candidate))

    # os.path.commonpath is used rather than startswith so that a sibling
    # directory like "/data-evil" cannot masquerade as being under "/data".
    try:
        if os.path.commonpath([base_real, target_real]) != base_real:
            raise UnsafePathError("Path escapes the allowed base directory.")
    except ValueError as exc:  # different drives on Windows
        raise UnsafePathError("Path escapes the allowed base directory.") from exc

    return target_real


def resolve_dataset(name: str) -> str:
    """Map a logical dataset name to an on-disk CSV path.

    Args:
        name: A key of :data:`DATASET_REGISTRY`, e.g. ``"welfake"``.

    Returns:
        Absolute path to the dataset CSV.

    Raises:
        UnsafePathError: If the name is unknown or the file is missing.
    """
    key = str(name or "").strip().lower()
    if key not in DATASET_REGISTRY:
        allowed = ", ".join(sorted(DATASET_REGISTRY))
        raise UnsafePathError(f"Unknown dataset '{name}'. Allowed: {allowed}.")

    path = resolve_within(DATA_DIR, DATASET_REGISTRY[key])
    if not os.path.isfile(path):
        raise UnsafePathError(f"Dataset '{key}' is not present on this server.")
    return path


def resolve_model_output(filename: str = "fake_news_model.joblib") -> str:
    """Validate a model artefact filename and return its path under MODELS_DIR.

    Args:
        filename: Bare filename (no directory components) ending in ``.joblib``.

    Returns:
        Absolute path inside :data:`MODELS_DIR`.

    Raises:
        UnsafePathError: If the filename contains a path separator or does not
            match the safe-name pattern.
    """
    name = str(filename or "").strip()
    if not _SAFE_MODEL_NAME.match(name):
        raise UnsafePathError(
            "Model filename must match [A-Za-z0-9._-]{1,64}.joblib "
            "with no directory components."
        )
    os.makedirs(MODELS_DIR, exist_ok=True)
    return resolve_within(MODELS_DIR, name)
