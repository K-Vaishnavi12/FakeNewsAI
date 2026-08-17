"""Dataset sync utilities for pulling training CSVs from a public Drive folder.

Usage::

    python -m server.datasets sync --purge-existing
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, List

from .logging_config import get_logger, setup_logging
from .paths import DATA_DIR

logger = get_logger(__name__)

DEFAULT_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1vuV3ALf6JmHLR8HVGENPPK9LG9bcWSnC"
)

# Files the training pipeline can consume today.
REQUIRED_TRAINING_FILES = (
    "WELFake_Dataset.csv",
    "Fake.csv",
    "True.csv",
    "BuzzFeed_fake_news_content.csv",
    "BuzzFeed_real_news_content.csv",
)


def _iter_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def _flatten_to_dir(files: Iterable[str], target_dir: str) -> List[str]:
    """Move downloaded files into target_dir and return final absolute paths."""
    final_paths: List[str] = []
    os.makedirs(target_dir, exist_ok=True)

    for src in files:
        if not os.path.isfile(src):
            continue
        dst = os.path.join(target_dir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)
        final_paths.append(dst)

    return final_paths


def _clean_download_artifacts(root: str) -> None:
    """Remove empty directories left by folder downloads."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def _purge_existing_dataset_files(data_dir: str) -> int:
    removed = 0
    for path in _iter_files(data_dir):
        if os.path.splitext(path)[1].lower() in {".csv", ".txt", ".mat"}:
            os.remove(path)
            removed += 1
    return removed


def _extract_folder_id(url: str) -> str:
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError("Drive folder URL must contain '/folders/<id>'.")
    return m.group(1)


def sync_from_drive(
    folder_url: str = DEFAULT_DRIVE_FOLDER_URL,
    data_dir: str = DATA_DIR,
    purge_existing: bool = False,
) -> List[str]:
    """Download a public Google Drive folder into server/data.

    Returns:
        List of absolute file paths now present in data_dir.
    """
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for dataset sync. Install dependencies with "
            "`pip install -r server/requirements.txt`."
        ) from exc

    os.makedirs(data_dir, exist_ok=True)

    if purge_existing:
        removed = _purge_existing_dataset_files(data_dir)
        logger.info("Removed %d existing dataset files from %s", removed, data_dir)

    folder_id = _extract_folder_id(folder_url)
    canonical_url = f"https://drive.google.com/drive/folders/{folder_id}"

    logger.info("Downloading datasets from %s", canonical_url)
    downloaded = gdown.download_folder(
        url=canonical_url,
        output=data_dir,
        quiet=False,
        remaining_ok=True,
    )

    if not downloaded:
        raise RuntimeError("No files were downloaded from the provided Drive folder.")

    final_paths = _flatten_to_dir(downloaded, data_dir)
    _clean_download_artifacts(data_dir)

    names = {os.path.basename(p) for p in final_paths}
    missing = [name for name in REQUIRED_TRAINING_FILES if name not in names]
    if missing:
        logger.warning(
            "Download completed but some training files are missing: %s",
            ", ".join(missing),
        )

    logger.info("Dataset sync finished: %d files available in %s", len(final_paths), data_dir)
    return sorted(final_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset sync helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Download datasets into server/data")
    sync.add_argument("--drive-url", default=DEFAULT_DRIVE_FOLDER_URL,
                      help="Public Google Drive folder URL")
    sync.add_argument("--data-dir", default=DATA_DIR,
                      help="Target data directory (default: server/data)")
    sync.add_argument("--purge-existing", action="store_true",
                      help="Remove existing .csv/.txt/.mat files before download")

    args = parser.parse_args()
    setup_logging("INFO")

    if args.command == "sync":
        files = sync_from_drive(
            folder_url=args.drive_url,
            data_dir=args.data_dir,
            purge_existing=args.purge_existing,
        )
        print(f"Synced {len(files)} files into {os.path.abspath(args.data_dir)}")


if __name__ == "__main__":
    main()
