"""Dataset acquisition helpers.

The training CSVs are large (~408 MB) and are deliberately **not** committed to
Git -- ``WELFake_Dataset.csv`` alone is 234 MB and GitHub rejects any file over
100 MB inside a repository. They are published instead as assets on the
``datasets-v1`` GitHub release, mirrored from the original Google Drive folder.
See ``server/data/README.md``.

This module is the single supported way to populate ``server/data/``::

    python -m server.datasets check
    python -m server.datasets sync                    # GitHub release (default)
    python -m server.datasets sync --from drive       # Google Drive mirror
    python -m server.datasets sync --source "D:/Data" # already-downloaded folder
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from typing import Iterable, List

from .logging_config import get_logger, setup_logging
from .paths import DATA_DIR

logger = get_logger(__name__)

DEFAULT_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1vuV3ALf6JmHLR8HVGENPPK9LG9bcWSnC"
)

# Datasets are published as release assets. The repository is private, so these
# downloads are authenticated -- hence the GitHub CLI rather than a plain URL.
GITHUB_REPO = "K-Vaishnavi12/FakeNewsAI"
RELEASE_TAG = "datasets-v1"

# Files the training pipeline can consume today.
REQUIRED_TRAINING_FILES = (
    "WELFake_Dataset.csv",
    "Fake.csv",
    "True.csv",
    "BuzzFeed_fake_news_content.csv",
    "BuzzFeed_real_news_content.csv",
)

# Extensions we are willing to copy in from an untrusted --source directory.
ALLOWED_DATASET_SUFFIXES = {".csv", ".txt", ".tsv", ".jsonl"}

# Extensions treated as dataset payload when purging.
_PURGEABLE_SUFFIXES = {".csv", ".txt", ".tsv", ".jsonl", ".mat"}


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
    """Remove empty directories and truncated ``.part`` files.

    gdown writes ``<name><random>.part`` while downloading and renames on
    success. An interrupted run therefore leaves a large partial file behind
    that looks like data but is not -- exactly how a 177 MB truncated
    WELFake_Dataset ended up in this repo's data dir.
    """
    for path in list(_iter_files(root)):
        if path.endswith(".part"):
            size_mb = os.path.getsize(path) / 1024 / 1024
            logger.warning(
                "Removing truncated download artifact %s (%.1f MB)",
                os.path.basename(path), size_mb,
            )
            os.remove(path)

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
        if os.path.splitext(path)[1].lower() in _PURGEABLE_SUFFIXES:
            os.remove(path)
            removed += 1
    return removed


def _extract_folder_id(url: str) -> str:
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError("Drive folder URL must contain '/folders/<id>'.")
    return m.group(1)


def missing_training_files(data_dir: str = DATA_DIR) -> List[str]:
    """Return the REQUIRED_TRAINING_FILES that are absent or empty."""
    missing = []
    for name in REQUIRED_TRAINING_FILES:
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            missing.append(name)
    return missing


def check(data_dir: str = DATA_DIR) -> List[str]:
    """Print a presence/size report for the dataset directory."""
    print(f"Dataset directory: {os.path.abspath(data_dir)}")
    if not os.path.isdir(data_dir):
        print("  (directory does not exist)")
        return list(REQUIRED_TRAINING_FILES)

    for name in REQUIRED_TRAINING_FILES:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            print(f"  [ok]      {name:<38} {os.path.getsize(path)/1024/1024:8.2f} MB")
        else:
            print(f"  [MISSING] {name}")

    partials = [p for p in _iter_files(data_dir) if p.endswith(".part")]
    for path in partials:
        print(f"  [PARTIAL] {os.path.basename(path)} "
              f"({os.path.getsize(path)/1024/1024:.1f} MB) -- incomplete download")

    missing = missing_training_files(data_dir)
    print(f"\n{len(REQUIRED_TRAINING_FILES) - len(missing)}"
          f"/{len(REQUIRED_TRAINING_FILES)} required training files present.")
    if missing:
        print("Run: python -m server.datasets sync")
    return missing


def sync_from_local(
    source_dir: str,
    data_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> List[str]:
    """Copy dataset files from an already-downloaded local folder.

    Only files whose extension is in :data:`ALLOWED_DATASET_SUFFIXES` are
    copied, and each destination is forced to be a direct child of ``data_dir``
    (``os.path.basename`` strips any directory component), so a crafted source
    tree cannot write outside the dataset directory.

    Args:
        source_dir: Folder containing the manually downloaded Drive files.
        data_dir: Destination dataset directory.
        overwrite: Replace files that already exist in ``data_dir``.

    Returns:
        Absolute paths of the files copied.
    """
    if not os.path.isdir(source_dir):
        raise RuntimeError(f"Source directory does not exist: {source_dir}")

    data_real = os.path.realpath(data_dir)
    os.makedirs(data_real, exist_ok=True)

    copied: List[str] = []
    skipped_existing = 0
    skipped_type = 0

    for src in _iter_files(source_dir):
        suffix = os.path.splitext(src)[1].lower()
        if suffix not in ALLOWED_DATASET_SUFFIXES:
            skipped_type += 1
            continue

        # basename() collapses any traversal attempt in the source tree.
        dst = os.path.join(data_real, os.path.basename(src))
        if os.path.dirname(os.path.realpath(dst)) != data_real:
            logger.warning("Refusing to write outside dataset dir: %s", dst)
            continue

        if os.path.exists(dst) and not overwrite:
            skipped_existing += 1
            continue

        shutil.copy2(src, dst)
        copied.append(dst)
        logger.info("Copied %s (%.1f MB)",
                    os.path.basename(dst), os.path.getsize(dst) / 1024 / 1024)

    logger.info(
        "Local sync finished: %d copied, %d already present, %d non-dataset skipped",
        len(copied), skipped_existing, skipped_type,
    )
    return sorted(copied)


def sync_from_github_release(
    data_dir: str = DATA_DIR,
    repo: str = GITHUB_REPO,
    tag: str = RELEASE_TAG,
    overwrite: bool = False,
) -> List[str]:
    """Download every dataset asset from the GitHub release into ``data_dir``.

    Uses the GitHub CLI because the repository is private, so the asset URLs
    require an authenticated request. ``gh`` already holds the credentials, and
    shelling out to it avoids inventing a token-handling path of our own --
    nothing here reads, stores or logs a credential.

    Raises:
        RuntimeError: If ``gh`` is missing, unauthenticated, or the download
            fails. The message states exactly what the caller must do.
    """
    if shutil.which("gh") is None:
        raise RuntimeError(
            "The GitHub CLI ('gh') is required to download datasets from the "
            f"private repo {repo}.\n"
            "  Install:       https://cli.github.com\n"
            "  Authenticate:  gh auth login\n"
            "Alternatively use the Google Drive mirror:\n"
            "  python -m server.datasets sync --from drive"
        )

    auth = subprocess.run(
        ["gh", "auth", "status"], capture_output=True, text=True
    )
    if auth.returncode != 0:
        raise RuntimeError(
            "The GitHub CLI is installed but not authenticated. Run:\n"
            "  gh auth login\n"
            "then re-run this command."
        )

    os.makedirs(data_dir, exist_ok=True)
    before = {
        p for p in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, p))
    }

    cmd = [
        "gh", "release", "download", tag,
        "--repo", repo,
        "--dir", data_dir,
        "--pattern", "*",
    ]
    if overwrite:
        cmd.append("--clobber")
    else:
        cmd.append("--skip-existing")

    logger.info("Downloading release assets from %s@%s", repo, tag)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh release download failed (exit {proc.returncode}).\n"
            f"{proc.stderr.strip()}\n"
            "If GitHub is reporting an incident (https://www.githubstatus.com), "
            "retry later or use: python -m server.datasets sync --from drive"
        )

    _clean_download_artifacts(data_dir)

    after = {
        p for p in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, p))
    }
    new = sorted(os.path.join(data_dir, n) for n in (after - before))
    logger.info("Release sync finished: %d new file(s) in %s", len(new), data_dir)
    return new


def sync_from_drive(
    folder_url: str = DEFAULT_DRIVE_FOLDER_URL,
    data_dir: str = DATA_DIR,
    purge_existing: bool = False,
) -> List[str]:
    """Download the public Google Drive folder into ``data_dir``.

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

    # gdown 5.x accepts remaining_ok; 6.x removed it. requirements.txt allows
    # both, so probe rather than assume -- passing it unconditionally made
    # `sync` die with TypeError on any fresh install that resolved to 6.x.
    kwargs = dict(url=canonical_url, output=data_dir, quiet=False)
    try:
        downloaded = gdown.download_folder(**kwargs, remaining_ok=True)
    except TypeError:
        downloaded = gdown.download_folder(**kwargs)

    if not downloaded:
        raise RuntimeError("No files were downloaded from the provided Drive folder.")

    final_paths = _flatten_to_dir(downloaded, data_dir)
    _clean_download_artifacts(data_dir)

    missing = missing_training_files(data_dir)
    if missing:
        logger.warning(
            "Download completed but some training files are missing: %s",
            ", ".join(missing),
        )

    logger.info("Dataset sync finished: %d files available in %s",
                len(final_paths), data_dir)
    return sorted(final_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset sync helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Populate server/data/")
    sync.add_argument("--from", dest="source_kind",
                      choices=("release", "drive"), default="release",
                      help="Where to fetch from: the GitHub release (default) "
                           "or the Google Drive mirror")
    sync.add_argument("--repo", default=GITHUB_REPO,
                      help="owner/name of the repo holding the release assets")
    sync.add_argument("--tag", default=RELEASE_TAG,
                      help="Release tag holding the dataset assets")
    sync.add_argument("--drive-url", default=DEFAULT_DRIVE_FOLDER_URL,
                      help="Public Google Drive folder URL (with --from drive)")
    sync.add_argument("--data-dir", default=DATA_DIR,
                      help="Target data directory (default: server/data)")
    sync.add_argument("--source", default=None,
                      help="Copy from this already-downloaded local folder "
                           "instead of contacting GitHub or Drive")
    sync.add_argument("--overwrite", action="store_true",
                      help="Replace files that already exist locally")
    sync.add_argument("--purge-existing", action="store_true",
                      help="Remove existing dataset files before downloading")

    check_p = sub.add_parser("check", help="Report which datasets are present")
    check_p.add_argument("--data-dir", default=DATA_DIR)

    args = parser.parse_args()
    setup_logging("INFO")

    if args.command == "check":
        raise SystemExit(1 if check(args.data_dir) else 0)

    if args.command == "sync":
        if args.source:
            files = sync_from_local(
                source_dir=args.source,
                data_dir=args.data_dir,
                overwrite=args.overwrite,
            )
        elif args.source_kind == "drive":
            files = sync_from_drive(
                folder_url=args.drive_url,
                data_dir=args.data_dir,
                purge_existing=args.purge_existing,
            )
        else:
            if args.purge_existing:
                removed = _purge_existing_dataset_files(args.data_dir)
                logger.info("Removed %d existing dataset files", removed)
            files = sync_from_github_release(
                data_dir=args.data_dir,
                repo=args.repo,
                tag=args.tag,
                overwrite=args.overwrite,
            )

        print(f"Synced {len(files)} files into {os.path.abspath(args.data_dir)}")
        remaining = missing_training_files(args.data_dir)
        if remaining:
            print("Still missing: " + ", ".join(remaining))


if __name__ == "__main__":
    main()
