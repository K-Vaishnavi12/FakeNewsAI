"""Step 2c — stratified 80/10/10 split written to ``data/processed``.

Run::

    python -m src.data.split

Produces ``train.csv``, ``val.csv``, ``test.csv`` (schema: text,label,source,language).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, SEED, ensure_dirs, get_logger, set_seeds

LOG = get_logger("veritruth.data.split")

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


def _can_stratify(labels: pd.Series, n_parts: int) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and int(counts.min()) >= n_parts


def stratified_split(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 80/10/10, stratified by label when class counts allow it."""
    if len(df) < 10:
        raise ValueError(f"Need at least 10 rows to split, got {len(df)}")

    strat = df["label"] if _can_stratify(df["label"], 10) else None
    train, holdout = train_test_split(
        df, test_size=0.20, random_state=seed, shuffle=True, stratify=strat
    )
    strat2 = holdout["label"] if _can_stratify(holdout["label"], 2) else None
    val, test = train_test_split(
        holdout, test_size=0.50, random_state=seed, shuffle=True, stratify=strat2
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_splits(out_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Full pipeline: ingest -> clean -> split -> write CSVs."""
    from src.data.ingest import load_raw
    from src.data.preprocess import clean_frame

    set_seeds()
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned = clean_frame(load_raw())
    train, val, test = stratified_split(cleaned)

    train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    val.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")
    LOG.info("Wrote splits to %s (train=%s val=%s test=%s)", out_dir, len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}


def load_splits(out_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Load splits from disk, building them first if they are missing."""
    paths = {"train": out_dir / "train.csv", "val": out_dir / "val.csv", "test": out_dir / "test.csv"}
    if not all(p.exists() for p in paths.values()):
        LOG.info("Processed splits missing; building them now.")
        return build_splits(out_dir)
    return {k: pd.read_csv(v).fillna({"text": "", "source": "unknown", "language": "en"}) for k, v in paths.items()}


def main() -> dict[str, pd.DataFrame]:
    splits = build_splits()
    print("\n--- STEP 2c VERIFICATION -------------------------------------")
    for name, frame in splits.items():
        dist = frame["label"].value_counts(normalize=True).round(3).to_dict()
        print(f"{name:5s}: {len(frame):6d} rows | label ratio {dist}")
    print(f"Files written    : {TRAIN_PATH.name}, {VAL_PATH.name}, {TEST_PATH.name} in {PROCESSED_DIR}")
    print("Expected         : ~80/10/10 split, similar label ratios across splits")
    print("---------------------------------------------------------------\n")
    return splits


if __name__ == "__main__":
    main()
