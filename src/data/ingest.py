"""Step 2a — dataset ingestion.

Loads the ISOT dataset (``True.csv`` / ``Fake.csv``) and the LIAR dataset
(``train.tsv`` / ``valid.tsv`` / ``test.tsv``) from ``data/raw`` and normalises
both into a single schema::

    text, label, source, language

``label`` is 1 for REAL and 0 for FAKE.

If no raw dataset files are present, a 200-row synthetic sample dataset is
generated so that the pipeline always runs end-to-end (a loud warning is
printed).

Run::

    python -m src.data.ingest
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, SEED, get_logger, set_seeds

LOG = get_logger("veritruth.data.ingest")

SCHEMA = ["text", "label", "source", "language"]

# LIAR truth labels -> binary. LIAR ships 6 ordinal classes.
LIAR_LABEL_MAP = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0,
}

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

_REAL_TEMPLATES = [
    "The {agency} released its quarterly report on {topic}, showing a {pct} percent change compared with the previous quarter.",
    "Officials from the {agency} confirmed on Tuesday that new guidance on {topic} will take effect next month.",
    "Researchers at {agency} published a peer-reviewed study on {topic} in a leading journal, based on a sample of {pct} thousand participants.",
    "The {agency} said in a statement that funding for {topic} programmes would increase by {pct} percent this fiscal year.",
    "According to data published by the {agency}, {topic} indicators remained broadly stable over the last {pct} months.",
]

_FAKE_TEMPLATES = [
    "SHOCKING: Secret {agency} documents PROVE that {topic} is a total hoax — the media will NEVER tell you this!!!",
    "BREAKING!!! Doctors are FURIOUS as this one weird trick about {topic} destroys the {agency} forever!",
    "You won't BELIEVE what the {agency} is hiding about {topic}. Share before they delete this!!!",
    "EXPOSED: {agency} insiders admit {topic} was staged all along — {pct} percent of people have no idea!",
    "URGENT WARNING: {topic} will be BANNED by the {agency} next week. Wake up, people!!!",
]

_AGENCIES = [
    "Ministry of Health",
    "Election Commission",
    "Reserve Bank",
    "World Health Organization",
    "Department of Education",
    "National Statistics Office",
    "Supreme Court",
    "Transport Authority",
]

_TOPICS = [
    "vaccination coverage",
    "the national election",
    "inflation figures",
    "school reopening",
    "currency circulation",
    "air quality levels",
    "the new tax rules",
    "public transport fares",
]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SCHEMA})


def load_isot(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load ISOT True.csv / Fake.csv if present, else return an empty frame."""
    frames: list[pd.DataFrame] = []
    for filename, label in (("True.csv", 1), ("Fake.csv", 0)):
        path = raw_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # corrupted / unreadable file
            LOG.warning("Could not read %s: %s", path.name, exc)
            continue
        title = df["title"].fillna("") if "title" in df.columns else ""
        body = df["text"].fillna("") if "text" in df.columns else ""
        combined = (title + " " + body) if isinstance(title, pd.Series) else body
        if not isinstance(combined, pd.Series):
            LOG.warning("%s has no 'title'/'text' columns; skipping.", path.name)
            continue
        frames.append(
            pd.DataFrame(
                {
                    "text": combined.astype(str).str.strip(),
                    "label": label,
                    "source": "isot",
                    "language": "en",
                }
            )
        )
        LOG.info("Loaded %s rows from ISOT %s", len(frames[-1]), path.name)
    return pd.concat(frames, ignore_index=True) if frames else _empty_frame()


def load_liar(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load LIAR train/valid/test TSV files if present, else an empty frame."""
    frames: list[pd.DataFrame] = []
    for filename in ("train.tsv", "valid.tsv", "test.tsv"):
        path = raw_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS, quoting=3)
        except Exception as exc:
            LOG.warning("Could not read %s: %s", path.name, exc)
            continue
        df = df.dropna(subset=["label", "statement"])
        df["binary"] = df["label"].astype(str).str.strip().str.lower().map(LIAR_LABEL_MAP)
        df = df.dropna(subset=["binary"])
        frames.append(
            pd.DataFrame(
                {
                    "text": df["statement"].astype(str).str.strip(),
                    "label": df["binary"].astype(int),
                    "source": "liar",
                    "language": "en",
                }
            )
        )
        LOG.info("Loaded %s rows from LIAR %s", len(frames[-1]), path.name)
    return pd.concat(frames, ignore_index=True) if frames else _empty_frame()


def synthetic_sample(n_rows: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Deterministic synthetic fallback dataset (balanced real/fake)."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n_rows):
        label = i % 2  # perfectly balanced
        template = rng.choice(_REAL_TEMPLATES if label == 1 else _FAKE_TEMPLATES)
        text = template.format(
            agency=rng.choice(_AGENCIES),
            topic=rng.choice(_TOPICS),
            pct=rng.randint(2, 97),
        )
        # Add a little lexical variety so models do not memorise 5 strings.
        suffix = (
            f" Reported on day {rng.randint(1, 28)} by the newsroom desk."
            if label == 1
            else f" Click here now!!! {rng.randint(1000, 99999)} shares and counting!!!"
        )
        rows.append(
            {
                "text": text + suffix,
                "label": label,
                "source": "synthetic",
                "language": "en",
            }
        )
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=SCHEMA)


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Combine every available raw dataset, falling back to synthetic data."""
    set_seeds()
    frames = [df for df in (load_isot(raw_dir), load_liar(raw_dir)) if not df.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        LOG.info("Combined raw dataset: %s rows", len(combined))
        return combined[SCHEMA]

    LOG.warning(
        "=" * 72
        + "\nNO RAW DATASET FOUND in %s.\n"
        "Expected ISOT (True.csv / Fake.csv) and/or LIAR (train.tsv, valid.tsv, test.tsv).\n"
        "Generating a 200-row SYNTHETIC sample dataset so the pipeline still runs.\n"
        "Metrics from synthetic data are NOT meaningful for real-world accuracy.\n" + "=" * 72,
        raw_dir,
    )
    return synthetic_sample()


def main() -> pd.DataFrame:
    df = load_raw()
    print("\n--- STEP 2a VERIFICATION -------------------------------------")
    print(f"Rows loaded       : {len(df)}")
    print(f"Sources           : {sorted(df['source'].unique().tolist())}")
    print(f"Label counts      : {df['label'].value_counts().to_dict()}  (1=REAL, 0=FAKE)")
    print(f"Example text      : {df['text'].iloc[0][:120]!r}")
    print("Expected          : >0 rows, both labels present, schema text/label/source/language")
    print("---------------------------------------------------------------\n")
    return df


if __name__ == "__main__":
    main()
