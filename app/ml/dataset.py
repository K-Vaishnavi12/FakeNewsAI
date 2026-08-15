"""Built-in bootstrap dataset for the fake-news classifier.

IMPORTANT AND DELIBERATE LIMITATION
-----------------------------------
This project ships with a *synthetic, template-generated* corpus so that the
application is runnable immediately without downloading a third-party dataset.

The generator models the **stylistic register** of sensational/fabricated
reporting versus neutral wire-service reporting. It therefore teaches the
classifier to recognise *writing style*, not *truth*. A carefully written false
statement will look "REAL" to this model, and a badly written true statement may
look "FAKE".

Consequently the ML signal is treated everywhere in this application as a weak
prior, never as proof. See `app/analysis/verdict.py`.

To train on real data instead, drop a CSV with `text,label` columns (label in
{0,1} where 1 = REAL) at `data/train.csv` and re-run the training script; it is
picked up automatically and takes precedence over the synthetic corpus.

You may additionally harvest genuine REAL-class headlines from the News API with
`python -m app.ml.train --augment-from-newsapi`.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT

LABEL_REAL = 1
LABEL_FAKE = 0

# --- Vocabulary slots -------------------------------------------------------

ORGS = [
    "the World Health Organization", "the European Central Bank", "NASA",
    "the Ministry of Health", "the Reserve Bank", "the Supreme Court",
    "the Department of Transport", "the National Weather Service",
    "the Election Commission", "the Federal Trade Commission",
    "the Indian Space Research Organisation", "the United Nations",
    "the Ministry of Finance", "the National Health Service",
    "the Securities and Exchange Board", "the Environment Agency",
    "the Civil Aviation Authority", "the State Education Board",
]

PLACES = [
    "Hyderabad", "Brussels", "Nairobi", "Toronto", "Jakarta", "Lisbon",
    "Chennai", "Manchester", "Osaka", "Bogota", "Warsaw", "Cape Town",
    "Dublin", "Seoul", "Helsinki", "Montreal", "Bengaluru", "Copenhagen",
]

TOPICS = [
    "public transport funding", "a coastal flood barrier", "school meal standards",
    "regional rail expansion", "hospital staffing levels", "air quality monitoring",
    "renewable energy targets", "municipal water treatment", "housing permits",
    "digital payment rules", "port modernisation", "wildfire preparedness",
    "vaccination coverage", "road safety enforcement", "grain storage capacity",
    "broadband rollout", "flood insurance reform", "waste recycling targets",
]

ROLES = [
    "a spokesperson", "the department's director", "a senior official",
    "the project lead", "an agency representative", "the committee chair",
    "the regional commissioner", "a programme manager",
]

# --- REAL-register templates (neutral, attributed, specific) -----------------

REAL_TEMPLATES = [
    "{org} said on {day} that it would review {topic} in {place} over the next {n} months.",
    "{org} published a report on {topic}, noting a {pct} percent change compared with the previous year.",
    "Officials in {place} approved {n} million in funding for {topic}, according to {org}.",
    "{role} at {org} told reporters that consultations on {topic} would begin in {place} in {month}.",
    "A study released by {org} examined {topic} across {n} districts in {place}.",
    "{org} confirmed that {topic} would be included in the budget presented in {month}.",
    "The proposal on {topic} passed with {n} votes in favour, {org} said in a statement.",
    "According to data from {org}, {topic} in {place} improved by {pct} percent since {year}.",
    "{org} opened a public consultation on {topic}; submissions close in {month}.",
    "Authorities in {place} said {n} sites had been inspected as part of a review of {topic}.",
    "{role} said the {topic} programme in {place} remains under review, {org} confirmed.",
    "{org} allocated additional resources to {topic} following a {pct} percent rise in demand.",
    "A committee established by {org} will report on {topic} by the end of {year}.",
    "{org} said the changes to {topic} take effect in {month} and apply to {place}.",
    "Regulators at {org} requested further information on {topic} before issuing a decision.",
    "{org} reported {n} inspections related to {topic} in {place} during the last quarter.",
    "The review of {topic} was extended by {n} weeks, {role} at {org} said.",
    "{place} council voted to fund {topic}, with {org} providing technical guidance.",
]

# --- FAKE-register templates (sensational, unsourced, absolute) --------------

FAKE_TEMPLATES = [
    "SHOCKING: {org} SECRETLY admits {topic} was a LIE all along — media SILENT!",
    "You WON'T BELIEVE what {org} is hiding about {topic} in {place}!!!",
    "BREAKING!!! {topic} BANNED overnight in {place} — officials REFUSE to comment!",
    "EXPOSED: The TRUTH about {topic} that {org} does NOT want you to know!",
    "They don't want you to know this ONE trick about {topic} — {org} is FURIOUS!",
    "URGENT: {place} residents told to PREPARE as {topic} collapses COMPLETELY!",
    "{org} CAUGHT red-handed falsifying every single record on {topic}!",
    "Insiders reveal {org} plans to ELIMINATE {topic} entirely — share before deleted!",
    "MUST READ: {topic} causes MASSIVE damage, and {org} covered it up for {n} years!",
    "ALERT!!! Everything you were told about {topic} in {place} is 100% FALSE!",
    "Anonymous whistleblower DESTROYS {org} over {topic} — mainstream media IGNORES it!",
    "The REAL reason {org} abandoned {topic} will SHOCK you to your core!",
    "{place} in CHAOS as {topic} scandal explodes — {org} silent!!! SHARE NOW!",
    "WARNING: {topic} is a total HOAX invented by {org} to control you!",
    "Doctors HATE this: the {topic} secret {org} buried for decades!",
    "UNBELIEVABLE! {org} quietly deleted ALL evidence about {topic} last night!",
    "This changes EVERYTHING about {topic} — and {org} is PANICKING!",
    "LEAKED documents PROVE {org} lied about {topic} in {place}! Wake up!!!",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class Dataset:
    texts: list[str]
    labels: list[int]
    origin: str

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def real_count(self) -> int:
        return sum(1 for lbl in self.labels if lbl == LABEL_REAL)

    @property
    def fake_count(self) -> int:
        return sum(1 for lbl in self.labels if lbl == LABEL_FAKE)


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        org=rng.choice(ORGS),
        place=rng.choice(PLACES),
        topic=rng.choice(TOPICS),
        role=rng.choice(ROLES),
        month=rng.choice(MONTHS),
        day=rng.choice(DAYS),
        n=rng.randint(2, 90),
        pct=rng.randint(1, 45),
        year=rng.randint(2015, 2024),
    )


def generate_synthetic_dataset(
    samples_per_class: int = 1200, seed: int = 42
) -> Dataset:
    """Deterministically generate the bootstrap corpus."""
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []

    for _ in range(samples_per_class):
        real = _fill(rng.choice(REAL_TEMPLATES), rng)
        # Occasionally build a multi-sentence "article" for length variety.
        if rng.random() < 0.35:
            real += " " + _fill(rng.choice(REAL_TEMPLATES), rng)
        texts.append(real)
        labels.append(LABEL_REAL)

        fake = _fill(rng.choice(FAKE_TEMPLATES), rng)
        if rng.random() < 0.35:
            fake += " " + _fill(rng.choice(FAKE_TEMPLATES), rng)
        texts.append(fake)
        labels.append(LABEL_FAKE)

    return Dataset(texts=texts, labels=labels, origin="synthetic-bootstrap-v1")


def load_csv_dataset(path: Path) -> Dataset | None:
    """Load an optional user-supplied `text,label` CSV."""
    if not path.exists():
        return None
    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return None
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        text_col = cols.get("text") or cols.get("title") or cols.get("content")
        label_col = cols.get("label") or cols.get("target") or cols.get("class")
        if not text_col or not label_col:
            return None
        for row in reader:
            text = (row.get(text_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip().lower()
            if not text or not raw_label:
                continue
            if raw_label in {"1", "real", "true"}:
                labels.append(LABEL_REAL)
            elif raw_label in {"0", "fake", "false"}:
                labels.append(LABEL_FAKE)
            else:
                continue
            texts.append(text)
    if len(texts) < 20:
        return None
    return Dataset(texts=texts, labels=labels, origin=f"csv:{path.name}")


def build_training_dataset(
    csv_path: Path | None = None,
    samples_per_class: int = 1200,
    extra_real_texts: list[str] | None = None,
) -> Dataset:
    """Prefer a user CSV; otherwise use the synthetic corpus.

    `extra_real_texts` lets the training script fold in genuine headlines
    harvested from the News API as additional REAL-class examples.
    """
    csv_path = csv_path or (PROJECT_ROOT / "data" / "train.csv")
    dataset = load_csv_dataset(csv_path)
    if dataset is None:
        dataset = generate_synthetic_dataset(samples_per_class=samples_per_class)

    if extra_real_texts:
        cleaned = [t.strip() for t in extra_real_texts if t and len(t.strip()) > 25]
        if cleaned:
            dataset = Dataset(
                texts=dataset.texts + cleaned,
                labels=dataset.labels + [LABEL_REAL] * len(cleaned),
                origin=f"{dataset.origin}+newsapi({len(cleaned)})",
            )
    return dataset
