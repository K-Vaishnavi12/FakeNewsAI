"""Step 2b — text cleaning and de-duplication.

Rules:
* strip HTML tags and URLs
* collapse whitespace / normalise unicode quotes
* drop rows shorter than 20 characters
* drop exact duplicates (case/whitespace-insensitive)

Run::

    python -m src.data.preprocess
"""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

from src.config import MAX_TEXT_CHARS, get_logger

LOG = get_logger("veritruth.data.preprocess")

MIN_CHARS = 20

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")

_QUOTE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


def clean_text(text: object) -> str:
    """Normalise a single document. Always returns a ``str`` (possibly empty)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text)
    value = html.unescape(value)
    value = _HTML_TAG_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = unicodedata.normalize("NFKC", value)
    for bad, good in _QUOTE_MAP.items():
        value = value.replace(bad, good)
    value = _CTRL_RE.sub(" ", value)
    value = _WS_RE.sub(" ", value).strip()
    return value[:MAX_TEXT_CHARS]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, filter and de-duplicate an ingested dataframe."""
    if df.empty:
        return df
    out = df.copy()
    before = len(out)

    out["text"] = out["text"].map(clean_text)
    out = out[out["text"].str.len() >= MIN_CHARS]
    after_len = len(out)

    out["_key"] = out["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    after_dedupe = len(out)

    out["label"] = out["label"].astype(int)
    out["source"] = out["source"].fillna("unknown").astype(str)
    out["language"] = out["language"].fillna("en").astype(str)
    out = out.reset_index(drop=True)

    LOG.info(
        "Cleaning: %s -> %s after length filter -> %s after dedupe (dropped %s)",
        before,
        after_len,
        after_dedupe,
        before - after_dedupe,
    )
    return out[["text", "label", "source", "language"]]


def main() -> pd.DataFrame:
    from src.data.ingest import load_raw

    raw = load_raw()
    cleaned = clean_frame(raw)
    print("\n--- STEP 2b VERIFICATION -------------------------------------")
    print(f"Rows in           : {len(raw)}")
    print(f"Rows out          : {len(cleaned)}")
    print(f"Min length        : {cleaned['text'].str.len().min()} (must be >= {MIN_CHARS})")
    print(f"Duplicates left   : {cleaned['text'].str.lower().duplicated().sum()} (must be 0)")
    print("---------------------------------------------------------------\n")
    return cleaned


if __name__ == "__main__":
    main()
