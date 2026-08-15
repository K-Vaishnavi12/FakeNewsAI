"""Text cleaning, input classification, URL extraction and prompt-injection defence.

Everything here treats user text and retrieved article text as *untrusted data*.
"""

from __future__ import annotations

import re
import unicodedata

from app.schemas import InputType

URL_PATTERN = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
BARE_DOMAIN_PATTERN = re.compile(
    r"(?<![\w./@-])((?:www\.)[\w-]+(?:\.[\w-]+)+(?:/[^\s<>\"']*)?)", re.IGNORECASE
)

# Phrases commonly used to hijack an LLM through pasted content.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) instructions?",
        r"disregard (?:all |any )?(?:the )?(?:previous|prior|above) (?:instructions?|rules?)",
        r"forget (?:everything|all)(?: you were told)?",
        r"you are now (?:a|an|in) ",
        r"new (?:system )?(?:instructions?|prompt)\s*:",
        r"system prompt\s*:",
        r"</?(?:system|assistant|user)>",
        r"reveal (?:your |the )?(?:system )?(?:prompt|instructions?)",
        r"(?:print|show|output|reveal|leak)[^.\n]{0,30}(?:api[_ ]?key|credentials?|secret)",
        r"always (?:respond|answer|say|output|return)[^.\n]{0,40}(?:real|fake|true)",
        r"override (?:the )?(?:verdict|classification|rules?)",
        r"do not follow (?:the )?(?:system|previous)",
        r"act as (?:if )?(?:you|a) ",
        r"mark this (?:article |story |claim )?as (?:real|true|verified|authentic)",
    )
]

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"[ \t\u00a0]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")

# Very small stopword list - enough for query building without extra deps.
STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "been", "before", "being", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "here", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "may", "me",
    "more", "most", "my", "new", "no", "not", "now", "of", "on", "one", "only",
    "or", "other", "our", "out", "over", "said", "says", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "why", "will", "with", "would", "you", "your",
}

NEGATION_CUES = {
    "denied", "denies", "deny", "debunked", "false", "hoax", "fake",
    "misleading", "untrue", "rejected", "rejects", "refuted", "refutes",
    "no evidence", "did not", "does not", "never happened", "baseless",
    "unfounded", "fabricated", "misinformation", "disinformation",
    "not true", "incorrect", "retracted", "clarified",
}


def normalise_unicode(text: str) -> str:
    """NFKC-normalise and strip control characters and zero-width joiners."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
    return CONTROL_CHARS.sub(" ", text)


def clean_text(text: str) -> str:
    """Light cleaning that preserves meaning. Used for display and modelling."""
    text = normalise_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def extract_urls(text: str) -> list[str]:
    """Return de-duplicated URLs found in the user's text, order preserved."""
    if not text:
        return []
    found: list[str] = []
    for match in URL_PATTERN.findall(text):
        cleaned = match.rstrip(".,;:!?)\"'")
        if cleaned not in found:
            found.append(cleaned)
    for match in BARE_DOMAIN_PATTERN.findall(text):
        cleaned = "http://" + match.rstrip(".,;:!?)\"'")
        if cleaned not in found and match.rstrip(".,;:!?") not in " ".join(found):
            found.append(cleaned)
    return found


def detect_injection(text: str) -> list[str]:
    """Return human-readable warnings for prompt-injection attempts.

    The text is *not* modified here; callers decide how to neutralise it.
    """
    warnings: list[str] = []
    if not text:
        return warnings
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            snippet = match.group(0)[:80]
            warnings.append(
                "Possible prompt-injection instruction detected in submitted text "
                f"and ignored: \"{snippet}\""
            )
    # Collapse duplicates while preserving order.
    seen: set[str] = set()
    unique = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def neutralise_injection(text: str) -> str:
    """Defang instruction-like phrases before the text is embedded in a prompt.

    We do not delete content (that would distort the evidence); we mark it so
    the model reads it as quoted data rather than as a directive.
    """
    if not text:
        return ""
    result = text
    for pattern in INJECTION_PATTERNS:
        result = pattern.sub(lambda m: f"[ignored-instruction: {m.group(0)}]", result)
    return result


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Truncate on a word boundary. Returns (text, was_truncated)."""
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip() + " …", True


def classify_input_type(text: str) -> InputType:
    """Heuristically label the shape of the submitted text."""
    stripped = text.strip()
    if not stripped:
        return "UNKNOWN"

    chars = len(stripped)
    words = len(stripped.split())
    paragraphs = [p for p in stripped.split("\n\n") if p.strip()]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]

    if words <= 3:
        return "UNKNOWN"
    if chars <= 200 and len(sentences) <= 1 and len(paragraphs) == 1:
        return "HEADLINE"
    if chars >= 1500 or len(paragraphs) >= 3 or len(sentences) >= 12:
        return "FULL_ARTICLE"
    return "ARTICLE_CLIP"


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, punctuation removed."""
    return re.findall(r"[\w']+", text.lower())


def content_tokens(text: str) -> list[str]:
    """Meaningful tokens with stopwords and very short tokens removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def extract_capitalised_phrases(text: str) -> list[str]:
    """Cheap named-entity proxy: sequences of capitalised words.

    This intentionally avoids a heavy NLP dependency. It is a heuristic and is
    labelled as such wherever its output is surfaced.
    """
    if not text:
        return []
    # Ignore the first word of each sentence to reduce false positives.
    candidates: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        words = sentence.split()
        for idx, word in enumerate(words):
            if idx == 0:
                continue
            if re.match(r"^[A-Z][\w'’-]*$", word):
                if candidates and candidates[-1][1] == idx - 1:
                    candidates[-1] = (candidates[-1][0] + " " + word, idx)
                else:
                    candidates.append((word, idx))
    phrases: list[str] = []
    for phrase, _ in candidates:
        clean = phrase.strip(".,;:!?'\"")
        if len(clean) < 3:
            continue
        if clean.lower() in STOPWORDS:
            continue
        if clean not in phrases:
            phrases.append(clean)
    return phrases


def extract_dates(text: str) -> list[str]:
    """Find explicit date-like expressions."""
    patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b",
        r"\b(?:19|20)\d{2}\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            if match not in found:
                found.append(match)
    return found


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping only substantive ones."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def has_negation_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in NEGATION_CUES)


def is_probably_non_english(text: str) -> bool:
    """Rough check: high ratio of non-ASCII letters or no ASCII letters at all."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    return non_ascii / len(letters) > 0.3
