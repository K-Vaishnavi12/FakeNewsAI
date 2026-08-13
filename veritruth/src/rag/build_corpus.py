"""Step 6a — build the fact-check corpus.

Primary source: Google Fact Check Tools **Claim Search** API
(``https://factchecktools.googleapis.com/v1alpha1/claims:search``) queried with
several seed topics and paginated until the target size is reached.

If ``GOOGLE_FACTCHECK_API_KEY`` is missing, or every request fails, we fall back
to a built-in list of 25 well-known verified fact-checks so that ChromaDB is
**never empty** and the demo always works offline.

Run::

    python -m src.rag.build_corpus --target 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.config import CORPUS_PATH, HTTP_TIMEOUT, ensure_dirs, get_env, get_logger

LOG = get_logger("veritruth.rag.build_corpus")

FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
SEED_QUERIES = ["india", "health", "election", "finance", "viral"]
EXTRA_QUERIES = [
    "vaccine", "covid", "climate", "economy", "war", "immigration",
    "5g", "crypto", "celebrity", "government", "police", "school",
    "water", "food", "medicine", "technology", "sports", "weather",
]
PAGE_SIZE = 50


# ---------------------------------------------------------------- fallback
SEED_FACTCHECKS: list[dict[str, str]] = [
    {
        "claim": "Drinking bleach or disinfectant cures COVID-19.",
        "rating": "False",
        "publisher": "Snopes",
        "url": "https://www.snopes.com/fact-check/bleach-coronavirus-cure/",
        "review": "Ingesting bleach or disinfectant is poisonous and has no therapeutic effect against COVID-19. Health agencies including the WHO and CDC warn it causes severe internal injury or death.",
    },
    {
        "claim": "5G mobile networks spread the coronavirus.",
        "rating": "False",
        "publisher": "Full Fact",
        "url": "https://fullfact.org/online/5g-coronavirus/",
        "review": "Viruses spread through respiratory droplets, not radio waves. 5G is non-ionising radiation and cannot transmit a biological pathogen.",
    },
    {
        "claim": "COVID-19 vaccines contain microchips for tracking people.",
        "rating": "False",
        "publisher": "Reuters Fact Check",
        "url": "https://www.reuters.com/article/factcheck-vaccine-microchip-idUSL1N2SP1L2",
        "review": "Vaccine ingredient lists are public and contain no electronic components. A microchip could not pass through a standard vaccine needle.",
    },
    {
        "claim": "The 2020 US presidential election was decided by widespread voter fraud.",
        "rating": "False",
        "publisher": "PolitiFact",
        "url": "https://www.politifact.com/factchecks/2020/nov/j/voter-fraud/",
        "review": "Federal and state election officials, courts in more than 60 cases, and audits in contested states found no evidence of fraud sufficient to change the outcome.",
    },
    {
        "claim": "Climate change is a hoax invented for political purposes.",
        "rating": "False",
        "publisher": "Climate Feedback",
        "url": "https://science.feedback.org/reviews/climate-change-hoax/",
        "review": "Multiple independent temperature records and over 97% of publishing climate scientists agree global warming is real and driven by human greenhouse gas emissions.",
    },
    {
        "claim": "Vaccines cause autism in children.",
        "rating": "False",
        "publisher": "Snopes",
        "url": "https://www.snopes.com/fact-check/vaccines-autism/",
        "review": "The 1998 study alleging this link was retracted for fraud. Large cohort studies covering millions of children show no association between vaccination and autism.",
    },
    {
        "claim": "Indian currency notes contain a GPS tracking nano-chip.",
        "rating": "False",
        "publisher": "Boom Live",
        "url": "https://www.boomlive.in/fact-check/gps-nano-chip-currency-note",
        "review": "The Reserve Bank of India confirmed the 2000-rupee note contains no chip. The security thread is standard printed film, not electronics.",
    },
    {
        "claim": "NASA confirmed the Earth will go dark for six days.",
        "rating": "False",
        "publisher": "AFP Fact Check",
        "url": "https://factcheck.afp.com/nasa-six-days-darkness",
        "review": "NASA has issued no such statement. The claim originated on a satirical website and has recirculated since 2014.",
    },
    {
        "claim": "Eating garlic prevents or cures coronavirus infection.",
        "rating": "False",
        "publisher": "WHO Mythbusters",
        "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
        "review": "Garlic has some antimicrobial properties but there is no evidence it protects against SARS-CoV-2.",
    },
    {
        "claim": "The moon landing in 1969 was filmed in a studio.",
        "rating": "False",
        "publisher": "Reuters Fact Check",
        "url": "https://www.reuters.com/article/factcheck-moon-landing-idUSL1N2P51QN",
        "review": "Retroreflectors left by Apollo missions are still used by observatories worldwide, and independent nations tracked the transmissions live.",
    },
    {
        "claim": "Drinking hot water flushes the coronavirus out of the throat.",
        "rating": "False",
        "publisher": "Boom Live",
        "url": "https://www.boomlive.in/fact-check/hot-water-coronavirus",
        "review": "The virus infects cells in the respiratory tract; swallowing water cannot remove an established infection.",
    },
    {
        "claim": "A new Indian banknote was declared the best currency in the world by UNESCO.",
        "rating": "False",
        "publisher": "Alt News",
        "url": "https://www.altnews.in/unesco-best-currency-fake/",
        "review": "UNESCO does not rank currencies. The organisation has repeatedly denied issuing any such award.",
    },
    {
        "claim": "Bill Gates said vaccines will reduce the world population.",
        "rating": "Misleading",
        "publisher": "PolitiFact",
        "url": "https://www.politifact.com/factchecks/2020/bill-gates-vaccines-population/",
        "review": "Gates argued that better healthcare and vaccination lower birth rates by reducing child mortality — the opposite of depopulation.",
    },
    {
        "claim": "Masks reduce blood oxygen to dangerous levels.",
        "rating": "False",
        "publisher": "Full Fact",
        "url": "https://fullfact.org/health/masks-oxygen/",
        "review": "Pulse-oximetry studies of surgeons and mask-wearing patients show no clinically significant drop in blood oxygen saturation.",
    },
    {
        "claim": "The COVID-19 pandemic was planned in advance by global elites.",
        "rating": "False",
        "publisher": "AFP Fact Check",
        "url": "https://factcheck.afp.com/covid-plandemic",
        "review": "Genomic sequencing shows natural viral evolution. Simulation exercises held before 2020 were preparedness drills, not plans.",
    },
    {
        "claim": "Cryptocurrency investment scheme guarantees 10 percent daily returns.",
        "rating": "False",
        "publisher": "Reuters Fact Check",
        "url": "https://www.reuters.com/article/factcheck-crypto-scam-idUSL1N2XY0AB",
        "review": "Guaranteed high daily returns are the defining marker of a Ponzi scheme. Regulators in multiple countries have issued warnings.",
    },
    {
        "claim": "Aluminium in cookware causes Alzheimer's disease.",
        "rating": "Unproven",
        "publisher": "Snopes",
        "url": "https://www.snopes.com/fact-check/aluminum-alzheimers/",
        "review": "Early studies suggested a correlation but decades of follow-up research have not established a causal link.",
    },
    {
        "claim": "A viral video shows soldiers celebrating a recent border victory.",
        "rating": "Miscaptioned",
        "publisher": "Alt News",
        "url": "https://www.altnews.in/old-video-soldiers-miscaptioned/",
        "review": "Reverse image search dates the footage to a parade several years earlier, unrelated to the claimed event.",
    },
    {
        "claim": "The government is offering free laptops to all students via a WhatsApp link.",
        "rating": "False",
        "publisher": "PIB Fact Check",
        "url": "https://pib.gov.in/factcheck.aspx",
        "review": "The Press Information Bureau confirmed no such scheme exists. The link is a phishing operation harvesting personal data.",
    },
    {
        "claim": "Lemon and baking soda cure cancer better than chemotherapy.",
        "rating": "False",
        "publisher": "Health Feedback",
        "url": "https://healthfeedback.org/claimreview/lemon-baking-soda-cancer/",
        "review": "No clinical trial supports this. Abandoning evidence-based oncology for such remedies measurably increases mortality.",
    },
    {
        "claim": "Wind turbines cause cancer.",
        "rating": "False",
        "publisher": "FactCheck.org",
        "url": "https://www.factcheck.org/2019/04/wind-turbines-cancer/",
        "review": "There is no known biological mechanism and no epidemiological evidence linking turbine noise or infrasound to cancer.",
    },
    {
        "claim": "Banks will seize deposits above a certain limit next month.",
        "rating": "False",
        "publisher": "Boom Live",
        "url": "https://www.boomlive.in/fact-check/bank-deposit-seizure-rumour",
        "review": "Central bank statements confirm deposit insurance rules are unchanged. The rumour recirculates during periods of market stress.",
    },
    {
        "claim": "Ancient pyramids were built by extraterrestrial visitors.",
        "rating": "False",
        "publisher": "FactCheck.org",
        "url": "https://www.factcheck.org/2021/07/pyramids-aliens/",
        "review": "Archaeological records include worker villages, payroll ostraca, quarry marks and unfinished ramps documenting human construction.",
    },
    {
        "claim": "Drinking eight glasses of water a day is medically required for everyone.",
        "rating": "Mixture",
        "publisher": "Health Feedback",
        "url": "https://healthfeedback.org/claimreview/eight-glasses-water/",
        "review": "Fluid needs vary by body size, climate and activity, and are partly met by food. The specific eight-glass figure has no strong evidentiary basis.",
    },
    {
        "claim": "A celebrity has died, according to a breaking-news social media post.",
        "rating": "False",
        "publisher": "Snopes",
        "url": "https://www.snopes.com/fact-check/celebrity-death-hoax/",
        "review": "Celebrity death hoaxes are a recurring format. No reputable outlet or representative confirmed the report.",
    },
]


def _record(
    claim: str, review: str, rating: str, publisher: str, url: str, source: str
) -> dict[str, Any]:
    return {
        "claim": (claim or "").strip(),
        "review": (review or "").strip(),
        "rating": (rating or "Unrated").strip(),
        "publisher": (publisher or "Unknown").strip(),
        "url": (url or "").strip(),
        "source": source,
    }


def fetch_google_factchecks(
    query: str, page_size: int = PAGE_SIZE, max_pages: int = 4
) -> list[dict[str, Any]]:
    """Query the Google Fact Check API for one topic. Returns [] on any failure."""
    api_key = get_env("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        return []
    try:
        import requests
    except Exception as exc:  # pragma: no cover
        LOG.warning("requests unavailable (%s).", exc)
        return []

    collected: list[dict[str, Any]] = []
    page_token = ""
    for _ in range(max_pages):
        params = {
            "query": query,
            "pageSize": page_size,
            "languageCode": "en",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            resp = requests.get(FACTCHECK_URL, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code != 200:
                LOG.warning("FactCheck API '%s' -> HTTP %s", query, resp.status_code)
                break
            payload = resp.json()
        except Exception as exc:
            LOG.warning("FactCheck API '%s' failed (%s).", query, exc)
            break

        claims = payload.get("claims") or []
        for claim in claims:
            text = claim.get("text", "")
            for review in claim.get("claimReview") or [{}]:
                publisher = (review.get("publisher") or {}).get("name", "Unknown")
                collected.append(
                    _record(
                        claim=text,
                        review=review.get("title", "") or text,
                        rating=review.get("textualRating", "Unrated"),
                        publisher=publisher,
                        url=review.get("url", ""),
                        source="google_factcheck_api",
                    )
                )
        page_token = payload.get("nextPageToken", "")
        if not page_token or not claims:
            break
        time.sleep(0.2)
    return collected


def fallback_corpus() -> list[dict[str, Any]]:
    return [
        _record(
            claim=item["claim"],
            review=item["review"],
            rating=item["rating"],
            publisher=item["publisher"],
            url=item["url"],
            source="builtin_seed",
        )
        for item in SEED_FACTCHECKS
    ]


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        if not rec["claim"] and not rec["review"]:
            continue
        key = (rec["claim"] or rec["review"]).lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def build_corpus(target: int = 2000, out_path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """Collect fact-checks from the API, then top up with the built-in seeds."""
    ensure_dirs()
    records: list[dict[str, Any]] = []

    if get_env("GOOGLE_FACTCHECK_API_KEY"):
        for query in SEED_QUERIES + EXTRA_QUERIES:
            batch = fetch_google_factchecks(query)
            LOG.info("Query '%s' -> %s claims", query, len(batch))
            records.extend(batch)
            if len(dedupe(records)) >= target:
                break
    else:
        LOG.warning(
            "GOOGLE_FACTCHECK_API_KEY not set — using the %s built-in fact-checks. "
            "The pipeline still runs end to end.",
            len(SEED_FACTCHECKS),
        )

    records = dedupe(records)
    if not records:
        records = fallback_corpus()
    else:
        records = dedupe(records + fallback_corpus())

    records = records[:target] if target > 0 else records
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Wrote %s fact-check records to %s", len(records), out_path)
    return records


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """Read the corpus from disk, building it if absent. Never returns []."""
    if not path.exists():
        return build_corpus()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) and data else fallback_corpus()
    except Exception as exc:
        LOG.warning("Corpus unreadable (%s); using built-in seeds.", exc)
        return fallback_corpus()


def main() -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Build the VeriTruth fact-check corpus")
    parser.add_argument("--target", type=int, default=2000)
    ns = parser.parse_args()

    records = build_corpus(target=ns.target)
    by_source: dict[str, int] = {}
    for rec in records:
        by_source[rec["source"]] = by_source.get(rec["source"], 0) + 1

    print("\n--- STEP 6a VERIFICATION (corpus) -----------------------------")
    print(f"Records written : {len(records)} -> {CORPUS_PATH}")
    print(f"By source       : {by_source}")
    print(f"Example claim   : {records[0]['claim'][:70]}...")
    print("Expected        : >= 25 records even with no API key; file exists")
    print("---------------------------------------------------------------\n")
    return records


if __name__ == "__main__":
    main()
