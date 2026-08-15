# TruthLens AI

**See Beyond the Headlines. Discover the Truth.**

TruthLens AI is an evidence-first news verification system. A user pastes a news
headline, clipping or full article, and the system retrieves related reporting
from a live news search, measures how closely each retrieved article matches the
submitted claim, and returns a verdict in which **every statement is traceable to
a labelled source**.

---

## 1. What this project does — and what it deliberately does not

| It does | It does not |
|---|---|
| Retrieve related articles and expose every URL | Decide what is objectively true |
| Measure how closely each article matches the claim | Act as a fact-checking authority |
| Separate a writing-style model signal from real evidence | Treat a model's opinion as proof |
| State clearly when the evidence is insufficient | Treat "no article found" as "fake" |
| Show which source contributed to which sentence | Bypass paywalls or scrape restricted content |

### The core design principle

**Evidence is separated from interpretation.**

Four source types are kept strictly distinct and are never blended:

| Type | ID | Meaning |
|---|---|---|
| `USER_SUBMITTED_TEXT` | `USER-001` | What the user pasted. Unverified. |
| `NEWS_API_RESULT` | `NEWS-001`… | Retrieved articles. **The only external evidence.** |
| `MODEL_OUTPUT` | `MODEL-001` | The ML writing-style signal. Never proof. |
| `AI_EXPLANATION` | `AI-001` | The LLM interpretation. Not an independent source. |

"Needs Verification" is a correct and common answer. It means the retrieved
evidence was not sufficient — **not** that the claim is false.

---

## 2. Technology used

### Backend

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| ML | scikit-learn (TF-IDF, Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest) |
| Model persistence | joblib |
| Numerics | NumPy, SciPy |
| HTTP client | requests |
| Config / secrets | python-dotenv |
| LLM client | `openai` SDK pointed at NVIDIA's OpenAI-compatible endpoint |
| Testing | pytest |

### Frontend

| Layer | Technology |
|---|---|
| Primary UI | Vite 8 + Tailwind CSS 4 + vanilla JavaScript (ES modules) |
| Alternative UI | Streamlit |
| Runtime | Node.js 20+ |

### External services

| Service | Purpose |
|---|---|
| **NewsAPI** (`newsapi.org`) | Retrieval of related news articles |
| **NVIDIA NIM** (`integrate.api.nvidia.com`) | `meta/llama-3.1-8b-instruct` for the grounded explanation |

### The ML ensemble

Four members, combined by soft voting:

1. TF-IDF → Logistic Regression
2. TF-IDF → Linear SVM (probability-calibrated)
3. TF-IDF → Multinomial Naive Bayes
4. Engineered stylometric features → Random Forest *(XGBoost used automatically if installed)*

**Explainability:** exact linear-model attribution —
`contribution(token) = tfidf_value(token) × coefficient(token)`.
For a linear model this is a faithful decomposition of the decision score.

> **Accuracy disclaimer.** The bundled corpus (`app/ml/dataset.py`) is
> **synthetic and template-generated**, so the app runs without a third-party
> dataset download. The classifier therefore learns *writing style*, not truth.
> Training reports ~100% accuracy — **that number is meaningless**, it is
> memorising templates. This is exactly why the ML signal can never determine
> the verdict. To train on real data, place a `text,label` CSV at
> `data/train.csv` (label `1` = REAL, `0` = FAKE); it takes precedence
> automatically.

> **Naming note.** The landing page currently advertises "SHAP Linguistic
> Attribution", "ChromaDB Source Retrieval" and "DistilBERT". Those technologies
> are **not** part of this implementation. The system uses TF-IDF + linear/tree
> models, linear-coefficient attribution, and NewsAPI retrieval. The marketing
> copy should be corrected before any external presentation.

---

## 3. System architecture

```
                    ┌──────────────────────────────────────┐
                    │  TruthLens AI UI (Vite + Tailwind)   │
                    │  splash → verification desk → report │
                    └──────────────────┬───────────────────┘
                                       │ HTTP (JSON)
                                       ▼
                    ┌──────────────────────────────────────┐
                    │        FastAPI backend               │
                    │  /health  /analyze  /feedback        │
                    │  (holds all API keys)                │
                    └──────────────────┬───────────────────┘
                                       ▼
                            AnalysisPipeline
                                       │
   ┌───────────────┬───────────────────┼───────────────────┬──────────────────┐
   ▼               ▼                   ▼                   ▼                  ▼
Input parser   ML ensemble       Query generator      Relevance scorer   NVIDIA LLM
(A1)           (style signal)    (≤3 typed queries)   (weighted, A3)     (explanation)
   │               │                   │                   │                  │
headline        LogReg / SVM      HEADLINE            0.30·title          grounded in
body            NB / RF           ENTITY_EVENT        0.25·claim          evidence only,
publisher       soft vote         DISTINCTIVE_PHRASE  0.20·entity         then guardrailed
author          + token                │              0.10·date
date            attribution            ▼              0.10·event
entities                          NewsAPI adapter     0.05·source
dates                             (timeout, 429,           │
numbers                            401, malformed,         ▼
claim sentences                    dedupe by          Decision engine (A6)
   │                               canonical URL)     six-case table
   └───────────────┬───────────────────┴───────────────────┘
                   ▼
        ┌──────────────────────────────────────────────┐
        │  Response: verdict + 4 separate scores +      │
        │  claim→source map + full source provenance    │
        └──────────────────────────────────────────────┘
```

### Request flow

1. **Input parsing** — extract headline, body, publisher, author, date, URLs,
   entities, dates, numbers and claim sentences.
2. **ML style signal** — soft-vote ensemble plus per-token attribution.
   Disagreement between members caps confidence at 55.
3. **Query generation** — at most three focused queries. Clickbait and filler
   are stripped. Each retrieved article records the query that found it.
4. **Retrieval** — NewsAPI, de-duplicated by canonical URL (tracking parameters
   removed). Every failure mode is contained.
5. **Relevance scoring** — the weighted model above, replacing whole-document
   TF-IDF cosine.
6. **Claim mapping** — the clip is split into claims; each is scored
   independently so citations are per-sentence.
7. **Decision** — the six-case table below.
8. **Explanation** — NVIDIA generates prose grounded strictly in the evidence,
   then guardrails clamp it.

### Relevance scoring

```
relevance = 0.30·title_similarity
          + 0.25·claim_similarity
          + 0.20·entity_overlap
          + 0.10·date_overlap
          + 0.10·event_overlap
          + 0.05·source_metadata_match
```

| Band | Meaning |
|---|---|
| ≥ 0.62 | Strongly relevant |
| 0.42 – 0.61 | Relevant → counts as evidence |
| 0.20 – 0.41 | Weakly relevant → context only |
| < 0.20 | Unrelated → shown with its link, never used as evidence |

> **Why not plain TF-IDF?** NewsAPI truncates article `content`, so comparing a
> full pasted article against a short stub systematically under-scored correct
> matches. A headline reporting the *same event* scored 0.33 against a 0.42
> threshold, while unrelated articles sharing vocabulary scored 0.5+. The
> weighted model fixed both directions.

### Decision table

| Case | Condition | Status | Confidence |
|---|---|---|---|
| 1 | ≥2 independent relevant sources agree | Supported by Retrieved Evidence | 75–95 |
| 2 | 1 relevant source, or metadata-only match | Partially Supported | 55–74 |
| 3 | Style model has an opinion, no relevant source | Needs Verification | 30–55 |
| 4 | Relevant sources contradict the claim | Contradicted by Retrieved Evidence | 70–95 |
| 5 | Search unavailable / rate-limited / paywalled | Unable to Verify | ≤ 50 |
| 6 | Ensemble members disagree strongly | Needs Verification | ≤ 55 |

A claim is **never** labelled fake merely because no article was found, the
publisher is obscure, the article is paywalled or new, or the style score is low.

### The four separate scores

Never blended into one misleading number:

| Score | Meaning |
|---|---|
| `ml_style_signal` | Writing-pattern signal. **Not** a probability of truth. |
| `evidence_relevance` | Match quality of the best retrieved source |
| `source_agreement_score` | How consistently independent publishers agree |
| `verification_confidence` | Confidence in the **verification outcome** |

The gauge is labelled *Verification Confidence*, never "Truth Score".

### Guardrails on the LLM

The model's output is untrusted. `enforce_guardrails()` will:

- strip cited source IDs that were not retrieved (including `MODEL-001` and
  `AI-001` — it may not cite itself or the classifier);
- overwrite any relation it asserts with the measured relation;
- recompute source agreement and `used_in_final_answer` from measurement;
- correct the verdict downward when it over-claims;
- clamp confidence to the evidence ceiling.

Every correction is recorded in `system_warnings` and displayed in the UI.

### Prompt-injection defence

Pasted text and retrieved article text are treated as untrusted data:

1. `detect_injection()` reports attempts as system warnings.
2. `neutralise_injection()` rewrites directives as `[ignored-instruction: …]`.
3. Untrusted text is fenced with explicit markers in the prompt.
4. The system prompt forbids following embedded instructions.
5. Guardrails mean a successful injection still cannot change the verdict.

---

## 4. Project structure

```
.
├── app/
│   ├── config.py                  # env-driven settings; secrets never serialised
│   ├── schemas.py                 # Pydantic API contract
│   ├── text_utils.py              # cleaning, URL extraction, injection defence
│   ├── pipeline.py                # end-to-end orchestration
│   ├── main.py                    # FastAPI backend
│   ├── ml/
│   │   ├── dataset.py             # synthetic corpus + CSV loader
│   │   ├── features.py            # engineered stylometric features
│   │   ├── train.py               # training script → artifacts/
│   │   ├── predictor.py           # soft-voting ensemble
│   │   └── attribution.py         # linear-model token attribution
│   ├── news/
│   │   ├── query_generator.py     # ≤3 typed queries
│   │   ├── adapter.py             # NewsAPI client, all failure modes
│   │   └── normalize.py           # normalization, canonical-URL dedupe
│   ├── analysis/
│   │   ├── input_parser.py        # structured clip parsing
│   │   ├── relevance.py           # weighted relevance scoring
│   │   ├── claims.py              # claim extraction + source mapping
│   │   ├── verification.py        # decision table + score separation
│   │   ├── verdict.py             # assembly + guardrails
│   │   └── provenance.py          # source provenance assembly
│   └── llm/
│       ├── prompts.py             # system prompt + evidence payload
│       └── nvidia_client.py       # OpenAI-compatible client, JSON recovery
├── FakeNewsAI/FakeNewsAI/client/  # TruthLens AI frontend (Vite)
│   ├── src/main.js                # UI + provenance sections
│   ├── src/api.js                 # backend client + response mapping
│   ├── src/style.css              # design system
│   └── public/                    # background assets
├── frontend/streamlit_app.py      # alternative Streamlit UI
├── tests/test_veritascheck.py     # 32 tests
├── docs/TEST_CASES.md             # full test-case report
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 5. Execution — Windows

> PowerShell. Run from the project root.

### 5.1 Prerequisites

- Python 3.11+ — <https://www.python.org/downloads/>
- Node.js 20+ — <https://nodejs.org/>

```powershell
python --version
node --version
```

### 5.2 Backend setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 5.3 Configure secrets

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in:

```ini
NEWS_API_KEY=your_key_from_newsapi_org
NVIDIA_API_KEY=your_key_from_build_nvidia_com
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

### 5.4 Train the ML ensemble

```powershell
python -m app.ml.train
```

Optional:

```powershell
python -m app.ml.train --samples-per-class 2000
python -m app.ml.train --csv data\train.csv
python -m app.ml.train --augment-from-newsapi
```

### 5.5 Run the backend

```powershell
uvicorn app.main:app --reload
```

→ <http://127.0.0.1:8000>  ·  docs at <http://127.0.0.1:8000/docs>

### 5.6 Run the TruthLens frontend

New PowerShell window:

```powershell
cd FakeNewsAI\FakeNewsAI\client
npm install
npm run dev
```

→ <http://localhost:5173> *(Vite selects the next free port if 5173 is taken)*

### 5.7 Alternative Streamlit UI

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend\streamlit_app.py
```

### 5.8 Tests

```powershell
python -m pytest tests -v
```

---

## 6. Execution — macOS

> Terminal (zsh/bash). Run from the project root.

### 6.1 Prerequisites

```bash
brew install python@3.11 node
python3 --version
node --version
```

### 6.2 Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.3 Configure secrets

```bash
cp .env.example .env
nano .env
```

Fill in:

```ini
NEWS_API_KEY=your_key_from_newsapi_org
NVIDIA_API_KEY=your_key_from_build_nvidia_com
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

### 6.4 Train the ML ensemble

```bash
python -m app.ml.train
```

Optional:

```bash
python -m app.ml.train --samples-per-class 2000
python -m app.ml.train --csv data/train.csv
python -m app.ml.train --augment-from-newsapi
```

### 6.5 Run the backend

```bash
uvicorn app.main:app --reload
```

→ <http://127.0.0.1:8000>  ·  docs at <http://127.0.0.1:8000/docs>

### 6.6 Run the TruthLens frontend

New terminal tab:

```bash
cd FakeNewsAI/FakeNewsAI/client
npm install
npm run dev
```

→ <http://localhost:5173>

### 6.7 Alternative Streamlit UI

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

### 6.8 Tests

```bash
python -m pytest tests -v
```

---

## 7. API reference

### `POST /analyze`

```json
{ "text": "Your headline or article", "max_sources": 10 }
```

Returns `request_id`, `analyzed_at`, `user_input`, `parsed_input`,
`final_analysis`, `verification_scores`, `structured_explanation`, `claims`,
`ml_result`, `news_search`, `source_provenance`, `system_warnings`.

### `GET /health`

Returns booleans only — `news_api_configured`, `nvidia_configured`,
`ml_models_available`. Never a key, prefix or length.

### `POST /feedback`

Accepts UI accuracy feedback. Logged server-side only; deliberately **not** fed
back into the model or the verdict.

---

## 8. Choosing an NVIDIA model

Measured against the live endpoint with the full ~13 KB evidence prompt:

| Model | Result | Verdict |
|---|---|---|
| `meta/llama-3.1-8b-instruct` | 8.8 s, valid JSON | **Recommended default** |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | 59 s, valid JSON | Works; raise `NVIDIA_TIMEOUT_SECONDS` |
| `meta/llama-3.1-70b-instruct` | times out | Fine on tiny prompts only |
| `meta/llama-3.3-70b-instruct` | times out (>360 s) | Unusable on the tested account |

If the LLM is slow, unreachable or returns malformed JSON, the deterministic
engine produces the complete answer instead and the UI says so. **The verdict
never depends on the LLM.**

---

## 9. Security

- Secrets are read from environment variables only. `.env` is gitignored.
- The NewsAPI key is sent as an `X-Api-Key` **header**, never a query parameter,
  so it cannot leak into logs or proxy traces.
- The frontend holds no credentials; it only calls the backend.
- CORS is restricted to loopback origins.
- A regression test asserts no key material appears in any response.
- No paywall bypass and no scraping of restricted content.

> **If a key has ever been pasted into a chat, an issue tracker or a commit,
> rotate it.** Both providers allow revoking and reissuing from the dashboard.

---

## 10. Testing

```
pytest tests -v      →  32 tests
```

Covers matching sources, no match, unrelated results, conflicting publishers,
NewsAPI timeout / rate limit / invalid key / malformed body, missing URL and
publisher fields, duplicate URLs, user-pasted URLs, prompt injection, malformed
and empty LLM responses, ML disagreement, empty / short / long / non-English
input, and source-citation mapping.

Full report: `docs/TEST_CASES.md`.

---

## 11. Known limitations

- Similarity is TF-IDF based; heavy paraphrase and cross-language matches are
  missed. There are no embeddings in this build.
- Entity extraction is a capitalisation heuristic, not a trained NER model.
- NewsAPI's developer tier covers roughly the last month, truncates `content` to
  ~200 characters, and restricts non-localhost origins. Scoring therefore runs
  mostly on titles and descriptions.
- Contradiction detection is keyword-based; it can miss a denial phrased without
  a recognised cue word.
- The `language=en` filter means non-English claims retrieve little.
- The source-quality hint is a static domain list, not a reliability rating.
- The bundled ML corpus is synthetic — see the accuracy disclaimer in §2.

---

## 12. Future development

### Retrieval quality
- Replace TF-IDF with sentence embeddings for paraphrase-tolerant matching.
- Add a vector store for semantic retrieval over a persistent article corpus.
- Add multiple providers (GDELT, Bing News, Google News RSS) with fallback, to
  reduce single-provider blind spots.
- Add publisher-level deduplication for syndicated wire copy.

### Verification depth
- Integrate the Google Fact Check Tools API for published fact-check records.
- Named-entity recognition with a trained model instead of the capitalisation
  heuristic.
- Stance detection (agree / disagree / discuss) instead of keyword-based
  contradiction cues.
- Temporal reasoning so an old article is not treated as evidence for a new
  event.
- Cross-language verification with translation before retrieval.

### Machine learning
- Train on a real labelled corpus and publish an honest evaluation with
  precision, recall and a confusion matrix.
- Add proper SHAP attribution so the interface claim becomes accurate.
- Calibrate probabilities and report an uncertainty interval.
- Periodic retraining as language patterns drift.

### Product
- User accounts with verification history.
- Shareable permalinks for a completed report.
- Browser extension for in-page verification.
- Batch verification via file upload.
- Exportable PDF report with the full source provenance.
- Accessibility pass (WCAG 2.1 AA) and keyboard navigation audit.

### Engineering
- Persist feedback to a database and build a review queue.
- Cache retrieval results with a TTL to reduce API usage.
- Rate limiting and request quotas.
- Dockerfile and docker-compose for one-command startup.
- CI pipeline running tests and the frontend build on every push.
- Structured logging and observability.

---

## 13. Licence and attribution

Built for the Cognizant hackathon.

News content is retrieved through NewsAPI and remains the property of the
respective publishers. Article links are always presented in full so users can
read the original reporting at the source.
