# FakeNewsAI

Fake-news detection: a TF-IDF + Logistic Regression classifier cross-checked
against live news coverage, fused by a deterministic synthesis engine.

---

## Architecture

```
Browser (React/Vite :5173)
        │  POST /api/analyze          (Vite proxies /api/* → Flask :5000)
        ▼
Flask API (server/app.py)
  · CORS allow-list · per-IP rate limits · input-size cap · param validation
        ▼
LLMSearchAgent (server/agent.py)
        ├── Branch 1: ML classifier ──── server/ml/predict.py → fake_news_model.joblib
        └── Branch 2: live news ──────── server/news_fetcher.py (NewsAPI → Google News RSS)
        ▼
  Semantic corroboration → rule-based synthesis → [optional LLM prose pass]
        ▼
  JSON report {final_analysis, ml_classifier, news_sources, corroboration}
```

### Backend layout (`server/`, a proper Python package)

| Module | Responsibility |
|---|---|
| `__main__.py` | Entry point — `python -m server` (waitress). |
| `app.py` | Flask routes, CORS, rate limiting, request validation. |
| `agent.py` | Pipeline orchestration, corroboration scoring, synthesis. |
| `ml_model.py` | Adapter between routes and the ML package. |
| `ml/predict.py` | `NewsClassifier` singleton, inference, stylometrics, signals. |
| `ml/train_model.py` | Training pipeline; writes the model bundle + metadata. |
| `news_fetcher.py` | NewsAPI / RSS retrieval with TTL caching. |
| `local_llm.py` | Optional Hugging Face text generation (off by default). |
| `config.py` | All settings, from environment variables only. |
| `constants.py` | Single source of truth for shared lexicons. |
| `paths.py` | Filesystem sandboxing for request-reachable paths. |
| `cache.py` | Thread-safe TTL/LRU cache. |
| `logging_config.py` | `logging` setup; replaces `print`. |
| `cli.py` | Terminal interface — `python -m server.cli`. |

### Frontend (`client/`)

A single React 19 + Vite app: `index.html` → `src/main.jsx` → `src/App.jsx`.
There is no second frontend and no `api.js`; all requests are **relative**
(`/api/...`) and reach Flask through the Vite dev proxy.

---

## Prerequisites

- Python 3.9+
- Node.js 18+ and npm

---

## Setup

### 1. Clone

```bash
git clone https://github.com/<YOUR_USERNAME>/FakeNewsAI.git
cd FakeNewsAI
```

### 2. Configure environment

```bash
cp server/.env.example server/.env    # PowerShell: Copy-Item server\.env.example server\.env
```

Then edit `server/.env`. `server/.env` is gitignored — **never commit it**, and
never put a real key in `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `NEWSAPI_KEY` | *(empty)* | NewsAPI.org key. **Optional** — without it the app falls back to the keyless Google News RSS feed. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allow-list. Never `*`. |
| `MAX_INPUT_CHARS` | `10000` | Hard cap on analysed text. |
| `MAX_PAGE_SIZE` | `50` | Upper bound on `page_size`. |
| `RATE_LIMIT_ANALYZE` | `20 per minute;300 per hour` | Per-IP limit on the expensive endpoints. |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | Set to `redis://...` for multi-worker deployments. |
| `ENABLE_TRAIN_ENDPOINT` | `false` | Enables `/api/train_local`. Keep off in production. |
| `NEWS_CACHE_TTL_SECONDS` | `600` | TTL for cached news results. |
| `ENABLE_LLM` | `false` | Enables the optional LLM prose pass. |
| `LLM_PROVIDER` | `hf` | `hf` (local transformers) or `gemini`. |
| `LOCAL_HF_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Must be **instruction-tuned**. |
| `HF_TOKEN` / `GOOGLE_API_KEY` | *(empty)* | Only for gated models / Gemini. |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |

### 3. Backend

Run these from the **repository root** (not from `server/`) — `server` is a
package and is imported as one.

```bash
python -m venv server/venv
server\venv\Scripts\Activate.ps1     # Windows
source server/venv/bin/activate      # macOS / Linux

pip install -r server/requirements.txt
python -m server
```

Serves on `http://127.0.0.1:5000` via **waitress** (8 threads), so one slow
request no longer blocks every other client.

Useful flags:

```bash
python -m server --port 8000 --threads 16
python -m server --dev                 # Flask dev server + reloader (local only)
python -m server --host 0.0.0.0        # only behind a reverse proxy
```

Alternative production servers:

```bash
# gunicorn (Linux/macOS) — threads suit this I/O-bound workload
gunicorn --workers 2 --threads 4 --timeout 120 --bind 127.0.0.1:5000 server.app:app

# waitress explicitly
waitress-serve --host=127.0.0.1 --port=5000 --threads=8 server.app:app
```

> With more than one **worker process**, set `RATE_LIMIT_STORAGE_URI` to a
> Redis URI. The in-memory limiter is per-process, so N workers otherwise
> enforce N× the intended limit.

### 4. Frontend

In a second terminal:

```bash
cd client
npm install
npm run dev
```

Runs on `http://localhost:5173` and proxies `/api/*` to `http://127.0.0.1:5000`.

---

## API

All routes are served under `/api` (they are no longer double-registered at
the bare path).

| Method | Route | Notes |
|---|---|---|
| `GET` | `/api/health` | Status + model metadata. Rate-limit exempt. |
| `POST` | `/api/analyze` | `{text, page_size?}` → full report. |
| `POST` | `/api/run_prompt` | Legacy-shaped wrapper around `/api/analyze`. |
| `POST` | `/api/classify` | `{text}` → ML output only. |
| `POST` | `/api/search` | `{query, page_size?}` → scored articles. |
| `POST` | `/api/train_local` | Retrain. `403` unless `ENABLE_TRAIN_ENDPOINT=true`. |

Error codes: `400` invalid/oversized input, `403` disabled endpoint,
`413` body too large, `429` rate limited, `500` internal (details are logged
server-side, never returned to the client).

---

## Training

```bash
python -m server.ml.train_model --data-dir server/data
```

To pull datasets from the public Google Drive folder into `server/data`:

```bash
python -m server.datasets sync --purge-existing
```

One-command bootstrap + train (downloads first, then trains):

```bash
python -m server.ml.train_model --data-dir server/data --download-data --purge-existing-data
```

Datasets are **not** in the repository (they are large; see `.gitignore`).
Place the CSVs in `server/data/`: `WELFake_Dataset.csv`, `Fake.csv`,
`True.csv`, `BuzzFeed_*_news_content.csv`.

The saved bundle records `accuracy`, `eval_method`, `model_type`, `classes`
and `trained_at`. The API and UI read accuracy from this metadata — it is
never hardcoded.

### Evaluation methodology

Augmentation emits two samples per article (full body + headline). The split
is therefore **grouped by source article** (`GroupShuffleSplit`), so an
article's headline can never be in the training set while its body is in the
test set. The earlier `train_test_split(stratify=y)` allowed exactly that,
which inflated the reported score.

### Measured accuracy

Only the BuzzFeed set (**178 usable articles**) is currently present, so these
numbers are from a very small corpus and carry wide error bars.

| Method | Accuracy |
|---|---|
| Grouped 80/20 holdout (what the bundle records) | **83.33%** |
| 5-fold CV, grouped by article — *most trustworthy* | **67.12%** ±2.3 |
| 5-fold CV, random split (old leaky method) | 82.59% ±2.0 |
| Full articles only, no augmentation | 74.71% ±8.6 |
| Headlines only (closest to real user input) | 60.67% ±5.5 |

Majority-class baseline is 51.1%. Treat **~67%** as the realistic figure and
~61% for short headline claims. The single 83.33% holdout is optimistic: its
test set is only 36 articles. To improve this meaningfully, add the WELFake
and ISOT datasets to `server/data/` and retrain.

> **Data integrity note:** `PolitiFact_fake_news_content.csv` and
> `PolitiFact_real_news_content.csv` in `server/data/` are byte-identical, and
> both contain the *real* article set (rows carry ids like `Real_1-Webpage`).
> They are deliberately excluded from `DATASET_REGISTRY` and are not loaded by
> the training pipeline. Do not use them until a correct fake set is obtained.

---

## Testing

```bash
python -m pytest
```

---

## Security notes

**Implemented**

- **No secrets in code.** `config.py` has no default for any key; a missing
  `NEWSAPI_KEY` degrades to the keyless RSS feed rather than using a fallback.
- **CORS allow-list.** The request `Origin` is echoed only if it is on
  `CORS_ALLOWED_ORIGINS`. Wildcard `*` is never emitted.
- **Input cap.** Text over `MAX_INPUT_CHARS` is rejected with `400`; oversized
  bodies are refused at the WSGI layer via `MAX_CONTENT_LENGTH`.
- **Rate limiting.** Per-IP limits via Flask-Limiter on all heavy endpoints.
- **Path sandboxing.** `/api/train_local` accepts logical dataset names and a
  bare model filename — never a path. `server/paths.py` rejects absolute
  paths, `..`, drive letters, UNC paths and symlink escapes. This matters
  because `joblib.dump` writes a pickle, so an attacker-chosen output path
  would be a code-execution primitive.
- **Prompt-injection defence.** User text is wrapped in `<user_claim>` tags,
  literal closing tags are stripped so it cannot break out, and the system
  prompt declares the block to be untrusted data. Structurally, the LLM can
  only rewrite four allow-listed prose fields — verdict, `verdict_type`,
  `confidence_score` and `red_flags` come from the rule engine, so a
  successful injection still cannot flip a result.
- **XSS.** React escapes text, but `href` is not auto-sanitised. Every
  article URL passes through `safeUrl()` in `App.jsx`, which permits only
  `http:`/`https:` — blocking `javascript:` payloads from search results.
- **No debug server.** `debug=True` (the Werkzeug interactive debugger, an RCE
  hazard) is gone, and the default bind is loopback.
- **Error opacity.** Internal exception text is logged with a stack trace, not
  returned to clients.

**Not implemented — required before public deployment**

- **Authentication.** Every endpoint is unauthenticated; rate limiting is the
  only abuse control. To add auth: issue per-client API keys, validate them in
  a `before_request` hook, and switch the limiter's key function from
  `get_remote_address` to the API-key identity so limits are per-tenant rather
  than per-IP (trivially bypassed via proxies).
- **HTTPS / reverse proxy.** Terminate TLS at nginx or similar. If you do, wrap
  the app in `ProxyFix` so `get_remote_address` reads the real client IP from
  `X-Forwarded-For` — otherwise every request appears to come from the proxy
  and shares a single rate-limit bucket.
- **Persistent abuse logging.** Logs currently go to stderr. Ship them to a
  file or aggregator; rejected-oversize and rate-limit events are already
  logged at `WARNING` with the client address.
