# VeriTruth

Agentic, explainable fake-news investigation. Paste a headline or article and get
a verdict, a calibrated trust score, the exact words that drove the decision, and
a cited, evidence-backed explanation.

* **Verdict** — Real / Suspicious / Fake
* **Trust score** — calibrated 0–100 (Platt / temperature scaling, not a raw softmax)
* **Explanation** — SHAP token attributions, highlighted in the article text
* **Evidence** — an LLM agent decomposes the article into atomic claims, retrieves
  fact-checks through RAG + the Google Fact Check API, and returns a cited verdict

Everything degrades gracefully: no Gemini key, no fact-check key, no GPU, no
internet — the system still runs end to end and tells you it is in degraded mode.

---

## Architecture

```
                          ┌──────────────────────────┐
                          │   Streamlit UI  :8501    │
                          │  verdict · SHAP · cites  │
                          └────────────┬─────────────┘
                                       │ HTTP/JSON
                          ┌────────────▼─────────────┐
                          │    FastAPI API  :8000    │
                          │  /predict   /explain     │
                          │  /investigate  /feedback │
                          │  /health                 │
                          └──┬──────────┬─────────┬──┘
                             │          │         │
          ┌──────────────────▼──┐  ┌────▼──────┐  ┌▼──────────────┐
          │   MODEL LAYER       │  │  AGENT    │  │  STORAGE      │
          │ DistilBERT (or      │  │  LAYER    │  │  SQLite       │
          │ TF-IDF + LogReg)    │  │ LangGraph │  │  feedback     │
          │ + calibration       │  │           │  └───────────────┘
          │ + SHAP / LIME       │  │  decompose│
          └─────────────────────┘  │     ↓     │
                                   │   verify  │
                                   │     ↓     │
                                   │ synthesize│
                                   └─────┬─────┘
                                         │ MCP (stdio)
        ┌────────────────┬───────────────┼───────────────┬────────────────┐
        │                │               │               │                │
┌───────▼──────┐ ┌───────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐
│ classifier   │ │ evidence     │ │ factcheck    │ │ explainer    │
│ -server      │ │ -server      │ │ -server      │ │ -server      │
│ classify_    │ │ search_      │ │ search_fact_ │ │ explain_     │
│ news()       │ │ evidence()   │ │ checks()     │ │ prediction() │
└──────────────┘ └──────┬───────┘ └──────┬───────┘ └──────────────┘
                        │                │
                 ┌──────▼──────┐  ┌──────▼──────────────┐
                 │  ChromaDB   │  │ Google Fact Check   │
                 │ factchecks  │  │ Tools API           │
                 │ MiniLM-L6   │  └─────────────────────┘
                 └─────────────┘
```

**`/investigate` request flow:** classify → decompose into ≤4 atomic claims →
retrieve evidence per claim via MCP tools → synthesize a 3-sentence cited verdict
→ merge with SHAP token weights → return one JSON payload.

---

## Quick start

```bash
git clone <your-repo-url>
cd veritruth

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
copy .env.example .env          # cp on macOS/Linux — keys are optional

python -m src.data.split         # build splits (synthetic fallback if no raw data)
python -m src.models.baseline    # train TF-IDF + XGBoost, save the best
python -m src.models.calibrate   # fit the trust-score calibration
python -m src.rag.ingest_vectors # seed ChromaDB

uvicorn src.api.main:app --port 8000          # terminal 1
streamlit run app/streamlit_app.py            # terminal 2
```

Open <http://localhost:8501> for the UI, <http://localhost:8000/docs> for the API.

Full instructions, including Docker and troubleshooting, live in
[SETUP.md](SETUP.md).

---

## Using real datasets

The pipeline runs on a 200-row synthetic sample when no raw data is present, and
prints a loud warning saying so. For meaningful metrics, drop real data into
`data/raw/` and re-run `python -m src.data.split`:

| Dataset | Files to place in `data/raw/`      |
| ------- | ---------------------------------- |
| ISOT    | `True.csv`, `Fake.csv`             |
| LIAR    | `train.tsv`, `valid.tsv`, `test.tsv` |

Then optionally fine-tune the transformer (CPU-safe, batch drops to 8
automatically):

```bash
python -m src.models.train_transformer
```

If training is skipped or fails, `predict()` transparently keeps using the
baseline behind the identical interface.

---

## API examples

```bash
# Health
curl http://localhost:8000/health

# Classify
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Doctors are SHOCKED by this one weird trick!!!"}'
# {"verdict":"Fake","band":"Fake","trust_score":0.32,
#  "probability_real":0.0041,"model":"tfidf_logreg","degraded":false}

# Explain (negative weight = pushes toward Fake)
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"text":"Doctors are SHOCKED by this one weird trick!!!","top_k":5}'

# Full agentic investigation
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"text":"Scientists confirm drinking bleach cures every known virus."}'

# Feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"text":"...","predicted_verdict":"Fake","user_verdict":"Suspicious","trust_score":21.5}'
```

`/investigate` response shape:

```jsonc
{
  "verdict": "Fake",
  "band": "Fake",
  "trust_score": 0.0,
  "claims": [
    { "claim": "...", "status": "refuted", "reason": "...", "evidence_ids": [1, 2] }
  ],
  "evidence":  [ { "text": "...", "publisher": "Snopes", "url": "...", "score": 0.83 } ],
  "citations": [ { "id": 1, "publisher": "Snopes", "url": "...", "rating": "False" } ],
  "explanation": "Three sentences with [1] style citations.",
  "tokens": [ { "word": "SHOCKED", "weight": -0.31 } ],
  "degraded": false,
  "notes": []
}
```

---

## Configuration

All configuration is environment-driven — see `.env.example`. Nothing is
hardcoded and no key is required to run.

| Variable                   | Default                  | Effect when unset                                  |
| -------------------------- | ------------------------ | -------------------------------------------------- |
| `GEMINI_API_KEY`           | *(empty)*                | Agent uses deterministic heuristics instead of an LLM |
| `GOOGLE_FACTCHECK_API_KEY` | *(empty)*                | Corpus seeds from 25 built-in verified fact-checks  |
| `AGENT_USE_MCP`            | `0`                      | Tools called in-process instead of over MCP stdio   |
| `AGENT_TIMEOUT_SECONDS`    | `30`                     | Hard wall-clock cap on `/investigate`               |
| `AGENT_MAX_TOOL_CALLS`     | `6`                      | Hard cap on tool invocations per request            |
| `EXPLAINER_BACKEND`        | *(auto)*                 | Tries SHAP, then LIME, then occlusion               |
| `FORCE_BASELINE`           | `0`                      | Set to `1` to skip the transformer entirely         |

---

## MCP servers

Four standalone FastMCP servers speak stdio and can be attached to any MCP
client (Claude Desktop, opencode, etc.):

| Server              | Tool                             | Backing |
| ------------------- | -------------------------------- | ------- |
| `classifier-server` | `classify_news(text)`            | calibrated model |
| `evidence-server`   | `search_evidence(claim, k)`      | ChromaDB RAG |
| `factcheck-server`  | `search_fact_checks(query, k)`   | Google Fact Check API |
| `explainer-server`  | `explain_prediction(text, top_k)`| SHAP / LIME |

```bash
python -m src.mcp_servers.evidence_server
```

Set `AGENT_USE_MCP=1` to make the agent route its tool calls through them.

---

## Testing

```bash
pytest -q          # 52 tests, fully offline
ruff check .
```

`tests/conftest.py` pins the environment to offline defaults, so the suite never
touches the network, an LLM, or a GPU.

---

## Screenshots

| View | File |
| ---- | ---- |
| Verdict card, trust gauge and SHAP-highlighted article | `docs/screenshot-verdict.png` |
| Per-claim evidence expanders with clickable citations  | `docs/screenshot-evidence.png` |

To regenerate: start the API and UI, open <http://localhost:8501>, click
**Fake example** then **Full investigation**, and capture the page.

---

## Project layout

```
veritruth/
├── data/  models/  vectordb/  notebooks/
├── src/
│   ├── data/         ingest.py  preprocess.py  split.py
│   ├── models/       baseline.py  train_transformer.py  calibrate.py  predict.py
│   ├── explain/      shap_explainer.py
│   ├── rag/          build_corpus.py  ingest_vectors.py  retriever.py
│   ├── mcp_servers/  classifier_ · evidence_ · factcheck_ · explainer_server.py
│   ├── agent/        prompts.py  graph.py
│   └── api/          main.py  schemas.py
├── app/              streamlit_app.py
├── tests/            test_model.py  test_api.py  test_agent.py
├── docker/           Dockerfile  docker-compose.yml
└── .github/workflows/ci.yml
```

Every module is runnable on its own and prints a verification block:

```bash
python -m src.models.predict "some headline"
python -m src.explain.shap_explainer "some headline"
python -m src.rag.retriever "vaccines cause autism"
python -m src.agent.graph "some article"
```

---

## Command reference

Every command in the project, in one place. Run all of them from the repo root
(`veritruth/`) with the virtualenv activated.

### Setup

```bash
python -m venv .venv                     # create the virtualenv
.venv\Scripts\activate                   # activate  (Windows)
source .venv/bin/activate                # activate  (macOS / Linux)
pip install -r requirements.txt          # install all pinned dependencies
copy .env.example .env                   # create config  (cp on macOS/Linux)
```

### Build the pipeline — run in this order

```bash
python -m src.data.split                 # -> data/processed/{train,val,test}.csv
python -m src.models.baseline            # -> models/baseline.joblib
python -m src.models.calibrate           # -> models/threshold.json
python -m src.rag.ingest_vectors         # -> vectordb/chroma.sqlite3
python -m src.models.train_transformer   # optional: fine-tune DistilBERT
```

### Run the app

```bash
uvicorn src.api.main:app --port 8000              # API      -> :8000/docs
uvicorn src.api.main:app --port 8000 --reload     # API, dev autoreload
streamlit run app/streamlit_app.py                # UI       -> :8501
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build   # build and start both
docker compose -f docker/docker-compose.yml logs -f api  # tail API logs
docker compose -f docker/docker-compose.yml down         # stop
docker compose -f docker/docker-compose.yml down -v      # stop and wipe volumes
```

### Individual modules — each prints its own verification block

```bash
python -m src.data.ingest                     # dataset discovery report
python -m src.data.preprocess                 # cleaning demo
python -m src.models.predict "some headline"  # verdict + trust score
python -m src.explain.shap_explainer "text"   # top token weights
python -m src.rag.build_corpus                # rebuild the fact-check corpus
python -m src.rag.retriever "vaccines cause autism"   # top-k evidence
python -m src.agent.graph "some article"      # full agent investigation
```

### MCP servers (stdio — they block, waiting for a client)

```bash
python -m src.mcp_servers.classifier_server
python -m src.mcp_servers.evidence_server
python -m src.mcp_servers.factcheck_server
python -m src.mcp_servers.explainer_server
python scripts/smoke_mcp.py               # spawn all four and call each tool
```

### Tests and lint

```bash
pytest -q                                 # 52 tests, fully offline
pytest -q tests/test_agent.py             # one file
pytest -q -k investigate                  # match by name
pytest -q -v                              # verbose, per-test names
ruff check .                              # lint
ruff check . --fix                        # lint and autofix
```

### API smoke tests

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Doctors are SHOCKED by this one weird trick!!!"}'

curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"text":"Doctors are SHOCKED by this one weird trick!!!","top_k":5}'

curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"text":"Scientists confirm drinking bleach cures every known virus."}'
```

On Windows PowerShell `curl` is an alias for `Invoke-WebRequest`; use
`curl.exe` explicitly, or:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType 'application/json' `
  -Body '{"text":"Doctors are SHOCKED by this one weird trick!!!"}'
```

### Housekeeping

```bash
Remove-Item -Recurse -Force data/processed, models, vectordb   # Windows: full reset
rm -rf data/processed models vectordb                          # macOS / Linux
Get-Process python | Stop-Process -Force                       # kill stuck servers
```

After a reset, re-run the four pipeline commands above.
