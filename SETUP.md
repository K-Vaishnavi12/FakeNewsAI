# SETUP.md — VeriTruth

Exact commands from clone to a running demo, plus the failures you are most
likely to hit and how to fix them.

---

## 1. Prerequisites

* Python **3.11** (`python --version` must print 3.11.x)
* ~4 GB free disk (PyTorch + MiniLM embeddings)
* Optional: Docker Desktop 4.x for the container path
* Optional: `GEMINI_API_KEY`, `GOOGLE_FACTCHECK_API_KEY` — the system runs
  without both

---

## 2. Local setup (Windows PowerShell)

```powershell
git clone <your-repo-url>
cd veritruth

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env      # edit if you have API keys
```

macOS / Linux:

```bash
git clone <your-repo-url>
cd veritruth
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

---

## 3. Build the pipeline (once, ~2 minutes)

Run in order. Each command prints its own verification block.

```bash
python -m src.data.split          # -> data/processed/{train,val,test}.csv
python -m src.models.baseline     # -> models/baseline.joblib
python -m src.models.calibrate    # -> models/threshold.json
python -m src.rag.ingest_vectors  # -> vectordb/chroma.sqlite3
```

Expected output markers:

| Command | Confirms success when you see |
| ------- | ----------------------------- |
| `src.data.split` | `train: N rows \| label ratio {...}` and three CSVs written |
| `src.models.baseline` | metrics for `tfidf_logreg` **and** `tfidf_xgboost`, best model saved |
| `src.models.calibrate` | `Brier before/after` with after ≤ before, `threshold.json` written |
| `src.rag.ingest_vectors` | `Status: OK` and `Collection count: > 0` |

Optional — fine-tune DistilBERT (CPU-safe, 3 epochs, auto batch-size 8):

```bash
python -m src.models.train_transformer
```

Skip it and everything still works: `predict()` keeps the identical interface
and uses the baseline.

---

## 4. Run the demo

Two terminals, both with the venv activated:

```bash
# terminal 1 — API
uvicorn src.api.main:app --port 8000

# terminal 2 — UI
streamlit run app/streamlit_app.py
```

Verify:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"tfidf_logreg","model_loaded":true,"evidence_chunks":25,...}
```

* UI  → <http://localhost:8501>
* Docs → <http://localhost:8000/docs>

---

## 5. Run with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

The API container seeds the vector store on boot, then serves on `:8000`. The UI
waits for the API healthcheck and serves on `:8501`. Models, the vector DB, the
SQLite feedback file and the Hugging Face cache all live in named volumes, so a
restart does not re-download or re-train anything.

```bash
docker compose -f docker/docker-compose.yml down       # stop
docker compose -f docker/docker-compose.yml down -v    # stop and wipe volumes
```

---

## 6. Tests and lint

```bash
pytest -q      # 52 tests, no network required
ruff check .
```

---

## 7. Troubleshooting

| # | Symptom | Cause | Fix |
| - | ------- | ----- | --- |
| 1 | `ModuleNotFoundError: No module named 'src'` | Running from the wrong directory | Run every command from the `veritruth/` repo root, not from `src/` |
| 2 | `NO RAW DATASET FOUND ... Generating a 200-row SYNTHETIC sample` | `data/raw/` is empty | Expected on a fresh clone. For real metrics, add ISOT `True.csv`/`Fake.csv` or LIAR `*.tsv` to `data/raw/` and re-run `python -m src.data.split` |
| 3 | `/health` returns `"status":"degraded"` | Model or vector store not built | Run the four Step-3 commands in order; check `model_loaded` and `evidence_chunks` in the response to see which one is missing |
| 4 | `TypeError: issubclass() arg 1 must be a class` from an MCP server | `from __future__ import annotations` stringifies annotations, which the MCP SDK cannot introspect | Never add that import to files containing `@mcp.tool()` functions |
| 5 | `/investigate` returns `"degraded": true` with `notes: ["fallback:timeout"]` | Agent exceeded its wall-clock budget | Raise `AGENT_TIMEOUT_SECONDS` in `.env`, or shorten the input text |
| 6 | Streamlit shows *"Cannot reach the VeriTruth API"* | Backend not running, or wrong URL | Start uvicorn, then confirm `VERITRUTH_API_URL` matches its address (`http://api:8000` inside Docker, `http://localhost:8000` locally) |
| 7 | `torch` install is huge or fails on CPU-only machines | pip defaults to CUDA wheels | `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1` before `pip install -r requirements.txt` |
| 8 | `chromadb.telemetry ... capture() takes 1 positional argument` | Harmless telemetry bug in chromadb 0.5.5 | Ignore it, or set `ANONYMIZED_TELEMETRY=False` in `.env` |
| 9 | First `/explain` or `/investigate` call takes 30–60 s | SHAP permutation explainer plus a cold MiniLM download | Set `EXPLAINER_BACKEND=occlusion` for a fast deterministic explainer; subsequent calls are cached and fast |
| 10 | `ValueError: not enough values to unpack` during `src.models.baseline` | `data/processed/` is stale or partially written | Delete `data/processed/` and re-run `python -m src.data.split` |
| 11 | Port already in use on 8000/8501 | Previous run still alive | `uvicorn ... --port 8010`, or kill the old process (`Get-Process python \| Stop-Process` on Windows) |
| 12 | Agent never cites anything, `citations: []` | Vector store empty | `python -m src.rag.ingest_vectors` and confirm `Collection count > 0` |

---

## 8. Sixty-second demo script

**0:00–0:08 — Setup.**
Two browser tabs already open: the Streamlit UI on `:8501`, the FastAPI docs on
`:8000/docs`.

> "This is VeriTruth. It doesn't just guess whether news is fake — it shows you
> its reasoning and cites its sources."

**0:08–0:15 — Point at the sidebar.**
Point to `status: ok`, `Model: tfidf_logreg`, `Evidence chunks: 25`.

> "The backend is live, the classifier is loaded, and there are twenty-five
> fact-checks indexed in the vector store."

**0:15–0:22 — Click "Fake example", then click "Full investigation".**

> "I'll paste a classic piece of health disinformation and run a full
> investigation."

**0:22–0:33 — The verdict card renders. Point at it.**

> "Verdict: likely fake. Trust score zero out of a hundred — and that's a
> *calibrated* score, not a raw softmax number, so it actually means something."

**0:33–0:42 — Scroll to "What the model reacted to".**
Hover one red token.

> "Here's the explainability layer. Every red word pushed the model toward
> 'fake' — 'SHOCKED', 'BREAKING', 'SHARE'. Hover any word and you get its exact
> SHAP weight."

**0:42–0:54 — Scroll to "Claims and evidence". Expand Claim 1.**

> "And this is the agent. It broke the article into three atomic claims, and for
> each one it retrieved fact-check evidence. Claim one: refuted — and here are
> the actual sources, Snopes and the WHO, with live links. Nothing is
> hallucinated; every citation is retrieved."

**0:54–1:00 — Point at the sidebar's "Gemini configured: False".**

> "And notice: no LLM key configured, no internet needed. It degraded to
> deterministic heuristics and still delivered a fully cited verdict. That's the
> whole point — it never silently fails."
