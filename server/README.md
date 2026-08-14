# Server

This folder contains the Flask server, LLM integration, and CLI for running news search + classification.

Quick run (from `server`):

```bash
py -3 -m pip install -r requirements.txt
py -3 cli.py --prompt "latest technology news" --page-size 3
```
Server README — Flask + LLM agent + News fetcher

Quick setup (Windows PowerShell):

```powershell
cd server
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the `server` folder (or set environment variables) using `.env.example` and fill in `NEWSAPI_KEY` and, for local Hugging Face use, set `LOCAL_HF_MODEL` (default `gpt2`). For Google Gemini use set `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_API_KEY`.

Run the server:

```powershell
.\venv\Scripts\Activate.ps1
python app.py
```

API endpoints:
- `GET /health` — basic health check
- `POST /search` — body JSON { "query": "..." } returns found articles and `prediction` fields

Notes:
 - This scaffold can use a local Hugging Face model (`LLM_PROVIDER=hf`) or Google Gemini. By default the repo now prefers a local Hugging Face model for zero-cloud testing. If you want a local ML classifier, set `MODEL_MODE=local` and train a model with `ml_model.train_local_model(csv_path)` and save to `server/models/local_model.joblib`.

Local Hugging Face quick start:

1. Create and activate a Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
2. Install dependencies (note: large models may require `torch` with CUDA):
```powershell
pip install -r requirements.txt
```
3. Set a small model for testing in `.env` or environment (example):
```powershell
setx LOCAL_HF_MODEL gpt2
```
4. Run the server:
```powershell
py -3 app.py
```

If you want to use Google Gemini instead, set `LLM_PROVIDER=gemini` and configure `GOOGLE_APPLICATION_CREDENTIALS` to point to a service-account JSON, then run `py -3 gemini_rest_probe.py` to verify access.
