# VeriTruth — multi-stage image (Python 3.11, CPU-only, non-root).
#
# Build from the repository root so that `src/`, `app/` and the data folders are
# in the build context:
#
#     docker build -f docker/Dockerfile -t veritruth:latest .

# ----------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheels.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
 && /opt/venv/bin/pip install -r requirements.txt

# ----------------------------------------------------------------- runtime
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/veritruth/.cache/huggingface \
    VERITRUTH_API_URL=http://api:8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 veritruth

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY --chown=veritruth:veritruth src/ ./src/
COPY --chown=veritruth:veritruth app/ ./app/
COPY --chown=veritruth:veritruth tests/ ./tests/
COPY --chown=veritruth:veritruth requirements.txt pyproject.toml ./

RUN mkdir -p /app/data /app/models /app/vectordb \
 && chown -R veritruth:veritruth /app

USER veritruth

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
