"""Step 4c — the single inference seam for the whole project.

``predict(text)`` returns the same schema whether it is backed by the
fine-tuned DistilBERT or the TF-IDF baseline, so the API, the MCP servers and
the agent never need to know which model is live.

Schema::

    {
      "verdict": "Real" | "Suspicious" | "Fake",
      "band": same as verdict,
      "trust_score": float 0-100,
      "probability_real": float 0-1,
      "model": "distilbert" | "tfidf_logreg" | ...,
      "degraded": bool
    }

Run::

    python -m src.models.predict "Some headline to score"
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config import (
    CPU_BATCH_SIZE,
    MAX_TEXT_CHARS,
    TRANSFORMER_DIR,
    TRANSFORMER_MAX_LEN,
    force_baseline,
    get_logger,
)
from src.models.calibrate import Calibration

LOG = get_logger("veritruth.models.predict")

_LOCK = threading.Lock()
_PREDICTOR: "Predictor | None" = None


def _truncate(text: str) -> str:
    text = (text or "").strip()
    return text[:MAX_TEXT_CHARS]


class Predictor:
    """Loads the best available model once and scores text in batches."""

    def __init__(self, model_dir: Path = TRANSFORMER_DIR) -> None:
        self.calibration = Calibration.load()
        self.kind = "baseline"
        self.name = "tfidf_baseline"
        self.degraded = False
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._baseline = None

        if not force_baseline():
            self._try_load_transformer(Path(model_dir))
        if self.kind != "transformer":
            self._load_baseline()

    # ------------------------------------------------------------- loading
    def _try_load_transformer(self, model_dir: Path) -> None:
        if not (model_dir / "config.json").exists():
            LOG.info("No fine-tuned transformer at %s; using baseline.", model_dir)
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            self._model.eval()
            self._torch = torch
            self.kind = "transformer"
            self.name = "distilbert"
            LOG.info("Loaded fine-tuned transformer from %s", model_dir)
        except Exception as exc:
            LOG.warning("Transformer load failed (%s); falling back to baseline.", exc)
            self._tokenizer = self._model = self._torch = None
            self.kind = "baseline"

    def _load_baseline(self) -> None:
        try:
            from src.models.baseline import load_baseline

            self._baseline = load_baseline()
            self.name = getattr(self._baseline, "name", "tfidf_baseline")
            LOG.info("Loaded baseline model '%s'.", self.name)
        except Exception as exc:
            LOG.error("Baseline unavailable (%s); using neutral heuristic.", exc)
            self._baseline = None
            self.name = "heuristic"
            self.degraded = True

    # ------------------------------------------------------------ scoring
    def _proba_transformer(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), CPU_BATCH_SIZE):
                chunk = texts[start : start + CPU_BATCH_SIZE]
                enc = self._tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    max_length=TRANSFORMER_MAX_LEN,
                    return_tensors="pt",
                )
                logits = self._model(**enc).logits.detach().cpu().numpy()
                margin = logits[:, 1] - logits[:, 0]
                out.extend(np.asarray(self.calibration.apply_logit(margin)).ravel().tolist())
        return np.asarray(out, dtype=float)

    def _proba_baseline(self, texts: list[str]) -> np.ndarray:
        if self._baseline is None:
            return np.full(len(texts), 0.5, dtype=float)
        raw = np.asarray(self._baseline.predict_proba(texts), dtype=float).ravel()
        return np.asarray(self.calibration.apply_proba(raw), dtype=float).ravel()

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Calibrated P(REAL) for each text. Never raises."""
        items = [_truncate(str(t)) for t in texts]
        if not items:
            return np.zeros(0, dtype=float)
        try:
            if self.kind == "transformer":
                return self._proba_transformer(items)
            return self._proba_baseline(items)
        except Exception as exc:
            LOG.error("Scoring failed (%s); returning neutral 0.5.", exc)
            return np.full(len(items), 0.5, dtype=float)

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        probas = self.predict_proba(texts)
        results: list[dict[str, Any]] = []
        for raw_text, p in zip(texts, probas.tolist()):
            p = float(np.clip(p, 0.0, 1.0))
            score = self.calibration.trust_score(p)
            band = self.calibration.band(score)
            results.append(
                {
                    "verdict": band,
                    "band": band,
                    "trust_score": score,
                    "probability_real": round(p, 6),
                    "model": self.name,
                    "degraded": self.degraded or not str(raw_text).strip(),
                }
            )
        return results


def get_predictor() -> Predictor:
    """Process-wide singleton, built once (thread-safe double-checked lock)."""
    global _PREDICTOR
    if _PREDICTOR is None:
        with _LOCK:
            if _PREDICTOR is None:
                _PREDICTOR = Predictor()
    return _PREDICTOR


def predict(text: str) -> dict[str, Any]:
    """Public one-shot prediction used by the API, MCP servers and agent."""
    return get_predictor().predict(text)


def predict_proba(texts: Iterable[str]) -> np.ndarray:
    """Vectorised P(REAL); required by SHAP/LIME explainers."""
    return get_predictor().predict_proba(texts)


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or (
        "BREAKING!!! Doctors are SHOCKED by this one weird trick the government "
        "does not want you to know about."
    )
    result = predict(text)
    print("\n--- STEP 4c VERIFICATION (predict) ----------------------------")
    print(f"Input   : {text[:80]}{'...' if len(text) > 80 else ''}")
    for key in ("verdict", "band", "trust_score", "probability_real", "model", "degraded"):
        print(f"  {key:<18} {result[key]}")
    assert 0.0 <= result["trust_score"] <= 100.0, "trust_score out of range"
    assert result["band"] in {"Real", "Suspicious", "Fake"}, "invalid band"
    print("Expected: trust_score in [0,100], band in {Real,Suspicious,Fake}")
    print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
