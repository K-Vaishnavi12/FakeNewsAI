"""Ensemble prediction module.

Loads the persisted pipelines lazily and combines them by soft voting. If the
artifacts are missing the predictor degrades to `available=False` and the rest of
the application continues to work (the verdict then rests on evidence alone).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings
from app.ml.attribution import token_attributions
from app.schemas import MLResult, ModelVote, TokenAttribution

# Human-friendly display names.
DISPLAY_NAMES = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "multinomial_nb": "TF-IDF + Multinomial Naive Bayes",
    "random_forest_style_features": "Random Forest (engineered features)",
    "xgboost_style_features": "XGBoost (engineered features)",
}

_LOCK = threading.Lock()


class EnsemblePredictor:
    """Soft-voting ensemble over the persisted pipelines."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or get_settings().model_dir
        self._models: dict[str, object] = {}
        self._metadata: dict = {}
        self._loaded = False

    # --- loading ----------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        with _LOCK:
            if self._loaded:
                return
            if self.model_dir.exists():
                for path in sorted(self.model_dir.glob("*.joblib")):
                    try:
                        self._models[path.stem] = joblib.load(path)
                    except Exception:  # noqa: BLE001 - a bad artifact must not crash the app
                        continue
                meta_path = self.model_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        self._metadata = json.loads(
                            meta_path.read_text(encoding="utf-8")
                        )
                    except (json.JSONDecodeError, OSError):
                        self._metadata = {}
            self._loaded = True

    @property
    def available(self) -> bool:
        self.load()
        return bool(self._models)

    @property
    def metadata(self) -> dict:
        self.load()
        return self._metadata

    # --- prediction -------------------------------------------------------

    @staticmethod
    def _probability_real(model, text: str) -> float | None:
        """Return P(REAL) in [0,1], or None if the model cannot be scored."""
        X = np.asarray([text], dtype=object)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                if 1 in classes:
                    return float(proba[classes.index(1)])
                return float(proba[-1])
            if hasattr(model, "decision_function"):
                score = float(np.ravel(model.decision_function(X))[0])
                return float(1.0 / (1.0 + np.exp(-score)))
            pred = int(np.ravel(model.predict(X))[0])
            return 1.0 if pred == 1 else 0.0
        except Exception:  # noqa: BLE001 - one broken model must not break the vote
            return None

    def predict(self, text: str) -> MLResult:
        self.load()

        if not text or not text.strip():
            return MLResult(
                prediction="UNKNOWN",
                confidence=0,
                available=self.available,
                note="No text was supplied, so no classification was attempted.",
            )

        if not self._models:
            return MLResult(
                prediction="UNKNOWN",
                confidence=0,
                available=False,
                models_agree=True,
                note=(
                    "ML artifacts were not found. Run `python -m app.ml.train` to "
                    "train the ensemble. The verdict is based on retrieved "
                    "evidence alone."
                ),
            )

        votes: list[ModelVote] = []
        probabilities: list[float] = []

        for name, model in self._models.items():
            prob_real = self._probability_real(model, text)
            if prob_real is None:
                continue
            probabilities.append(prob_real)
            prediction = "REAL" if prob_real >= 0.5 else "FAKE"
            confidence = int(round(max(prob_real, 1.0 - prob_real) * 100))
            votes.append(
                ModelVote(
                    model_name=DISPLAY_NAMES.get(name, name),
                    prediction=prediction,
                    confidence=confidence,
                )
            )

        if not probabilities:
            return MLResult(
                prediction="UNKNOWN",
                confidence=0,
                available=False,
                note="All ensemble members failed to score this input.",
            )

        mean_prob = float(np.mean(probabilities))
        prediction = "REAL" if mean_prob >= 0.5 else "FAKE"
        confidence = int(round(max(mean_prob, 1.0 - mean_prob) * 100))

        distinct = {v.prediction for v in votes}
        models_agree = len(distinct) == 1

        # Real attribution from the linear model, used by the "Model Attribution
        # & Keywords" panel. Empty list if unavailable - never placeholder data.
        attributions: list[TokenAttribution] = []
        linear_model = self._models.get("logistic_regression")
        if linear_model is not None:
            try:
                attributions = [
                    TokenAttribution(**item)
                    for item in token_attributions(linear_model, text)
                ]
            except Exception:  # noqa: BLE001 - attribution is best-effort only
                attributions = []

        note = None
        if not models_agree:
            # Disagreement is a real uncertainty signal - surface it and damp
            # the confidence rather than hiding it behind the average.
            agreeing = sum(1 for v in votes if v.prediction == prediction)
            note = (
                f"The ensemble members disagree ({agreeing} of {len(votes)} models "
                f"predict {prediction}). Confidence has been reduced accordingly."
            )
            confidence = min(confidence, 55)

        return MLResult(
            model_name="VeritasCheck Ensemble",
            prediction=prediction,  # type: ignore[arg-type]
            confidence=confidence,
            votes=votes,
            models_agree=models_agree,
            available=True,
            note=note,
            token_attributions=attributions,
        )


_predictor: EnsemblePredictor | None = None


def get_predictor() -> EnsemblePredictor:
    """Process-wide singleton so artifacts load only once."""
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor


def reset_predictor() -> None:
    """Used by tests."""
    global _predictor
    _predictor = None
