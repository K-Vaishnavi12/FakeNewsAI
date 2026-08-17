"""Inference service for the fake-news classifier.

Loads the TF-IDF + Logistic Regression bundle produced by
:mod:`server.ml.train_model` and combines the model's statistical output with
a small stylometric adjustment that helps on very short claims/headlines.
"""

import os
import re
from typing import Dict, List, Optional

import joblib
import numpy as np

from ..constants import (
    ALLOWED_ACRONYMS,
    CLASS_LABELS,
    FALLBACK_FAKE_INDICATORS,
    FALLBACK_REAL_INDICATORS,
    JOURNALISTIC_CUES,
    SENSATIONALIST_CUES,
)
from ..logging_config import get_logger

logger = get_logger(__name__)


class NewsClassifier:
    """Singleton wrapper around the persisted sklearn pipeline."""

    _instance: Optional["NewsClassifier"] = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self._find_model_path()
        self.pipeline = None
        self.metadata: Dict = {}
        self.is_loaded = False
        # Maps a label name ('fake'/'real') to its column index in
        # predict_proba output. Derived from pipeline.classes_, never assumed.
        self._label_to_index: Dict[str, int] = {}
        self._load_model()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "NewsClassifier":
        """Return the process-wide classifier, loading it on first access."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _find_model_path(self) -> str:
        """Locate the model bundle relative to this file."""
        default = os.path.join(os.path.dirname(__file__), "models",
                               "fake_news_model.joblib")
        candidates = [
            default,
            os.path.join(os.path.dirname(__file__), "..", "models",
                         "fake_news_model.joblib"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        return default

    def reload_model(self) -> None:
        """Force a reload from disk (used after retraining)."""
        self._load_model()

    def _load_model(self) -> None:
        """Load the joblib bundle and derive the probability->label mapping."""
        if not os.path.isfile(self.model_path):
            logger.error(
                "Model not found at '%s'. Run `python -m server.ml.train_model` "
                "first; falling back to heuristics.", self.model_path,
            )
            self.is_loaded = False
            return

        try:
            bundle = joblib.load(self.model_path)
        except Exception:
            logger.exception("Failed to load model from '%s'", self.model_path)
            self.is_loaded = False
            return

        if isinstance(bundle, dict) and "pipeline" in bundle:
            self.pipeline = bundle["pipeline"]
            self.metadata = {k: v for k, v in bundle.items() if k != "pipeline"}
        elif isinstance(bundle, tuple) and len(bundle) == 2:
            # Legacy format: (classifier, vectorizer)
            from sklearn.pipeline import Pipeline
            clf, vectorizer = bundle
            self.pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
            self.metadata = {}
        else:
            self.pipeline = bundle
            self.metadata = {}

        try:
            self._label_to_index = self._build_label_index()
        except ValueError:
            logger.exception("Model class labels are inconsistent with metadata")
            self.is_loaded = False
            self.pipeline = None
            return

        self.is_loaded = True
        logger.info(
            "Classifier loaded from '%s' (test accuracy: %s, classes: %s)",
            self.model_path, self.accuracy_display, self._label_to_index,
        )

    def _build_label_index(self) -> Dict[str, int]:
        """Derive {label_name: proba_column_index} from ``pipeline.classes_``.

        The previous implementation hardcoded ``probs[0] == fake`` and
        ``probs[1] == real``. sklearn orders ``predict_proba`` columns by
        ``classes_``, so any change to the training label encoding would have
        silently inverted every prediction. We now read the truth from the
        fitted estimator and cross-check it against the saved metadata.

        Raises:
            ValueError: If the model's classes cannot be mapped to the
                expected ``('fake', 'real')`` labels.
        """
        classes = getattr(self.pipeline, "classes_", None)
        if classes is None:
            # Pipelines proxy classes_ to the final step; try it explicitly.
            final_step = getattr(self.pipeline, "_final_estimator", None)
            classes = getattr(final_step, "classes_", None)
        if classes is None:
            raise ValueError("Fitted estimator exposes no classes_ attribute.")

        classes = list(classes)
        # Saved label names, index-aligned with the integer class values.
        saved = list(self.metadata.get("classes") or CLASS_LABELS)

        mapping: Dict[str, int] = {}
        for column, class_value in enumerate(classes):
            if isinstance(class_value, (int, np.integer)):
                idx = int(class_value)
                if not 0 <= idx < len(saved):
                    raise ValueError(
                        f"Class value {idx} has no entry in saved labels {saved}."
                    )
                mapping[str(saved[idx]).lower()] = column
            else:
                # Model was trained on string labels directly.
                mapping[str(class_value).lower()] = column

        missing = {"fake", "real"} - set(mapping)
        if missing:
            raise ValueError(
                f"Model classes {classes} do not cover required labels {missing}."
            )
        return mapping

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------

    @property
    def accuracy(self) -> Optional[float]:
        """Held-out test accuracy recorded at training time, if available."""
        value = self.metadata.get("accuracy")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def accuracy_display(self) -> str:
        """Accuracy formatted for UI/API, or ``'unknown'`` if not recorded."""
        acc = self.accuracy
        return f"{acc * 100:.2f}%" if acc is not None else "unknown"

    @property
    def model_type(self) -> str:
        """Human-readable description of the estimator."""
        return self.metadata.get(
            "model_type", "Augmented Multi-Scale TF-IDF + Logistic Regression"
        )

    def model_info(self) -> dict:
        """Return a JSON-safe summary of the loaded model for API responses."""
        return {
            "is_loaded": self.is_loaded,
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "accuracy_display": self.accuracy_display,
            "trained_at": self.metadata.get("trained_at"),
            "total_samples": self.metadata.get("total_samples"),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _compute_stylometric_boost(self, text: str) -> float:
        """Return a real-vs-fake adjustment in roughly [-0.25, +0.25].

        Positive: attribution verbs, institutional sourcing.
        Negative: shouting, clickbait punctuation, conspiracy vocabulary.
        """
        t_lower = text.lower()
        score_adj = 0.0

        real_matches = sum(1 for cue in JOURNALISTIC_CUES if cue in t_lower)
        if real_matches:
            score_adj += min(0.20, real_matches * 0.06)

        fake_matches = sum(1 for cue in SENSATIONALIST_CUES if cue in t_lower)
        if fake_matches:
            score_adj -= min(0.22, fake_matches * 0.08)

        if "!" in text:
            score_adj -= min(0.12, text.count("!") * 0.04)
        if "?" in text and len(text.split()) < 15:
            score_adj -= 0.05  # short question headline == clickbait pattern

        caps_words = [
            w for w in text.split()
            if w.isupper() and len(w) > 2 and w not in ALLOWED_ACRONYMS
        ]
        if len(caps_words) >= 2:
            score_adj -= 0.10

        return score_adj

    def predict(self, text: str) -> dict:
        """Classify a claim or article.

        Args:
            text: Raw user-supplied text.

        Returns:
            dict with ``label`` ('real'|'fake'|'unknown'), ``fake_probability``,
            ``real_probability``, ``confidence``, ``score``, ``top_signals``
            and ``is_loaded``.
        """
        if not text or not text.strip():
            return {
                "label": "unknown",
                "fake_probability": 0.0,
                "real_probability": 0.0,
                "confidence": 0.0,
                "score": 0.0,
                "top_signals": [],
                "is_loaded": self.is_loaded,
            }

        if not self.is_loaded or self.pipeline is None:
            return self._heuristic_fallback(text)

        clean = re.sub(r"http\S+|www\.\S+", "", text).strip()

        try:
            probs = self.pipeline.predict_proba([clean])[0]
        except Exception:
            logger.exception("predict_proba failed; using heuristic fallback")
            return self._heuristic_fallback(text)

        # Index by label name, derived from classes_ -- not by position.
        base_fake_prob = float(probs[self._label_to_index["fake"]])
        base_real_prob = float(probs[self._label_to_index["real"]])

        sty_adj = self._compute_stylometric_boost(clean)
        adjusted_real_prob = float(np.clip(base_real_prob + sty_adj, 0.02, 0.98))
        adjusted_fake_prob = 1.0 - adjusted_real_prob

        label = "real" if adjusted_real_prob >= 0.50 else "fake"
        confidence = max(adjusted_fake_prob, adjusted_real_prob)

        return {
            "label": label,
            "fake_probability": round(adjusted_fake_prob, 4),
            "real_probability": round(adjusted_real_prob, 4),
            "confidence": round(confidence, 4),
            "score": round(confidence, 4),
            "top_signals": self._extract_signals(clean),
            "is_loaded": True,
        }

    def _extract_signals(self, text: str, max_signals: int = 6) -> List[dict]:
        """Return the input tokens that most influenced the decision.

        Contribution is ``coefficient * tfidf_value`` per present feature.
        The sign convention follows the positive class of the fitted model.
        """
        try:
            tfidf = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]
        except (AttributeError, KeyError):
            logger.debug("Pipeline has no tfidf/clf steps; no signals available")
            return []

        try:
            feature_names = np.array(tfidf.get_feature_names_out())
            coefs = clf.coef_[0]
            x_vec = tfidf.transform([text])

            # For binary LR, coef_[0] points toward classes_[1]. Determine
            # which label that is instead of assuming positive == 'real'.
            positive_label = "real" if self._label_to_index.get("real", 1) == 1 \
                else "fake"
            negative_label = "fake" if positive_label == "real" else "real"

            signals = []
            for idx in x_vec.nonzero()[1]:
                weight = float(coefs[idx] * x_vec[0, idx])
                signals.append({
                    "word": str(feature_names[idx]),
                    "impact": positive_label if weight > 0 else negative_label,
                    "weight": round(weight, 4),
                })

            signals.sort(key=lambda s: abs(s["weight"]), reverse=True)
            return signals[:max_signals]
        except Exception:
            # Signal attribution is a nice-to-have; never fail the prediction.
            logger.warning("Signal extraction failed", exc_info=True)
            return []

    def _heuristic_fallback(self, text: str) -> dict:
        """Keyword heuristic used only when the trained model is unavailable."""
        t = (text or "").lower()
        real_count = sum(1 for w in FALLBACK_REAL_INDICATORS if w in t)
        fake_count = sum(1 for w in FALLBACK_FAKE_INDICATORS if w in t)

        if real_count > fake_count:
            fake_p, real_p, label = 0.20, 0.80, "real"
        elif fake_count > real_count:
            fake_p, real_p, label = 0.85, 0.15, "fake"
        else:
            fake_p, real_p, label = 0.50, 0.50, "unknown"

        return {
            "label": label,
            "fake_probability": fake_p,
            "real_probability": real_p,
            "confidence": max(fake_p, real_p),
            "score": max(fake_p, real_p),
            "top_signals": [],
            "is_loaded": False,
        }


def predict_news(text: str) -> dict:
    """Convenience wrapper around the singleton classifier."""
    return NewsClassifier.get_instance().predict(text)


def get_classifier() -> NewsClassifier:
    """Return the process-wide :class:`NewsClassifier`."""
    return NewsClassifier.get_instance()
