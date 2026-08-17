"""Adapter between the Flask/CLI layer and the ML inference package.

Keeps route handlers free of sklearn specifics and provides the small,
stable surface (``classify``, ``classify_with_probabilities``,
``get_model_info``) that the rest of the app depends on.
"""

from typing import Optional, Tuple

from .logging_config import get_logger
from .ml.predict import get_classifier, predict_news
from .ml.train_model import train as train_model_pipeline

logger = get_logger(__name__)

# Loading the joblib bundle is expensive, so warm the singleton at import time.
_classifier = get_classifier()

_EMPTY_RESULT = {
    "label": "unknown",
    "fake_probability": 0.0,
    "real_probability": 0.0,
    "confidence": 0.0,
    "score": 0.0,
    "top_signals": [],
}


def classify(text: str) -> Tuple[str, float]:
    """Classify ``text`` and return ``(label, confidence)``.

    Retained for backward compatibility with the older route/CLI signature.
    """
    if not text or not text.strip():
        return "unknown", 0.0
    result = predict_news(text)
    return result.get("label", "unknown"), result.get("score", 0.0)


def classify_with_probabilities(text: str) -> dict:
    """Classify ``text`` and return full metadata (probabilities + signals)."""
    if not text or not text.strip():
        return {**_EMPTY_RESULT, "is_loaded": _classifier.is_loaded}
    return predict_news(text)


def train_local_model(data_dir: Optional[str] = None,
                      model_out: Optional[str] = None) -> str:
    """Run the training pipeline and reload the in-memory model.

    Args:
        data_dir: Directory containing dataset CSVs.
        model_out: Destination path. When invoked from an HTTP route this MUST
            already have been validated by ``server.paths.resolve_model_output``.

    Returns:
        The path the model bundle was written to.
    """
    saved_path, accuracy, _report, _cm = train_model_pipeline(
        data_dir=data_dir, output_path=model_out
    )
    logger.info("Training finished: %s (accuracy=%.4f)", saved_path, accuracy)
    # Swap the freshly trained model in without restarting the process.
    _classifier.reload_model()
    return saved_path


def is_model_loaded() -> bool:
    """Return True if the trained model is resident in memory."""
    return _classifier.is_loaded


def get_model_info() -> dict:
    """Return the loaded model's metadata (type, accuracy, training date)."""
    return _classifier.model_info()


def get_model_accuracy_display() -> str:
    """Return the model's accuracy as a display string, e.g. ``'98.80%'``."""
    return _classifier.accuracy_display
