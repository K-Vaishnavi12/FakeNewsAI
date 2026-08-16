"""
ML Model Integration Layer
Interfaces with the trained TF-IDF + Logistic Regression model in server/ml/predict.py
Provides classify() and classify_with_probabilities() for backend endpoints and AI agent.
"""

import os
import sys

try:
    from .ml.predict import predict_news, get_classifier
    from .ml.train_model import train as train_model_pipeline
except ImportError:
    try:
        from ml.predict import predict_news, get_classifier
        from ml.train_model import train as train_model_pipeline
    except ImportError:
        # Fallback if path manipulation needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from ml.predict import predict_news, get_classifier
        from ml.train_model import train as train_model_pipeline


# Ensure singleton is initialized on import
_classifier = get_classifier()


def classify(text: str):
    """Return tuple (label, score) for backward compatibility with existing routes.
    label: 'real' | 'fake' | 'unknown'
    score: confidence score float (0.0 - 1.0)
    """
    if not text or not text.strip():
        return "unknown", 0.0

    res = predict_news(text)
    return res.get('label', 'unknown'), res.get('score', 0.0)


def classify_with_probabilities(text: str) -> dict:
    """Return comprehensive classification metadata including probabilities and top keyword signals."""
    if not text or not text.strip():
        return {
            'label': 'unknown',
            'fake_probability': 0.0,
            'real_probability': 0.0,
            'confidence': 0.0,
            'score': 0.0,
            'top_signals': [],
            'is_loaded': _classifier.is_loaded,
        }

    return predict_news(text)


def train_local_model(csv_path: str = None, model_out: str = None):
    """Trigger training pipeline."""
    return train_model_pipeline(news_csv_path=csv_path, output_path=model_out)


def is_model_loaded() -> bool:
    """Check if ML model is ready in memory."""
    return _classifier.is_loaded
