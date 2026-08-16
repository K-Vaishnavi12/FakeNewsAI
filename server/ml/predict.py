"""
Prediction Service for Fake News Detection
Loads the augmented multi-scale TF-IDF + Logistic Regression model from models/fake_news_model.joblib
Combines statistical n-grams with stylometric linguistic analysis for high accuracy on both short claims and full articles.
"""

import os
import sys
import re
import joblib
import numpy as np


class NewsClassifier:
    _instance = None

    def __init__(self, model_path: str = None):
        self.model_path = model_path or self._find_model_path()
        self.pipeline = None
        self.metadata = {}
        self.is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls, model_path: str = None):
        """Singleton accessor to ensure model is loaded into memory once on backend start."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def _find_model_path(self) -> str:
        candidates = [
            os.path.join(os.path.dirname(__file__), 'models', 'fake_news_model.joblib'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_model.joblib'),
            os.path.join(os.getcwd(), 'ml', 'models', 'fake_news_model.joblib'),
            os.path.join(os.getcwd(), 'server', 'ml', 'models', 'fake_news_model.joblib'),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return os.path.join(os.path.dirname(__file__), 'models', 'fake_news_model.joblib')

    def reload_model(self):
        """Force reload model from disk."""
        self._load_model()

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            print(f"[NewsClassifier] Model not found at '{self.model_path}'. Run train_model.py first.")
            self.is_loaded = False
            return

        try:
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and 'pipeline' in bundle:
                self.pipeline = bundle['pipeline']
                self.metadata = {k: v for k, v in bundle.items() if k != 'pipeline'}
            elif isinstance(bundle, tuple) and len(bundle) == 2:
                clf, vectorizer = bundle
                from sklearn.pipeline import Pipeline
                self.pipeline = Pipeline([('tfidf', vectorizer), ('clf', clf)])
            else:
                self.pipeline = bundle

            self.is_loaded = True
            acc = self.metadata.get('accuracy', 0.0)
            print(f"[NewsClassifier] Loaded fake news model successfully. (Test Accuracy: {acc*100:.2f}%)")
        except Exception as e:
            print(f"[NewsClassifier] Failed to load model: {e}")
            self.is_loaded = False

    def _compute_stylometric_boost(self, text: str) -> float:
        """Compute a stylometric adjustment score between -0.25 (fake) and +0.25 (real):
        - Positive signals: formal reporting verbs, institutional attribution, quotes, neutral dates
        - Negative signals: ALL-CAPS words, exclamation marks, sensationalist clickbait triggers
        """
        t_lower = text.lower()
        score_adj = 0.0

        # Journalistic attribution verbs
        journalistic_cues = [
            'confirmed', 'announced', 'reported', 'stated', 'ruled', 'published',
            'spokesman', 'spokesperson', 'officials', 'authorities', 'agency',
            'department', 'court order', 'investigators', 'according to', 'researchers',
            'scientists', 'study published', 'press release', 'white house', 'pentagon',
            'supreme court', 'reuters', 'associated press', 'bbc', 'nasa'
        ]
        real_matches = sum(1 for c in journalistic_cues if c in t_lower)
        if real_matches > 0:
            score_adj += min(0.20, real_matches * 0.06)

        # Clickbait / Sensationalist triggers
        sensationalist_cues = [
            'shocking', 'leaked', 'secret plot', 'deep state', 'conspiracy', 'cabal',
            'unbelievable', 'you won\'t believe', 'miracle cure', 'they don\'t want you to know',
            'exposed', 'insane', 'disturbing', 'explodes', 'humiliated', 'destroys'
        ]
        fake_matches = sum(1 for c in sensationalist_cues if c in t_lower)
        if fake_matches > 0:
            score_adj -= min(0.22, fake_matches * 0.08)

        # Formatting markers: multiple exclamation marks or all-caps words
        if '!' in text:
            score_adj -= min(0.12, text.count('!') * 0.04)
        if '?' in text and len(text.split()) < 15:
            # Clickbait question headline
            score_adj -= 0.05

        caps_words = [w for w in text.split() if w.isupper() and len(w) > 2 and w not in ('USA', 'NASA', 'UN', 'FBI', 'CIA', 'EU', 'WHO', 'UK', 'US')]
        if len(caps_words) >= 2:
            score_adj -= 0.10

        return score_adj

    def predict(self, text: str) -> dict:
        """Run inference on the given news text or claim.
        Returns:
            label: 'real' or 'fake'
            fake_probability: float (0.0 to 1.0)
            real_probability: float (0.0 to 1.0)
            confidence: float (0.0 to 1.0)
            score: max probability
            top_signals: list of extracted keyword signals with weights
        """
        if not text or not text.strip():
            return {
                'label': 'unknown',
                'fake_probability': 0.0,
                'real_probability': 0.0,
                'confidence': 0.0,
                'score': 0.0,
                'top_signals': [],
                'is_loaded': self.is_loaded,
            }

        if not self.is_loaded or self.pipeline is None:
            return self._heuristic_fallback(text)

        clean = re.sub(r'http\S+|www\.\S+', '', text).strip()
        probs = self.pipeline.predict_proba([clean])[0]
        base_fake_prob = float(probs[0])
        base_real_prob = float(probs[1])

        # Apply stylometric adjustments for short headlines
        sty_adj = self._compute_stylometric_boost(clean)
        adjusted_real_prob = np.clip(base_real_prob + sty_adj, 0.02, 0.98)
        adjusted_fake_prob = 1.0 - adjusted_real_prob

        label = 'real' if adjusted_real_prob >= 0.50 else 'fake'
        confidence = float(max(adjusted_fake_prob, adjusted_real_prob))

        top_signals = self._extract_signals(clean)

        return {
            'label': label,
            'fake_probability': round(float(adjusted_fake_prob), 4),
            'real_probability': round(float(adjusted_real_prob), 4),
            'confidence': round(float(confidence), 4),
            'score': round(float(confidence), 4),
            'top_signals': top_signals,
            'is_loaded': True,
        }

    def _extract_signals(self, text: str, max_signals: int = 6) -> list:
        """Extract top feature tokens present in the input that pushed prediction towards Real or Fake."""
        try:
            tfidf = self.pipeline.named_steps['tfidf']
            clf = self.pipeline.named_steps['clf']
            feature_names = np.array(tfidf.get_feature_names_out())
            coefs = clf.coef_[0]

            X_vec = tfidf.transform([text])
            nonzero_indices = X_vec.nonzero()[1]

            signals = []
            for idx in nonzero_indices:
                feature_name = feature_names[idx]
                tf_idf_val = X_vec[0, idx]
                weight = coefs[idx] * tf_idf_val
                signals.append({
                    'word': feature_name,
                    'impact': 'real' if weight > 0 else 'fake',
                    'weight': round(float(weight), 4)
                })

            signals.sort(key=lambda x: abs(x['weight']), reverse=True)
            return signals[:max_signals]
        except Exception:
            return []

    def _heuristic_fallback(self, text: str) -> dict:
        t = (text or "").lower()
        real_indicators = ['reuters', 'associated press', 'apnews', 'bbc', 'cnn', 'nytimes', 'washington post', 'bloomberg', 'official', 'statement', 'announced']
        fake_indicators = ['shocking', 'conspiracy', 'illuminati', 'hoax', 'secret plot', 'deepfake', 'miracle cure', 'unbelievable', 'fake', 'debunked']

        real_count = sum(1 for w in real_indicators if w in t)
        fake_count = sum(1 for w in fake_indicators if w in t)

        if real_count > fake_count:
            return {'label': 'real', 'fake_probability': 0.20, 'real_probability': 0.80, 'confidence': 0.80, 'score': 0.80, 'top_signals': [], 'is_loaded': False}
        elif fake_count > real_count:
            return {'label': 'fake', 'fake_probability': 0.85, 'real_probability': 0.15, 'confidence': 0.85, 'score': 0.85, 'top_signals': [], 'is_loaded': False}
        else:
            return {'label': 'unknown', 'fake_probability': 0.50, 'real_probability': 0.50, 'confidence': 0.50, 'score': 0.50, 'top_signals': [], 'is_loaded': False}


def predict_news(text: str) -> dict:
    return NewsClassifier.get_instance().predict(text)


def get_classifier() -> NewsClassifier:
    return NewsClassifier.get_instance()
