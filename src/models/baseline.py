"""Step 3 — classical baselines: TF-IDF + LogisticRegression and XGBoost.

Both models are evaluated on the validation split; the better ROC-AUC (with
accuracy as tie-breaker) is persisted to ``models/baseline.joblib`` together with
its fitted vectorizer and metadata.

Run::

    python -m src.models.baseline
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BASELINE_PATH, MODEL_DIR, SEED, ensure_dirs, get_logger, set_seeds

LOG = get_logger("veritruth.models.baseline")

MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)


@dataclass
class BaselineArtifact:
    """Everything needed to score raw text with the persisted baseline."""

    vectorizer: TfidfVectorizer
    model: Any
    name: str
    metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return P(REAL) for each input text."""
        features = self.vectorizer.transform(texts)
        proba = self.model.predict_proba(features)
        return np.asarray(proba)[:, 1]


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        min_df=1,
    )


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Standard binary metrics; ROC-AUC is NaN-safe for single-class inputs."""
    preds = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, proba)) if len(set(y_true.tolist())) > 1 else 0.5
    except ValueError:
        auc = 0.5
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": auc,
    }


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    line = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"{title:<28} {line}")


def train_logreg(x_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=2000,
        C=4.0,
        solver="liblinear",
        random_state=SEED,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost(x_train, y_train):
    """Train XGBoost; returns ``None`` if xgboost is unavailable/fails."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - optional dependency
        LOG.warning("XGBoost unavailable (%s); skipping.", exc)
        return None
    try:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=2,
            random_state=SEED,
            tree_method="hist",
            eval_metric="logloss",
        )
        model.fit(x_train, y_train)
        return model
    except Exception as exc:  # pragma: no cover
        LOG.warning("XGBoost training failed (%s); skipping.", exc)
        return None


def train_baseline(out_path: Path = BASELINE_PATH) -> BaselineArtifact:
    """Train both baselines, keep the best, persist it, return the artifact."""
    from src.data.split import load_splits

    set_seeds()
    ensure_dirs()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    train, val = splits["train"], splits["val"]
    x_text = train["text"].astype(str).tolist()
    y_train = train["label"].to_numpy(dtype=int)
    y_val = val["label"].to_numpy(dtype=int)

    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(x_text)
    x_val = vectorizer.transform(val["text"].astype(str).tolist())
    LOG.info("TF-IDF matrix: %s x %s", x_train.shape[0], x_train.shape[1])

    candidates: list[tuple[str, Any, dict[str, float]]] = []

    logreg = train_logreg(x_train, y_train)
    logreg_metrics = evaluate(y_val, logreg.predict_proba(x_val)[:, 1])
    candidates.append(("tfidf_logreg", logreg, logreg_metrics))

    xgb = train_xgboost(x_train, y_train)
    if xgb is not None:
        xgb_metrics = evaluate(y_val, xgb.predict_proba(x_val)[:, 1])
        candidates.append(("tfidf_xgboost", xgb, xgb_metrics))

    print("\n--- STEP 3 VERIFICATION (validation split) --------------------")
    for name, _, metrics in candidates:
        _print_metrics(name, metrics)

    best_name, best_model, best_metrics = max(
        candidates, key=lambda c: (c[2]["roc_auc"], c[2]["accuracy"])
    )
    artifact = BaselineArtifact(
        vectorizer=vectorizer, model=best_model, name=best_name, metrics=best_metrics
    )
    joblib.dump(artifact, out_path)
    (MODEL_DIR / "baseline_metrics.json").write_text(
        json.dumps({"best": best_name, "metrics": best_metrics}, indent=2), encoding="utf-8"
    )

    print(f"Best model saved  : {best_name} -> {out_path}")
    print("Expected          : file exists, metrics printed for both models")
    print("---------------------------------------------------------------\n")
    return artifact


def load_baseline(path: Path = BASELINE_PATH) -> BaselineArtifact:
    """Load the persisted baseline, training it on demand when absent."""
    if not path.exists():
        LOG.info("Baseline artifact missing at %s; training now.", path)
        return train_baseline(path)
    try:
        return joblib.load(path)
    except Exception as exc:
        LOG.warning("Failed to load baseline (%s); retraining.", exc)
        return train_baseline(path)


if __name__ == "__main__":
    train_baseline()
