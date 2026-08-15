"""Train the VeritasCheck ensemble and persist artifacts to `artifacts/`.

Usage
-----
    python -m app.ml.train
    python -m app.ml.train --samples-per-class 2000
    python -m app.ml.train --csv data/train.csv
    python -m app.ml.train --augment-from-newsapi

Ensemble members
----------------
1. TF-IDF (word 1-2 grams)      -> Logistic Regression
2. TF-IDF (word 1-2 grams)      -> Linear SVM (calibrated for probabilities)
3. TF-IDF (word 1-2 grams)      -> Multinomial Naive Bayes
4. Engineered style features    -> Random Forest (or XGBoost if installed)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from app.config import PROJECT_ROOT, get_settings
from app.ml.dataset import build_training_dataset
from app.ml.features import StyleFeatureExtractor

ARTIFACT_VERSION = "1.0.0"


def _build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=50000,
        sublinear_tf=True,
    )


def _build_tree_model():
    """Prefer XGBoost when available, otherwise RandomForest."""
    try:
        from xgboost import XGBClassifier  # type: ignore

        return (
            "xgboost_style_features",
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        )
    except ImportError:
        return (
            "random_forest_style_features",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        )


def build_pipelines() -> dict[str, Pipeline]:
    tree_name, tree_estimator = _build_tree_model()
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", _build_tfidf()),
                ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=42)),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", _build_tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(C=1.0, random_state=42), cv=3, method="sigmoid"
                    ),
                ),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", _build_tfidf()),
                ("clf", MultinomialNB(alpha=0.3)),
            ]
        ),
        tree_name: Pipeline(
            [
                ("style", StyleFeatureExtractor()),
                ("scale", StandardScaler()),
                ("clf", tree_estimator),
            ]
        ),
    }


def _harvest_real_headlines(limit: int = 200) -> list[str]:
    """Optionally pull genuine headlines from the News API as REAL examples.

    Fails soft: returns [] on any error so training never breaks.
    """
    from app.news.adapter import NewsAPIAdapter

    settings = get_settings()
    if not settings.news_api_configured:
        print("  News API key not configured - skipping augmentation.")
        return []

    adapter = NewsAPIAdapter()
    seeds = [
        "government policy", "central bank", "public health",
        "transport infrastructure", "election commission", "climate report",
    ]
    collected: list[str] = []
    for seed in seeds:
        result = adapter.search(seed, page_size=50)
        if not result.ok:
            print(f"  News API error for '{seed}': {result.error}")
            continue
        for article in result.articles:
            parts = [
                p
                for p in (article.get("title"), article.get("description"))
                if p and isinstance(p, str)
            ]
            if parts:
                collected.append(". ".join(parts))
        if len(collected) >= limit:
            break
    print(f"  Harvested {len(collected)} REAL-class samples from the News API.")
    return collected[:limit]


def train(
    samples_per_class: int = 1200,
    csv_path: Path | None = None,
    augment_from_newsapi: bool = False,
    output_dir: Path | None = None,
) -> dict:
    settings = get_settings()
    output_dir = output_dir or settings.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_real: list[str] = []
    if augment_from_newsapi:
        print("Augmenting REAL class from the News API ...")
        extra_real = _harvest_real_headlines()

    dataset = build_training_dataset(
        csv_path=csv_path,
        samples_per_class=samples_per_class,
        extra_real_texts=extra_real,
    )
    print(
        f"Dataset: {len(dataset)} samples "
        f"(real={dataset.real_count}, fake={dataset.fake_count}) "
        f"origin={dataset.origin}"
    )

    X = np.asarray(dataset.texts, dtype=object)
    y = np.asarray(dataset.labels, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = build_pipelines()
    metrics: dict[str, dict] = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name} ...")
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))
        metrics[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"  accuracy={acc:.4f} macro_f1={f1:.4f}")
        print(
            classification_report(
                y_test, preds, target_names=["FAKE", "REAL"], zero_division=0
            )
        )
        joblib.dump(pipeline, output_dir / f"{name}.joblib")

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_origin": dataset.origin,
        "dataset_size": len(dataset),
        "label_mapping": {"0": "FAKE", "1": "REAL"},
        "models": list(pipelines.keys()),
        "metrics": metrics,
        "caveat": (
            "Trained on a stylistic corpus. The model detects sensational writing "
            "style, not factual truth. Treat its output as a weak signal only."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts written to {output_dir}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the VeritasCheck ensemble.")
    parser.add_argument("--samples-per-class", type=int, default=1200)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--augment-from-newsapi", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    train(
        samples_per_class=args.samples_per_class,
        csv_path=args.csv or (PROJECT_ROOT / "data" / "train.csv"),
        augment_from_newsapi=args.augment_from_newsapi,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
