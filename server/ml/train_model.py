"""Training pipeline for the fake-news classifier.

Builds a multi-scale TF-IDF + Logistic Regression model from whichever of the
supported datasets are present in the data directory:

- WELFake_Dataset.csv
- ISOT Fake.csv / True.csv
- BuzzFeed_{fake,real}_news_content.csv

Only the datasets actually found on disk are used, so the resulting model's
quality depends entirely on what has been downloaded. Check the printed
article count and the ``accuracy``/``eval_method`` fields in the saved bundle
before trusting a number.

Each article yields two training samples (full body and headline) so the model
handles both long articles and the short claims users paste in. The train/test
split is grouped by source article to keep that augmentation leakage-free.
"""

import argparse
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from ..constants import CLASS_LABELS
from ..paths import DATA_DIR
from ..datasets import (
    DEFAULT_DRIVE_FOLDER_URL,
    sync_from_drive,
    sync_from_github_release,
)

MODEL_TYPE = "Augmented Multi-Scale TF-IDF + Logistic Regression"


def clean_text(text: str) -> str:
    """Normalize text while preserving journalistic cues."""
    if not isinstance(text, str):
        return ""
    # Remove publisher location/tag patterns like "WASHINGTON (Reuters) -" or "(Reuters) -"
    cleaned = re.sub(r'^[A-Z\s,/\.]+\s*\([A-Za-z\s]+\)\s*[-—–]\s*', '', text)
    cleaned = re.sub(r'^\([A-Za-z\s]+\)\s*[-—–]\s*', '', cleaned)
    cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def find_data_dir(custom_dir: str = None) -> str:
    """Resolve the dataset directory.

    Uses the single ``paths.DATA_DIR`` constant rather than guessing across
    four candidate locations. The old search list silently fell back to a
    non-existent path, which is how an empty data directory turned into a
    model trained on whatever happened to be lying around.
    """
    if custom_dir and os.path.isdir(custom_dir):
        return custom_dir
    return DATA_DIR


def load_all_datasets(data_dir: str = None) -> pd.DataFrame:
    """Loads and standardizes all available datasets in data directory:
    Standard schema: DataFrame with columns ['title', 'text', 'label']
    Label convention: 0 = Fake, 1 = Real
    """
    data_dir = find_data_dir(data_dir)
    print(f"Scanning for datasets in: {data_dir}")

    dfs = []

    # 1. WELFake Dataset (WELFake_Dataset.csv)
    # Format: ['Unnamed: 0', 'title', 'text', 'label'] where 1=Fake, 0=Real
    welfake_path = os.path.join(data_dir, 'WELFake_Dataset.csv')
    if os.path.isfile(welfake_path):
        print(f"Loading WELFake Dataset: {welfake_path}")
        df_wel = pd.read_csv(welfake_path)
        # Drop unnamed columns
        df_wel = df_wel[['title', 'text', 'label']].dropna(subset=['label'])
        # Map WELFake labels (1=fake, 0=real) to our convention (0=fake, 1=real)
        df_wel['label'] = df_wel['label'].apply(lambda x: 0 if int(x) == 1 else 1)
        df_wel['title'] = df_wel['title'].fillna('')
        df_wel['text'] = df_wel['text'].fillna('')
        print(f"  -> Loaded WELFake: {len(df_wel):,} articles ({sum(df_wel['label']==0):,} Fake, {sum(df_wel['label']==1):,} Real)")
        dfs.append(df_wel)

    # 2. ISOT Dataset (Fake.csv & True.csv)
    fake_path = os.path.join(data_dir, 'Fake.csv')
    true_path = os.path.join(data_dir, 'True.csv')
    if os.path.isfile(fake_path) and os.path.isfile(true_path):
        print(f"Loading ISOT Dataset (Fake.csv & True.csv)")
        df_fake = pd.read_csv(fake_path)[['title', 'text']].dropna(how='all')
        df_fake['label'] = 0  # 0 = Fake
        df_true = pd.read_csv(true_path)[['title', 'text']].dropna(how='all')
        df_true['label'] = 1  # 1 = Real
        df_isot = pd.concat([df_fake, df_true], ignore_index=True)
        print(f"  -> Loaded ISOT: {len(df_isot):,} articles ({len(df_fake):,} Fake, {len(df_true):,} Real)")
        dfs.append(df_isot)

    # 3. BuzzFeed News Content
    bf_fake_path = os.path.join(data_dir, 'BuzzFeed_fake_news_content.csv')
    bf_real_path = os.path.join(data_dir, 'BuzzFeed_real_news_content.csv')
    if os.path.isfile(bf_fake_path) and os.path.isfile(bf_real_path):
        print("Loading BuzzFeed News Dataset")
        df_bf_fake = pd.read_csv(bf_fake_path)[['title', 'text']].dropna(how='all')
        df_bf_fake['label'] = 0
        df_bf_real = pd.read_csv(bf_real_path)[['title', 'text']].dropna(how='all')
        df_bf_real['label'] = 1
        df_bf = pd.concat([df_bf_fake, df_bf_real], ignore_index=True)
        print(f"  -> Loaded BuzzFeed: {len(df_bf):,} articles")
        dfs.append(df_bf)

    if not dfs:
        raise FileNotFoundError(f"No valid dataset CSV files found in {data_dir}")

    # Combine all
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['title'] = combined_df['title'].fillna('').astype(str)
    combined_df['text'] = combined_df['text'].fillna('').astype(str)
    
    # Remove exact duplicates
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['title', 'text']).reset_index(drop=True)
    print(f"\nTotal combined unique articles: {len(combined_df):,} (Deduplicated {initial_len - len(combined_df):,} duplicates)")
    fake_count = (combined_df['label'] == 0).sum()
    real_count = (combined_df['label'] == 1).sum()
    print(f"Dataset balance: {fake_count:,} Fake ({fake_count/len(combined_df)*100:.1f}%), {real_count:,} Real ({real_count/len(combined_df)*100:.1f}%)")
    
    return combined_df


def build_augmented_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Emit two training samples per article: full text, and title-only.

    Training on headlines as well as full bodies is what lets the model handle
    the short claims users actually paste in.

    Returns a DataFrame with columns ``['content', 'label', 'group']``. The
    ``group`` column is the source article's index and is essential: both
    samples derived from one article MUST stay on the same side of the
    train/test split, otherwise the model sees an article's headline during
    training and is then scored on that same article's body. That is textbook
    leakage and inflates reported accuracy substantially (measured on the
    BuzzFeed set: 82.6% leaky vs 67.1% grouped).
    """
    clean_titles = [clean_text(t) for t in df['title']]
    clean_texts = [clean_text(t) for t in df['text']]
    groups = np.arange(len(df))

    # Sample 1: full content (title + body)
    df_full = pd.DataFrame({
        'content': [f"{t} {b}".strip() for t, b in zip(clean_titles, clean_texts)],
        'label': df['label'].values,
        'group': groups,
    })

    # Sample 2: title only (short claim / headline)
    df_titles = pd.DataFrame({
        'content': clean_titles,
        'label': df['label'].values,
        'group': groups,
    })

    augmented = pd.concat([df_full, df_titles], ignore_index=True)
    # Drop empty/trivial samples and exact duplicates.
    augmented = augmented[augmented['content'].str.len() > 3]
    augmented = augmented.drop_duplicates(subset=['content']).reset_index(drop=True)
    print(f"Augmented dataset size (Full Articles + Headlines): "
          f"{len(augmented):,} unique training samples "
          f"from {augmented['group'].nunique():,} articles")
    return augmented


def train(data_dir: str = None, output_path: str = None):
    """Train and persist the fake-news classifier.

    Args:
        data_dir: Directory to scan for dataset CSVs. Defaults to server/data.
        output_path: Destination .joblib path. Callers reachable from HTTP MUST
            pass a path already validated by ``server.paths.resolve_model_output``.

    Returns:
        Tuple of ``(output_path, accuracy, classification_report, confusion_matrix)``.
    """
    start_time = time.time()
    print("=" * 70)
    print("  FakeNewsAI Advanced Multi-Dataset Training Pipeline")
    print("=" * 70)

    # 1. Load & Augment
    df_raw = load_all_datasets(data_dir=data_dir)
    df = build_augmented_dataset(df_raw)

    X = df['content']
    y = df['label'].astype(int)
    groups = df['group']

    # 2. Grouped 80/20 split.
    # GroupShuffleSplit (not train_test_split) keeps every sample derived from
    # the same source article on one side of the split, so the reported test
    # accuracy is honest rather than leakage-inflated.
    print("\nSplitting into 80% Train and 20% Test (grouped by source article)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"  Train set: {len(X_train):,} samples "
          f"({groups.iloc[train_idx].nunique():,} articles)")
    print(f"  Test set:  {len(X_test):,} samples "
          f"({groups.iloc[test_idx].nunique():,} articles)")
    if len(set(y_test)) < 2:
        print("  WARNING: test split contains a single class; "
              "accuracy will not be meaningful.")

    # 3. Build Multi-Scale TF-IDF Pipeline
    print("\nBuilding and training Pipeline (Word N-grams 1-3 + Calibrated Logistic Regression)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=75000,
            ngram_range=(1, 3),
            stop_words='english',
            sublinear_tf=True,
            min_df=3,
            strip_accents='unicode'
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=3.0,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42
            # n_jobs removed: it has no effect for the lbfgs solver and is
            # deprecated in scikit-learn >= 1.8.
        ))
    ])

    t_fit_start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Training completed in {time.time() - t_fit_start:.2f} seconds.")

    # 4. Comprehensive Evaluation
    print("\nEvaluating model on 20% held-out test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)'], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 60)
    print("Classification Report:\n", report)
    print("Confusion Matrix:")
    print(f"  True Fake  -> Predicted Fake: {cm[0][0]:,} | Predicted Real: {cm[0][1]:,}")
    print(f"  True Real  -> Predicted Fake: {cm[1][0]:,} | Predicted Real: {cm[1][1]:,}")
    print("=" * 60)

    # 5. Extract top predictive vocabulary signals
    vectorizer = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]

    top_real_idx = np.argsort(coef)[-25:][::-1]
    top_fake_idx = np.argsort(coef)[:25]

    print("\nTop 10 Linguistic Indicators for REAL News:")
    for idx in top_real_idx[:10]:
        print(f"  + {feature_names[idx]:<28} (weight: +{coef[idx]:.3f})")

    print("\nTop 10 Linguistic Indicators for FAKE News:")
    for idx in top_fake_idx[:10]:
        print(f"  - {feature_names[idx]:<28} (weight: {coef[idx]:.3f})")

    # 6. Save model bundle
    if not output_path:
        output_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'fake_news_model.joblib')
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model_bundle = {
        'pipeline': pipeline,
        # Consumed by predict.py (accuracy_display) and surfaced by /api/health
        # so the UI never has to hardcode an accuracy figure.
        'accuracy': float(accuracy),
        # Records HOW accuracy was measured, so a future reader can tell a
        # grouped (honest) number from the older leakage-inflated one.
        'eval_method': 'grouped-80-20-holdout',
        'model_type': MODEL_TYPE,
        # Index-aligned with the integer class labels used during training
        # (0 = fake, 1 = real). predict.py cross-checks this against
        # pipeline.classes_ rather than assuming a column order.
        'classes': list(CLASS_LABELS),
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'total_samples': len(df),
        'total_articles': len(df_raw),
        'top_indicators': {
            'real': list(feature_names[top_real_idx]),
            'fake': list(feature_names[top_fake_idx])
        }
    }

    joblib.dump(model_bundle, output_path, compress=3)
    print(f"\nModel bundle saved successfully to:\n  -> {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print(f"Total pipeline elapsed time: {time.time() - start_time:.2f} seconds.")
    print("=" * 70)
    return output_path, accuracy, report, cm


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Multi-Dataset Fake News Classifier")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset CSVs")
    parser.add_argument("--output", type=str, default=None, help="Path to output .joblib model")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download datasets before training (see --download-from).",
    )
    parser.add_argument(
        "--download-from",
        choices=("release", "drive"),
        default="release",
        help="Source for --download-data: the GitHub release (default) or the "
             "Google Drive mirror.",
    )
    parser.add_argument(
        "--drive-url",
        type=str,
        default=DEFAULT_DRIVE_FOLDER_URL,
        help="Public Google Drive folder URL, used with --download-from drive.",
    )
    parser.add_argument(
        "--purge-existing-data",
        action="store_true",
        help="Delete existing dataset files in data dir before downloading.",
    )
    args = parser.parse_args()

    if args.download_data:
        target_dir = args.data_dir or find_data_dir(None)
        print(f"Preparing datasets in: {target_dir}")
        if args.download_from == "drive":
            synced = sync_from_drive(
                folder_url=args.drive_url,
                data_dir=target_dir,
                purge_existing=args.purge_existing_data,
            )
            print(f"Downloaded {len(synced)} file(s) from Google Drive.")
        else:
            synced = sync_from_github_release(data_dir=target_dir)
            print(f"Downloaded {len(synced)} file(s) from the GitHub release.")

    train(data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
