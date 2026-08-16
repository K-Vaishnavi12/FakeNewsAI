"""
Enhanced Multi-Dataset ML Training Pipeline for Fake News Detection
Trains an optimized TF-IDF + Logistic Regression Classifier on multiple benchmark datasets:
- WELFake Dataset (72,134 articles)
- ISOT Fake & True News (44,898 articles)
- BuzzFeed News (182 articles)
Totaling over 117,000+ real-world articles, augmented into 200,000+ training samples
for high accuracy on both short claims and long-form articles.
"""

import os
import re
import sys
import time
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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
    search_dirs = [
        custom_dir,
        os.path.join(os.path.dirname(__file__), 'data'),
        os.path.join(os.path.dirname(__file__), '..', 'data'),
        os.path.join(os.getcwd(), 'data'),
        os.path.join(os.getcwd(), 'server', 'data'),
    ]
    for d in search_dirs:
        if d and os.path.isdir(d):
            return d
    return os.path.join(os.path.dirname(__file__), '..', 'data')


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
    """Augment dataset with both full article text AND title-only samples.
    This ensures the model excels at both full articles and short user claims/headlines!
    """
    clean_titles = [clean_text(t) for t in df['title']]
    clean_texts = [clean_text(t) for t in df['text']]

    # Sample 1: Full content (Title + Text)
    full_content = [f"{t} {b}".strip() for t, b in zip(clean_titles, clean_texts)]
    df_full = pd.DataFrame({'content': full_content, 'label': df['label']})

    # Sample 2: Title-only (Short claim / headline)
    df_titles = pd.DataFrame({'content': clean_titles, 'label': df['label']})

    # Combine
    augmented = pd.concat([df_full, df_titles], ignore_index=True)
    # Filter out empty or trivial samples (less than 4 characters)
    augmented = augmented[augmented['content'].str.len() > 3].reset_index(drop=True)
    augmented = augmented.drop_duplicates(subset=['content']).reset_index(drop=True)
    print(f"Augmented dataset size (Full Articles + Headlines): {len(augmented):,} unique training samples")
    return augmented


def train(data_dir: str = None, news_csv_path: str = None, output_path: str = None):
    start_time = time.time()
    print("=" * 70)
    print("  FakeNewsAI Advanced Multi-Dataset Training Pipeline")
    print("=" * 70)

    # 1. Load & Augment
    df_raw = load_all_datasets(data_dir=data_dir)
    df = build_augmented_dataset(df_raw)

    X = df['content']
    y = df['label'].astype(int)

    # 2. Stratified Train/Test Split (80% Train, 20% Test)
    print("\nSplitting into 80% Train and 20% Test (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train set: {len(X_train):,} samples")
    print(f"  Test set:  {len(X_test):,} samples")

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
            random_state=42,
            n_jobs=-1
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
        'accuracy': float(accuracy),
        'classes': ['fake', 'real'],
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
    args = parser.parse_args()

    train(data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
