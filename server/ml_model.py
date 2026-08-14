import os
import json
import re
try:
    from .config import settings
except ImportError:
    from config import settings

try:
    import google.genai as genai
except Exception:
    try:
        import google.generativeai as genai
    except Exception:
        genai = None
try:
    import local_llm
except Exception:
    local_llm = None


def classify(text: str):
    """Return (label, score).

    If MODEL_MODE is 'local' this will attempt to load a local model at
    `server/models/local_model.joblib` (a joblib tuple (clf, vectorizer)).
    Otherwise it will call Google Gemini (via genai) to classify.
    """
    if not text:
        return "unknown", 0.0

    if settings.MODEL_MODE == 'local':
        try:
            import joblib
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'local_model.joblib')
            if os.path.exists(model_path):
                clf, vectorizer = joblib.load(model_path)
                X = vectorizer.transform([text])
                probs = clf.predict_proba(X)[0]
                label = clf.classes_[probs.argmax()]
                score = float(probs.max())
                return label, score
            else:
                return "unknown", 0.0
        except Exception:
            return "unknown", 0.0

    # default: use local Hugging Face model if configured, else Gemini via Google GenAI SDK
    if local_llm is not None and settings.LLM_PROVIDER in ('hf', 'local'):
        try:
            # Build a classification prompt for the local HF model
            prompt = (
                "You are a classifier. Classify the following news article text as either 'real' or 'fake'.\n"
                "IMPORTANT: Respond with ONLY a single JSON object and nothing else.\n"
                "The JSON must have keys 'label' and 'score'. 'label' must be either 'real' or 'fake' (lowercase).\n"
                "'score' must be a number between 0.0 and 1.0 indicating confidence.\n"
                "Example: {\"label\": \"real\", \"score\": 0.92}\n\n"
                f"Text: '''{text}'''"
            )
            resp = local_llm.generate(prompt, max_output_tokens=60, temperature=0.0)
            content = resp.get('candidates', [{}])[0].get('content')
            # Try to extract a JSON object from the response content
            def _extract_json(s: str):
                if not s:
                    return None
                # find all {...} blocks and try to parse them
                matches = re.findall(r"\{.*?\}", s, flags=re.DOTALL)
                for m in matches:
                    try:
                        return json.loads(m)
                    except Exception:
                        continue
                # try greedy match if non-greedy failed
                matches = re.findall(r"\{.*\}", s, flags=re.DOTALL)
                for m in matches:
                    try:
                        return json.loads(m)
                    except Exception:
                        continue
                return None

            parsed = _extract_json(content)
            if parsed and isinstance(parsed, dict):
                return parsed.get('label', 'unknown'), float(parsed.get('score', 0.0))

            # fallback: try to find simple label/score patterns
            if content:
                m_label = re.search(r"label\W*[:=]\W*([a-zA-Z]+)", content, flags=re.IGNORECASE)
                m_score = re.search(r"score\W*[:=]\W*([0-9]*\.?[0-9]+)", content, flags=re.IGNORECASE)
                if m_label:
                    label = m_label.group(1).lower()
                    score = float(m_score.group(1)) if m_score else 0.0
                    return label, score

            # let outer fallback handle unknown
            
        except Exception:
            pass

    if genai is not None and settings.LLM_PROVIDER in ('gemini', 'google'):
        prompt = (
            "You are a classifier. Classify the following news article text as either 'real' or 'fake'.\n"
            "IMPORTANT: Respond with ONLY a single JSON object and nothing else.\n"
            "The JSON must have keys 'label' and 'score'. 'label' must be either 'real' or 'fake' (lowercase).\n"
            "'score' must be a number between 0.0 and 1.0 indicating confidence.\n"
            "Example: {\"label\": \"fake\", \"score\": 0.13}\n\n"
            f"Text: '''{text}'''"
        )
        try:
            if hasattr(genai, 'chat') and hasattr(genai.chat, 'create'):
                resp = genai.chat.create(model='gemini-2.5-flash', messages=[{"role": "user", "content": prompt}], temperature=0.0)
                # normalize possible response shapes
                content = None
                if isinstance(resp, dict):
                    content = resp.get('candidates', [{}])[0].get('content')
                else:
                    cand = getattr(resp, 'candidates', [None])[0]
                    if hasattr(cand, 'content'):
                        content = cand.content
                if not content:
                    content = str(resp)
            elif hasattr(genai, 'generate_text'):
                resp = genai.generate_text(model='gemini-2.5-flash', prompt=prompt)
                content = getattr(resp, 'text', None) or (resp.get('candidates', [{}])[0].get('output', ''))
            else:
                content = None

            if content:
                # reuse the same extraction logic
                def _extract_json_inner(s: str):
                    if not s:
                        return None
                    matches = re.findall(r"\{.*?\}", s, flags=re.DOTALL)
                    for m in matches:
                        try:
                            return json.loads(m)
                        except Exception:
                            continue
                    matches = re.findall(r"\{.*\}", s, flags=re.DOTALL)
                    for m in matches:
                        try:
                            return json.loads(m)
                        except Exception:
                            continue
                    return None

                parsed = _extract_json_inner(content)
                if parsed and isinstance(parsed, dict):
                    return parsed.get('label', 'unknown'), float(parsed.get('score', 0.0))
                # fallback simple parse
                m_label = re.search(r"label\W*[:=]\W*([a-zA-Z]+)", content, flags=re.IGNORECASE)
                m_score = re.search(r"score\W*[:=]\W*([0-9]*\.?[0-9]+)", content, flags=re.IGNORECASE)
                if m_label:
                    label = m_label.group(1).lower()
                    score = float(m_score.group(1)) if m_score else 0.0
                    return label, score
        except Exception:
            pass

    # fallback: local model or unknown
    # simple heuristic fallback to catch obvious cases when LLM parsing fails
    def _heuristic(text: str):
        t = (text or "").lower()
        real_indicators = ['reuters', 'associated press', 'apnews', 'ap ', 'bbc', 'cnn', 'nytimes', 'washington post', 'press']
        fake_indicators = ['hoax', 'false', 'not true', 'satire', 'satirical', 'deepfake', 'misinformation', 'fabricated', 'unverified', 'alleged']
        for r in real_indicators:
            if r in t:
                return 'real', 0.75
        for f in fake_indicators:
            if f in t:
                return 'fake', 0.8
        if t.count('!') > 0 or (t.isupper() and len(t) < 200):
            return 'fake', 0.6
        return 'unknown', 0.0

    if settings.MODEL_MODE == 'local':
        try:
            import joblib
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'local_model.joblib')
            if os.path.exists(model_path):
                clf, vectorizer = joblib.load(model_path)
                X = vectorizer.transform([text])
                probs = clf.predict_proba(X)[0]
                label = clf.classes_[probs.argmax()]
                score = float(probs.max())
                return label, score
            else:
                return "unknown", 0.0
        except Exception:
            return "unknown", 0.0

    # Try heuristic before giving up
    return _heuristic(text)


def train_local_model(csv_path: str, model_out: str = None):
    """Train a simple TF-IDF + LogisticRegression model from a CSV with `text,label` columns.
    Saves a joblib (clf, vectorizer) tuple to the models directory by default.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib

    df = pd.read_csv(csv_path)
    X = df['text'].fillna('')
    y = df['label']
    vect = TfidfVectorizer(max_features=20000)
    Xv = vect.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv, y)

    out_path = model_out or os.path.join(os.path.dirname(__file__), 'models', 'local_model.joblib')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump((clf, vect), out_path)
    return out_path
