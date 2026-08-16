"""
Flask Backend for Fake News AI
Orchestrates AI Agent, ML Classifier, NewsAPI/RSS Search, and Gemini LLM Synthesis
Pre-loads trained TF-IDF + Logistic Regression model on startup.
"""

from flask import Flask, request, jsonify
import os
import sys

try:
    from .agent import LLMSearchAgent
    from .ml_model import classify, classify_with_probabilities, train_local_model, is_model_loaded
    from .config import settings
except ImportError:
    from agent import LLMSearchAgent
    from ml_model import classify, classify_with_probabilities, train_local_model, is_model_loaded
    from config import settings

app = Flask(__name__)

# Basic CORS headers support
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# Instantiate AI Agent once on startup (loads pre-trained ML model into memory)
agent = LLMSearchAgent(provider=settings.LLM_PROVIDER)
print(f"[Flask] Server initialized. Model loaded: {is_model_loaded()}. Provider: {settings.LLM_PROVIDER}")


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "model_type": "TF-IDF + Logistic Regression",
        "model_accuracy": "98.80%",
        "llm_provider": settings.LLM_PROVIDER,
    })


@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_claim():
    """Primary endpoint for the complete Fake News Detection Architecture:
    User -> Frontend -> Flask -> AI Agent -> [ML Classifier + NewsAPI] -> Gemini LLM -> Final Analysis
    """
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    payload = request.get_json() or {}
    text = payload.get('text') or payload.get('prompt') or payload.get('query')
    if not text or not str(text).strip():
        return jsonify({"error": "Missing 'text' in request body"}), 400

    page_size = int(payload.get('page_size') or 5)
    try:
        result = agent.analyze(str(text), page_size=page_size)
        return jsonify(result)
    except Exception as e:
        print(f"[Flask] Error during analyze: {e}")
        return jsonify({"error": "Analysis failed", "detail": str(e)}), 500


@app.route('/api/run_prompt', methods=['POST', 'OPTIONS'])
@app.route('/run_prompt', methods=['POST', 'OPTIONS'])
def run_prompt_route():
    """Backward-compatible endpoint providing both new analysis and legacy fields."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    payload = request.get_json() or {}
    prompt = payload.get('prompt') or payload.get('query') or payload.get('text')
    if not prompt or not str(prompt).strip():
        return jsonify({"error": "missing 'prompt' in request body"}), 400

    page_size = int(payload.get('page_size') or 5)
    try:
        full_res = agent.analyze(str(prompt), page_size=page_size)
        
        # Build backward-compatible fields
        final_an = full_res.get('final_analysis', {})
        ml = full_res.get('ml_classifier', {})
        sources = full_res.get('news_sources', [])
        
        verdict = "valid" if ml.get('label') == 'real' else ("invalid" if ml.get('label') == 'fake' else "unknown")
        matched_article = None
        if sources:
            first = sources[0]
            matched_article = {
                'title': first.get('title'),
                'source': first.get('source'),
                'url': first.get('url'),
                'label': ml.get('label'),
                'score': ml.get('confidence'),
                'paragraph': first.get('description'),
            }

        return jsonify({
            "prompt": prompt,
            "verdict": verdict,
            "message": final_an.get('executive_summary', ''),
            "article": matched_article,
            **full_res
        })
    except Exception as e:
        print(f"[Flask] Error in run_prompt: {e}")
        return jsonify({"error": "run_prompt error", "detail": str(e)}), 500


@app.route('/api/classify', methods=['POST', 'OPTIONS'])
@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify_text_route():
    """Direct ML classification endpoint returning probabilities and keyword signals."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    payload = request.get_json() or {}
    text = payload.get('text') or payload.get('query')
    if not text:
        return jsonify({"error": "missing 'text' in request body"}), 400

    try:
        result = classify_with_probabilities(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "classification error", "detail": str(e)}), 500


@app.route('/api/search', methods=['POST', 'OPTIONS'])
@app.route('/search', methods=['POST', 'OPTIONS'])
def search_route():
    """Search news articles and evaluate each with the ML model."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    payload = request.get_json() or {}
    query = payload.get('query')
    if not query:
        return jsonify({"error": "missing 'query' in request body"}), 400

    page_size = int(payload.get('page_size') or 5)
    try:
        articles = agent.search_and_fetch(query, page_size=page_size)
        results = []
        for a in articles:
            title = a.get('title') or ''
            source = a.get('source', {}).get('name') if isinstance(a.get('source'), dict) else a.get('source')
            url = a.get('url')
            content = a.get('content') or a.get('description') or ''
            classify_text = f"{title} {source or ''} {content}".strip()
            ml_meta = classify_with_probabilities(classify_text)
            results.append({
                'title': title,
                'source': source,
                'url': url,
                'label': ml_meta.get('label'),
                'fake_probability': ml_meta.get('fake_probability'),
                'real_probability': ml_meta.get('real_probability'),
                'score': ml_meta.get('score'),
            })
        return jsonify({"query": query, "results": results})
    except Exception as e:
        return jsonify({"error": "search error", "detail": str(e)}), 500


@app.route('/api/train_local', methods=['POST', 'OPTIONS'])
@app.route('/train_local', methods=['POST', 'OPTIONS'])
def train_local_route():
    """Trigger training pipeline."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    payload = request.get_json() or {}
    csv_path = payload.get('csv_path')
    out_path = payload.get('out_path')
    try:
        saved = train_local_model(csv_path=csv_path, model_out=out_path)
        return jsonify({"status": "ok", "model_path": saved})
    except Exception as e:
        return jsonify({"error": "training error", "detail": str(e)}), 500


if __name__ == '__main__':
    host = settings.FLASK_HOST
    port = settings.FLASK_PORT
    debug = settings.FLASK_DEBUG

    # Never expose Werkzeug debug mode on a public interface.
    if debug and host == '0.0.0.0':
        host = '127.0.0.1'

    app.run(host=host, port=port, debug=debug)
