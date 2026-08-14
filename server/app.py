from flask import Flask, request, jsonify
try:
    from .agent import LLMSearchAgent
    from .ml_model import classify, train_local_model
    from .config import settings
    from .cli import run_prompt
except ImportError:
    from agent import LLMSearchAgent
    from ml_model import classify, train_local_model
    from config import settings
    from cli import run_prompt

# Instantiate agent using configured provider (defaults to Gemini/HF as configured)
agent = LLMSearchAgent(provider=settings.LLM_PROVIDER, google_api_key=settings.GOOGLE_API_KEY)
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/search', methods=['POST'])
def search():
    payload = request.get_json() or {}
    query = payload.get('query')
    if not query:
        return jsonify({"error": "missing 'query' in request body"}), 400
    # allow optional provider/page_size overrides from client
    provider = payload.get('provider') or settings.LLM_PROVIDER
    page_size = int(payload.get('page_size') or 5)

    try:
        # create a fresh agent if caller requested a different provider
        if provider and provider != settings.LLM_PROVIDER:
            local_agent = LLMSearchAgent(provider=provider, google_api_key=settings.GOOGLE_API_KEY)
        else:
            local_agent = agent
        articles = local_agent.search_and_fetch(query, page_size=page_size)
    except Exception as e:
        return jsonify({"error": "agent error", "detail": str(e)}), 500

    results = []
    for a in articles:
        title = a.get('title') or ''
        source = a.get('source', {}).get('name') if isinstance(a.get('source'), dict) else a.get('source')
        url = a.get('url')
        content = a.get('content') or a.get('description') or ''
        classify_text = f"{title} {source or ''} {content}".strip()
        label, score = classify(classify_text)
        results.append({
            'title': title,
            'source': source,
            'url': url,
            'label': label,
            'score': score,
        })

    return jsonify({"query": query, "results": results})


@app.route('/classify', methods=['POST'])
def classify_text():
    """Classify a piece of text and return `{label, score}` for the client.

    Client posts JSON: {"text": "...", "provider": "hf|gemini|local" (optional)}
    """
    payload = request.get_json() or {}
    text = payload.get('text') or payload.get('query')
    if not text:
        return jsonify({"error": "missing 'text' in request body"}), 400

    try:
        label, score = classify(text)
        return jsonify({"label": label, "score": score, "text": text})
    except Exception as e:
        return jsonify({"error": "classification error", "detail": str(e)}), 500


@app.route('/run_prompt', methods=['POST'])
def run_prompt_route():
    """Run the same flow as the CLI: search + classification.

    Expects JSON: {"prompt": "...", "provider": "hf|gemini|local", "page_size": 5}
    """
    payload = request.get_json() or {}
    prompt = payload.get('prompt') or payload.get('query')
    if not prompt:
        return jsonify({"error": "missing 'prompt' in request body"}), 400
    provider = payload.get('provider')
    page_size = int(payload.get('page_size') or 5)
    try:
        result = run_prompt(prompt, provider=provider, page_size=page_size)
        return jsonify({
            "prompt": prompt,
            **result,
        })
    except Exception as e:
        return jsonify({"error": "run_prompt error", "detail": str(e)}), 500


@app.route('/train_local', methods=['POST'])
def train_local_route():
    """Trigger training of the local model. Expects JSON {"csv_path": "...", "out_path": "..."}.

    Use with caution — training may be slow and requires sklearn/pandas.
    """
    payload = request.get_json() or {}
    csv_path = payload.get('csv_path')
    out_path = payload.get('out_path')
    if not csv_path:
        return jsonify({"error": "missing 'csv_path' in request body"}), 400
    try:
        saved = train_local_model(csv_path, model_out=out_path)
        return jsonify({"status": "ok", "model_path": saved})
    except Exception as e:
        return jsonify({"error": "training error", "detail": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
