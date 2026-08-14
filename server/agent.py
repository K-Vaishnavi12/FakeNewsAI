import os
import json
try:
    from .news_fetcher import search_news
    from .config import settings
except ImportError:
    from news_fetcher import search_news
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


class LLMSearchAgent:
    def __init__(self, provider: str = None, google_api_key: str = None):
        # force Gemini as the default provider
        self.provider = provider or settings.LLM_PROVIDER or 'gemini'

        # google generative setup (API key approach)
        self.google_api_key = google_api_key or settings.GOOGLE_API_KEY or settings.GOOGLE_APPLICATION_CREDENTIALS
        if genai and self.provider in ('gemini', 'google') and self.google_api_key:
            try:
                # genai supports configure for API key
                genai.configure(api_key=self.google_api_key)
            except Exception:
                # some environments may prefer GOOGLE_APPLICATION_CREDENTIALS env var
                pass

    def _expand_query(self, query: str) -> dict:
        prompt = (
            "You are an assistant that helps craft web-search parameters.\n"
            "Given the user's query, return a JSON object with two keys:\n"
            "- \"keywords\": a short search phrase focused on the query.\n"
            "- \"domains\": a list of 5 trusted news domains (domain names only) to prefer.\n"
            f"User query: {query}\n\n"
            "Return only valid JSON."
        )

        # If configured to use a local Hugging Face model, use it first
        if self.provider in ('hf', 'local') and local_llm is not None:
            try:
                resp = local_llm.generate(prompt, max_output_tokens=200, temperature=0.0)
                content = resp.get('candidates', [{}])[0].get('content')
                parsed = json.loads(content)
                return parsed
            except Exception:
                pass

        # Use provider-specific LLM to expand query (prefer Gemini 2.5 Flash)
        if self.provider in ('gemini', 'google') and genai is not None:
            try:
                # try chat-style call
                # new and old SDKs expose slightly different shapes; try chat first
                if hasattr(genai, 'chat') and hasattr(genai.chat, 'create'):
                    resp = genai.chat.create(model='gemini-2.5-flash', messages=[{"role": "user", "content": prompt}])
                    # try attribute access styles
                    content = None
                    if isinstance(resp, dict):
                        content = resp.get('candidates', [{}])[0].get('content')
                    else:
                        # new SDKs may provide object with candidates
                        content = getattr(resp, 'candidates', [None])[0]
                        if hasattr(content, 'content'):
                            content = content.content
                    if not content:
                        content = str(resp)
                elif hasattr(genai, 'generate_text'):
                    resp = genai.generate_text(model='gemini-2.5-flash', prompt=prompt)
                    content = getattr(resp, 'text', None) or (resp.get('candidates', [{}])[0].get('output', ''))
                else:
                    # SDK doesn't have expected helpers; raise to trigger default fallback
                    raise Exception('genai installed but no compatible API surface found')
                parsed = json.loads(content)
                return parsed
            except Exception:
                pass

        # fallback: return a safe default set if Gemini/SDK is unavailable
        return {"keywords": query, "domains": [
            "reuters.com", "apnews.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com"
        ]}

    def search_and_fetch(self, query: str, page_size: int = 10):
        params = self._expand_query(query)
        keywords = params.get('keywords') or query
        domains = params.get('domains')
        articles = search_news(keywords, domains=domains, page_size=page_size)
        return articles

    def simple_search(self, query: str, page_size: int = 10):
        return search_news(query, page_size=page_size)
