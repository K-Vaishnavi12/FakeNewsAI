from cli import run_prompt


def test_cli_run_prompt(monkeypatch):
    class FakeAgent:
        def __init__(self, provider=None):
            pass

        def analyze(self, prompt, page_size=5):
            return {
                "query": prompt,
                "final_analysis": {
                    "verdict": "Likely Real / Verified",
                    "verdict_type": "real",
                    "confidence_score": 0.92,
                    "executive_summary": "Substantiated by verified reporting."
                },
                "ml_classifier": {
                    "label": "real",
                    "fake_probability": 0.08,
                    "real_probability": 0.92,
                    "confidence": 0.92,
                    "top_signals": []
                },
                "news_sources": [
                    {
                        "title": "Confirmed Real News Event",
                        "source": "BBC News",
                        "url": "https://bbc.com/news/123",
                        "stance": "SUPPORTS CLAIM"
                    }
                ]
            }

    import cli
    monkeypatch.setattr(cli, 'LLMSearchAgent', FakeAgent)

    res = run_prompt("Confirmed Real News Event", provider="hf", page_size=1)
    assert isinstance(res, dict)
    assert res["final_analysis"]["verdict_type"] == "real"
    assert res["ml_classifier"]["label"] == "real"
