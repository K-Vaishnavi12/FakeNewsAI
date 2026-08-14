import json


def test_run_prompt_monkeypatch(monkeypatch):
    # stubbed articles
    articles = [
        {
            'title': 'Test Article',
            'source': {'name': 'example.com'},
            'url': 'https://example.com/article',
            'content': 'Some text about testing.'
        }
    ]

    # monkeypatch the agent search
    class FakeAgent:
        def __init__(self, provider=None):
            pass

        def search_and_fetch(self, query, page_size=5):
            return articles

    # Load the `server/cli.py` module directly (avoid package import issues)
    import importlib.util, importlib.machinery, pathlib, sys
    cli_path = pathlib.Path(__file__).parents[1] / 'cli.py'
    spec = importlib.util.spec_from_file_location('server.cli', str(cli_path))
    cli = importlib.util.module_from_spec(spec)
    sys.modules['server.cli'] = cli
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'LLMSearchAgent', FakeAgent)

    # patch classify to return deterministic label
    monkeypatch.setattr(cli, 'classify', lambda text: ('real', 0.95))

    res = cli.run_prompt('test query', provider='hf', page_size=1)
    assert isinstance(res, dict)
    assert res['verdict'] == 'valid'
    assert res['article']['label'] == 'real'
    assert res['article']['score'] == 0.95


def test_choose_best_article_prefers_related_match():
    import importlib.util, pathlib

    cli_path = pathlib.Path(__file__).parents[1] / 'cli.py'
    spec = importlib.util.spec_from_file_location('server.cli', str(cli_path))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    prompt = 'trump is president of usa'
    articles = [
        {
            'title': 'Trump president of USA confirms policy',
            'source': {'name': 'BBC News'},
            'url': 'https://example.com/a',
            'description': '',
            'content': '',
        },
        {
            'title': 'Completely unrelated sports story',
            'source': {'name': 'BBC News'},
            'url': 'https://example.com/b',
            'description': '',
            'content': '',
        },
    ]

    best = cli._choose_best_article(prompt, articles)
    assert best is not None
    assert best['title'] == 'Trump president of USA confirms policy'
