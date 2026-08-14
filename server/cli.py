import argparse
import json
import re
try:
    from .agent import LLMSearchAgent
    from .ml_model import classify
except ImportError:
    from agent import LLMSearchAgent
    from ml_model import classify


STOPWORDS = {
    'the', 'and', 'for', 'with', 'this', 'that', 'from', 'what', 'when', 'where', 'why', 'how',
    'are', 'was', 'were', 'been', 'being', 'into', 'onto', 'over', 'under', 'than', 'then', 'them',
    'they', 'their', 'there', 'here', 'about', 'after', 'before', 'would', 'could', 'should', 'has',
    'have', 'had', 'will', 'just', 'not', 'but', 'you', 'your', 'out', 'who', 'whom', 'which', 'into',
    'also', 'said', 'says', 'say', 'news', 'article', 'articles', 'latest', 'today', 'yesterday'
}


def _tokenize(text: str) -> set:
    return {
        token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _article_text(article: dict) -> str:
    title = article.get('title') or ''
    source = article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else article.get('source') or ''
    description = article.get('description') or ''
    content = article.get('content') or ''
    return f"{title} {source} {description} {content}".strip()


def _choose_best_article(prompt: str, articles: list) -> dict | None:
    prompt_tokens = _tokenize(prompt)
    if not prompt_tokens or not articles:
        return None

    best_article = None
    best_score = 0.0
    for article in articles:
        title_tokens = _tokenize(article.get('title') or '')
        body_tokens = _tokenize(_article_text(article))
        article_tokens = title_tokens | body_tokens
        if not article_tokens:
            continue
        title_overlap = len(prompt_tokens & title_tokens)
        body_overlap = len(prompt_tokens & body_tokens)
        overlap = (title_overlap * 3) + body_overlap
        score = overlap / max(len(prompt_tokens), 1)
        if score > best_score:
            best_score = score
            best_article = article

    if best_article is not None:
        best_article['_match_score'] = best_score
    return best_article


def _article_paragraph(article: dict) -> str:
    description = article.get('description') or ''
    content = article.get('content') or ''
    title = article.get('title') or ''
    parts = [part.strip() for part in (description, content) if part and part.strip()]
    if parts:
        paragraph = ' '.join(parts)
    else:
        paragraph = title
    return paragraph.rstrip(' .') + '.' if paragraph else ''


def run_prompt(prompt: str, provider: str = None, page_size: int = 5):
    agent = LLMSearchAgent(provider=provider)
    articles = agent.search_and_fetch(prompt, page_size=page_size)

    best_article = _choose_best_article(prompt, articles)
    if not best_article:
        return {
            'verdict': 'invalid',
            'message': 'NewsAPI did not return any articles for this input.',
            'article': None,
        }

    title = best_article.get('title') or ''
    source = best_article.get('source', {}).get('name') if isinstance(best_article.get('source'), dict) else best_article.get('source')
    url = best_article.get('url')
    article_text = _article_text(best_article)
    label, score = classify(article_text)
    paragraph = _article_paragraph(best_article)

    if label == 'real':
        verdict = 'valid'
        message = 'A NewsAPI article was found and classified as real.'
    elif label == 'fake':
        verdict = 'invalid'
        message = 'A NewsAPI article was found, but the classifier marked it as fake.'
    else:
        verdict = 'unknown'
        message = 'A NewsAPI article was found, but the classifier could not decide.'

    return {
        'verdict': verdict,
        'message': message,
        'article': {
            'title': title,
            'source': source,
            'url': url,
            'label': label,
            'score': score,
            'paragraph': paragraph,
        },
    }


def main():
    p = argparse.ArgumentParser(description='Run news search + HF LLM classification from terminal')
    p.add_argument('--prompt', '-p', help='Query/prompt to search for', required=False)
    p.add_argument('--provider', help='LLM provider override (hf|gemini)', default=None)
    p.add_argument('--page-size', type=int, default=5)
    args = p.parse_args()

    if not args.prompt:
        try:
            args.prompt = input('Enter a prompt to search news for: ')
        except EOFError:
            print('No prompt provided, exiting')
            return

    res = run_prompt(args.prompt, provider=args.provider, page_size=args.page_size)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
