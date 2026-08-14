import requests
import xml.etree.ElementTree as ET
import re
import urllib.parse
try:
    from .config import settings
except ImportError:
    from config import settings


NEWSAPI_ENDPOINT = 'https://newsapi.org/v2/everything'
GOOGLE_NEWS_RSS = 'https://news.google.com/rss/search'


def _fetch_google_news_rss(query: str, page_size: int = 10) -> list:
    """Fallback: fetch articles from Google News RSS (free, no API key needed).

    Returns articles in the same dict format as NewsAPI for compatibility.
    """
    params = {
        'q': query,
        'hl': 'en-US',
        'gl': 'US',
        'ceid': 'US:en',
    }
    try:
        resp = requests.get(GOOGLE_NEWS_RSS, params=params, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    articles = []
    for item in root.iter('item'):
        if len(articles) >= page_size:
            break
        title_el = item.find('title')
        link_el = item.find('link')
        pub_date_el = item.find('pubDate')
        # Google News titles often have " - Source Name" at the end
        raw_title = title_el.text if title_el is not None else ''
        source_name = ''
        if ' - ' in raw_title:
            parts = raw_title.rsplit(' - ', 1)
            title = parts[0].strip()
            source_name = parts[1].strip()
        else:
            title = raw_title

        description_el = item.find('description')
        desc_text = ''
        if description_el is not None and description_el.text:
            # Strip HTML tags from description
            desc_text = re.sub(r'<[^>]+>', '', description_el.text).strip()

        articles.append({
            'title': title,
            'source': {'name': source_name} if source_name else {'name': 'Google News'},
            'url': link_el.text if link_el is not None else '',
            'description': desc_text or title,
            'content': desc_text,
            'publishedAt': pub_date_el.text if pub_date_el is not None else '',
        })

    return articles


def search_news(query: str, domains: list = None, page_size: int = 10) -> list:
    """Search news articles using NewsAPI.org, falling back to Google News RSS.

    Tries NewsAPI first. If it returns zero articles (common on free-tier plans),
    automatically falls back to Google News RSS which is free and keyless.
    """
    articles = []

    # 1. Try NewsAPI if key is available
    api_key = settings.NEWSAPI_KEY
    if api_key:
        try:
            params = {
                'q': query,
                'language': 'en',
                'pageSize': page_size,
                'apiKey': api_key,
            }
            if domains:
                params['domains'] = ','.join(domains)
            resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get('articles', [])
        except Exception:
            articles = []

    # 2. Fallback to Google News RSS if NewsAPI returned nothing
    if not articles:
        articles = _fetch_google_news_rss(query, page_size=page_size)

    return articles
