"""News API adapter.

Design rules enforced here:

* The API key is read from the environment and sent in a request **header** -
  never in a query string that could be logged, and never returned to a caller.
* Every network path is wrapped: timeout, invalid key, rate limit, empty result,
  malformed payload and generic network failure all produce a structured
  `NewsAPIResponse` instead of an exception.
* A failure returns an *empty evidence list*, so the pipeline degrades to
  "could not be independently verified" rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from app.config import Settings, get_settings


@dataclass
class NewsAPIResponse:
    """Outcome of one or more News API calls."""

    ok: bool
    articles: list[dict] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    total_results: int = 0
    retrieved_at: str = ""


# User-facing messages. Deliberately neutral and never leak configuration.
ERROR_MESSAGES = {
    "TIMEOUT": (
        "The News API request timed out. This result should be treated as "
        "unverified."
    ),
    "RATE_LIMITED": (
        "The News API request failed or was rate-limited. Please try again "
        "later. This result should be treated as unverified."
    ),
    "INVALID_KEY": (
        "The News API rejected the configured credentials, so no independent "
        "source search could be performed. This result should be treated as "
        "unverified."
    ),
    "NOT_CONFIGURED": (
        "News API search unavailable — this result was not independently "
        "verified through the news search service."
    ),
    "MALFORMED": (
        "The News API returned an unexpected response format, so no sources "
        "could be read. This result should be treated as unverified."
    ),
    "NETWORK": (
        "The News API could not be reached. This result should be treated as "
        "unverified."
    ),
    "SERVER_ERROR": (
        "The News API reported a server error. Please try again later. This "
        "result should be treated as unverified."
    ),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NewsAPIAdapter:
    """Thin, defensive client around a NewsAPI-compatible `everything` endpoint."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()

    # --- internals --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "X-Api-Key": self.settings.news_api_key,
            "Accept": "application/json",
            "User-Agent": "VeritasCheck/1.0",
        }

    @staticmethod
    def _classify_http_error(status: int, payload: dict | None) -> tuple[str, str]:
        code = ""
        if isinstance(payload, dict):
            code = str(payload.get("code", "")).lower()

        if status in (401, 403) or code in {
            "apikeyinvalid",
            "apikeymissing",
            "apikeydisabled",
            "apikeyexhausted",
        }:
            return "INVALID_KEY", ERROR_MESSAGES["INVALID_KEY"]
        if status == 429 or code == "ratelimited":
            return "RATE_LIMITED", ERROR_MESSAGES["RATE_LIMITED"]
        if status >= 500:
            return "SERVER_ERROR", ERROR_MESSAGES["SERVER_ERROR"]
        return "MALFORMED", ERROR_MESSAGES["MALFORMED"]

    # --- public API -------------------------------------------------------

    def search(self, query: str, page_size: int | None = None) -> NewsAPIResponse:
        """Run a single query. Never raises."""
        retrieved_at = _now_iso()

        if not self.settings.news_api_configured:
            return NewsAPIResponse(
                ok=False,
                error=ERROR_MESSAGES["NOT_CONFIGURED"],
                error_code="NOT_CONFIGURED",
                retrieved_at=retrieved_at,
            )

        if not query or not query.strip():
            return NewsAPIResponse(
                ok=True, articles=[], retrieved_at=retrieved_at, total_results=0
            )

        params = {
            "q": query.strip(),
            "pageSize": page_size or self.settings.news_api_page_size,
            "language": "en",
            "sortBy": "relevancy",
        }

        try:
            response = self.session.get(
                self.settings.news_api_url,
                params=params,
                headers=self._headers(),
                timeout=self.settings.news_api_timeout_seconds,
            )
        except requests.exceptions.Timeout:
            return NewsAPIResponse(
                ok=False,
                error=ERROR_MESSAGES["TIMEOUT"],
                error_code="TIMEOUT",
                retrieved_at=retrieved_at,
            )
        except requests.exceptions.RequestException:
            return NewsAPIResponse(
                ok=False,
                error=ERROR_MESSAGES["NETWORK"],
                error_code="NETWORK",
                retrieved_at=retrieved_at,
            )

        # Parse the body defensively before looking at the status code, because
        # NewsAPI puts a machine-readable `code` field in error bodies too.
        payload: dict | None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code != 200:
            error_code, message = self._classify_http_error(
                response.status_code, payload
            )
            return NewsAPIResponse(
                ok=False,
                error=message,
                error_code=error_code,
                retrieved_at=retrieved_at,
            )

        if not isinstance(payload, dict):
            return NewsAPIResponse(
                ok=False,
                error=ERROR_MESSAGES["MALFORMED"],
                error_code="MALFORMED",
                retrieved_at=retrieved_at,
            )

        if str(payload.get("status", "ok")).lower() == "error":
            error_code, message = self._classify_http_error(200, payload)
            return NewsAPIResponse(
                ok=False, error=message, error_code=error_code, retrieved_at=retrieved_at
            )

        articles = payload.get("articles")
        if articles is None:
            articles = []
        if not isinstance(articles, list):
            return NewsAPIResponse(
                ok=False,
                error=ERROR_MESSAGES["MALFORMED"],
                error_code="MALFORMED",
                retrieved_at=retrieved_at,
            )

        clean = [a for a in articles if isinstance(a, dict)]
        total = payload.get("totalResults")
        return NewsAPIResponse(
            ok=True,
            articles=clean,
            retrieved_at=retrieved_at,
            total_results=int(total) if isinstance(total, int) else len(clean),
        )

    def search_many(self, queries: list) -> tuple[list[tuple[str, str, dict]], str | None, str | None]:
        """Run several queries.

        Returns `(records, error, error_code)` where each record is
        `(query_id, query_text, raw_article)`.

        Partial success is honoured: if one query fails but another succeeds we
        keep the successful results and still report the error as a warning.
        """
        records: list[tuple[str, str, dict]] = []
        first_error: str | None = None
        first_error_code: str | None = None
        any_ok = False

        for query in queries:
            query_id = getattr(query, "query_id", "QUERY-001")
            query_text = getattr(query, "query_text", str(query))
            result = self.search(query_text)
            if result.ok:
                any_ok = True
                for article in result.articles:
                    records.append((query_id, query_text, article))
            elif first_error is None:
                first_error = result.error
                first_error_code = result.error_code

        if any_ok:
            # Downgrade a partial failure to a warning, not a hard error.
            return records, (None if not first_error else first_error), first_error_code
        return records, first_error, first_error_code
