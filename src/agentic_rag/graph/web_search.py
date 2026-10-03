from __future__ import annotations

import hashlib
import os
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agentic_rag.graph.documents import EvidenceDocument


class WebSearchResult(BaseModel):
    """Normalized result of one external web search."""

    status: Literal["complete", "empty", "unavailable", "failed"]
    query: str
    documents: list[EvidenceDocument] = Field(default_factory=list)
    error: str = ""


def is_safe_web_url(value: str) -> bool:
    """Accept only ordinary HTTP and HTTPS result URLs."""

    try:
        parsed = urlparse(value)
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def create_web_source_id(url: str) -> str:
    """Create a stable citation ID from the result URL."""

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return "web-" + digest[:12]


def convert_tavily_results(
    raw_results: list[dict[str, Any]],
    maximum_content_characters: int = 4000,
) -> list[EvidenceDocument]:
    """Convert Tavily dictionaries into normal graph evidence."""

    if maximum_content_characters <= 0:
        raise ValueError(
            "maximum_content_characters must be positive."
        )

    documents: list[EvidenceDocument] = []
    seen_urls: set[str] = set()

    for raw_result in raw_results:
        url = str(raw_result.get("url") or "").strip()
        content = str(raw_result.get("content") or "").strip()

        if not is_safe_web_url(url):
            continue

        if not content:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        if len(content) > maximum_content_characters:
            content = (
                content[:maximum_content_characters].rstrip()
                + "..."
            )

        title = str(
            raw_result.get("title") or "Untitled web source"
        ).strip()
        source_id = create_web_source_id(url)

        document = EvidenceDocument(
            source_id=source_id,
            paper_id=source_id,
            title=title,
            text=content,
            section_heading="Web search result",
            url=url,
            source="web",
            retrieval_sources=["tavily"],
        )

        documents.append(document)

    return documents


class TavilyWebSearch:
    """Small injectable wrapper around Tavily's search client."""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: str = "advanced",
        topic: str = "general",
        maximum_content_characters: int = 4000,
        client: Any | None = None,
    ) -> None:
        if max_results <= 0:
            raise ValueError("max_results must be positive.")

        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self.search_depth = search_depth
        self.topic = topic
        self.maximum_content_characters = (
            maximum_content_characters
        )
        self.client = client

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        client: Any | None = None,
    ) -> "TavilyWebSearch":
        """Create web search from the YAML configuration."""

        return cls(
            max_results=int(config.get("max_results", 5)),
            search_depth=str(
                config.get("search_depth", "advanced")
            ),
            topic=str(config.get("topic", "general")),
            maximum_content_characters=int(
                config.get(
                    "maximum_content_characters",
                    4000,
                )
            ),
            client=client,
        )

    def get_client(self) -> Any | None:
        """Create Tavily lazily so disabled workflows need no API key."""

        if self.client is not None:
            return self.client

        if not self.api_key:
            return None

        from tavily import TavilyClient

        self.client = TavilyClient(api_key=self.api_key)
        return self.client

    def search(self, query: str) -> WebSearchResult:
        """Search the web and return normalized citation-ready evidence."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Web search query cannot be empty.")

        client = self.get_client()

        if client is None:
            return WebSearchResult(
                status="unavailable",
                query=clean_query,
                error=(
                    "TAVILY_API_KEY is not configured, so web "
                    "fallback was skipped."
                ),
            )

        try:
            response = client.search(
                query=clean_query,
                search_depth=self.search_depth,
                topic=self.topic,
                max_results=self.max_results,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as error:
            return WebSearchResult(
                status="failed",
                query=clean_query,
                error=str(error),
            )

        raw_results = response.get("results", [])
        documents = convert_tavily_results(
            raw_results=raw_results,
            maximum_content_characters=(
                self.maximum_content_characters
            ),
        )

        if not documents:
            return WebSearchResult(
                status="empty",
                query=clean_query,
                error="Tavily returned no usable web evidence.",
            )

        return WebSearchResult(
            status="complete",
            query=clean_query,
            documents=documents,
        )
