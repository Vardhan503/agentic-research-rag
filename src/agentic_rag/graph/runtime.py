from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_rag.config import load_corpus_config


class AgenticRAGRuntime:
    """Lazily create services used by the LangGraph workflow."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        project_config: dict[str, Any] | None = None,
        llm: Any | None = None,
        retriever: Any | None = None,
        web_search: Any | None = None,
    ) -> None:
        self.config_path = config_path
        self._project_config = project_config
        self._llm = llm
        self._retriever = retriever
        self._web_search = web_search
        self._owns_retriever = retriever is None

    @property
    def project_config(self) -> dict[str, Any]:
        """Load the project configuration only when first requested."""

        if self._project_config is None:
            self._project_config = load_corpus_config(self.config_path)

        return self._project_config

    def require_section(
        self,
        section_name: str,
    ) -> dict[str, Any]:
        """Return one required YAML section."""

        section = self.project_config.get(section_name)

        if not isinstance(section, dict):
            raise KeyError("Missing configuration section: " + section_name)

        return section

    @property
    def agent_config(self) -> dict[str, Any]:
        return self.require_section("agentic_rag")

    @property
    def retrieval_config(self) -> dict[str, Any]:
        return self.require_section("retrieval")

    @property
    def web_config(self) -> dict[str, Any]:
        return self.require_section("web_search")

    def get_llm(self) -> Any:
        """Create the configured LLM client only when an LLM node runs."""

        if self._llm is None:
            from agentic_rag.llm.factory import create_llm_client

            self._llm = create_llm_client(self.agent_config)

        return self._llm

    def get_retriever(self) -> Any:
        """Load FAISS, SQLite, embeddings, and reranker on demand."""

        if self._retriever is None:
            from agentic_rag.retrieval.factory import (
                create_hybrid_retriever,
            )

            self._retriever = create_hybrid_retriever(self.retrieval_config)

        return self._retriever

    def get_web_search(self) -> Any:
        """Create Tavily search only when the web branch is used."""

        if self._web_search is None:
            from agentic_rag.graph.web_search import TavilyWebSearch

            self._web_search = TavilyWebSearch.from_config(self.web_config)

        return self._web_search

    def close(self) -> None:
        """Close the lazily created retriever database connection."""

        if self._retriever is None:
            return

        if not self._owns_retriever:
            return

        close_method = getattr(self._retriever, "close", None)

        if callable(close_method):
            close_method()

        self._retriever = None
