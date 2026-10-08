from __future__ import annotations

from typing import Any


SUPPORTED_PROVIDERS = ("ollama", "openai")


def create_llm_client(
    config: dict[str, Any],
    client: Any | None = None,
) -> Any:
    """Create the structured LLM client named by agentic_rag.provider."""

    provider = str(config.get("provider", "ollama")).strip().lower()

    if provider == "openai":
        from agentic_rag.llm.openai_client import OpenAIStructuredClient

        return OpenAIStructuredClient.from_config(config, client=client)

    if provider == "ollama":
        from agentic_rag.llm.ollama_client import OllamaStructuredClient

        return OllamaStructuredClient.from_config(config, client=client)

    raise ValueError(
        "Unsupported LLM provider: "
        + provider
        + ". Expected one of: "
        + ", ".join(SUPPORTED_PROVIDERS)
    )
