import argparse
from typing import Any

from agentic_rag.config import (
    load_agentic_rag_config,
    load_corpus_config,
)
from agentic_rag.graph.query_rewriting import (
    run_corrective_retrieval,
)
from agentic_rag.graph.web_fallback import run_web_fallback
from agentic_rag.graph.web_search import TavilyWebSearch
from agentic_rag.llm.ollama_client import OllamaStructuredClient
from agentic_rag.retrieval.factory import create_hybrid_retriever


def parse_arguments() -> argparse.Namespace:
    """Read the question used to test web fallback."""

    parser = argparse.ArgumentParser(description="Run corpus retrieval with Tavily fallback.")
    parser.add_argument("question")
    return parser.parse_args()


def require_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Read one required configuration section."""

    section = config.get(section_name)

    if section is None:
        raise KeyError("Missing configuration: " + section_name)

    return section


def main() -> None:
    """Run corrective retrieval and invoke web fallback when required."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    retrieval_config = require_section(
        project_config,
        "retrieval",
    )
    web_config = require_section(project_config, "web_search")
    agent_config = load_agentic_rag_config()

    retriever = create_hybrid_retriever(retrieval_config)
    llm = OllamaStructuredClient.from_config(agent_config)
    web_search = TavilyWebSearch.from_config(web_config)

    try:
        corrective_result = run_corrective_retrieval(
            question=arguments.question,
            retriever=retriever,
            llm=llm,
            top_k=int(agent_config.get("retrieval_top_k", 10)),
            maximum_rewrite_attempts=int(agent_config.get("maximum_rewrite_attempts", 2)),
            maximum_grade_characters_per_document=int(
                agent_config.get(
                    "maximum_grader_characters_per_document",
                    1600,
                )
            ),
            maximum_context_characters_per_document=int(
                agent_config.get(
                    "maximum_context_characters_per_document",
                    2000,
                )
            ),
        )
    finally:
        retriever.close()

    result = run_web_fallback(
        question=arguments.question,
        corrective_result=corrective_result,
        web_search=web_search,
        llm=llm,
        enabled=bool(web_config.get("enabled", True)),
        maximum_documents=int(web_config.get("maximum_combined_documents", 12)),
        maximum_grade_characters_per_document=int(
            agent_config.get(
                "maximum_grader_characters_per_document",
                1600,
            )
        ),
        maximum_context_characters_per_document=int(
            agent_config.get(
                "maximum_context_characters_per_document",
                2000,
            )
        ),
    )

    print()
    print("=" * 80)
    print("WEB FALLBACK RESULT")
    print("=" * 80)
    print("Corpus route: " + corrective_result.assessment.route)
    print("Web fallback used: " + str(result.used))
    print("Web status: " + result.status)
    print("Web query: " + result.query)
    print("Web results: " + str(len(result.web_documents)))
    print("Final evidence: " + str(len(result.documents)))
    print("Final CRAG route: " + result.assessment.route)

    if result.error:
        print("Fallback message: " + result.error)

    for document in result.web_documents:
        print()
        print("Source ID: " + document.source_id)
        print("Title: " + document.title)
        print("URL: " + str(document.url))


if __name__ == "__main__":
    main()
