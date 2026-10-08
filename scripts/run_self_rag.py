import argparse
from typing import Any

from agentic_rag.config import (
    load_agentic_rag_config,
    load_corpus_config,
)
from agentic_rag.graph.query_rewriting import (
    run_corrective_retrieval,
)
from agentic_rag.graph.self_rag import run_self_rag
from agentic_rag.graph.web_fallback import run_web_fallback
from agentic_rag.graph.web_search import TavilyWebSearch
from agentic_rag.llm.factory import create_llm_client
from agentic_rag.retrieval.factory import create_hybrid_retriever


def parse_arguments() -> argparse.Namespace:
    """Read the scientific question."""

    parser = argparse.ArgumentParser(
        description="Run corrective retrieval and Self-RAG verification."
    )
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
    """Run retrieval, fallback, generation, and answer verification."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    agent_config = load_agentic_rag_config()
    retrieval_config = require_section(
        project_config,
        "retrieval",
    )
    web_config = require_section(project_config, "web_search")

    llm = create_llm_client(agent_config)
    retriever = create_hybrid_retriever(retrieval_config)

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

    web_search = TavilyWebSearch.from_config(web_config)
    fallback_result = run_web_fallback(
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

    result = run_self_rag(
        question=arguments.question,
        documents=fallback_result.documents,
        llm=llm,
        maximum_generation_attempts=int(agent_config.get("maximum_generation_attempts", 2)),
        maximum_generation_characters_per_document=int(
            agent_config.get(
                "maximum_context_characters_per_document",
                2400,
            )
        ),
        maximum_verifier_characters_per_document=int(
            agent_config.get(
                "maximum_verifier_characters_per_document",
                1800,
            )
        ),
    )

    print()
    print("=" * 80)
    print("SELF-RAG RESULT")
    print("=" * 80)
    print("Status: " + result.status)
    print("Grounded: " + str(result.grounded))
    print("Useful: " + str(result.useful))
    print("Generation count: " + str(result.generation_count))
    print()
    print(result.answer)

    print()
    print("CITATIONS")
    print("-" * 80)

    for citation in result.citations:
        print("Source ID: " + citation.source_id)
        print("Title: " + citation.title)

        if citation.url:
            print("URL: " + citation.url)

        print()

    if result.unsupported_claims:
        print("Unsupported claims:")

        for claim in result.unsupported_claims:
            print("- " + claim)

    if result.critic_reason:
        print("Critic reason: " + result.critic_reason)


if __name__ == "__main__":
    main()
