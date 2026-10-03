import argparse
from typing import Any

from agentic_rag.config import (
    load_agentic_rag_config,
    load_corpus_config,
)
from agentic_rag.graph.query_rewriting import (
    run_corrective_retrieval,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient
from agentic_rag.retrieval.factory import create_hybrid_retriever


def parse_arguments() -> argparse.Namespace:
    """Read the question for corrective retrieval."""

    parser = argparse.ArgumentParser(
        description=(
            "Run CRAG retrieval with bounded query rewriting."
        )
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
    """Run corrective retrieval and print its decisions."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    retrieval_config = require_section(
        project_config,
        "retrieval",
    )
    agent_config = load_agentic_rag_config()

    retriever = create_hybrid_retriever(retrieval_config)
    llm = OllamaStructuredClient.from_config(agent_config)

    try:
        result = run_corrective_retrieval(
            question=arguments.question,
            retriever=retriever,
            llm=llm,
            top_k=int(agent_config.get("retrieval_top_k", 10)),
            maximum_rewrite_attempts=int(
                agent_config.get(
                    "maximum_rewrite_attempts",
                    2,
                )
            ),
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

    print()
    print("=" * 80)
    print("CORRECTIVE RETRIEVAL RESULT")
    print("=" * 80)
    print()
    print("Final route: " + result.assessment.route)
    print("Rewrite count: " + str(result.rewrite_count))
    print("Selected evidence: " + str(len(result.documents)))

    print()
    print("QUERY HISTORY")
    print("-" * 80)

    for query_number, query in enumerate(
        result.query_history,
        start=1,
    ):
        print(str(query_number) + ". " + query)

        reason_index = query_number - 2

        if reason_index >= 0:
            print(
                "   Reason: "
                + result.rewrite_reasons[reason_index]
            )

    print()
    print("Context reason: " + result.assessment.context_reason)

    if result.assessment.missing_information:
        print(
            "Remaining missing information: "
            + result.assessment.missing_information
        )


if __name__ == "__main__":
    main()
