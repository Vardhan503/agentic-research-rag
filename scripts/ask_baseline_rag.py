import argparse
import json
from typing import Any

from agentic_rag.config import (
    load_agentic_rag_config,
    load_corpus_config,
)
from agentic_rag.graph.baseline import run_baseline_rag
from agentic_rag.llm.factory import create_llm_client
from agentic_rag.retrieval.factory import create_hybrid_retriever


def parse_arguments() -> argparse.Namespace:
    """Read the research question and optional output mode."""

    parser = argparse.ArgumentParser(description="Ask a cited question over the research corpus.")

    parser.add_argument(
        "question",
        help="Question to answer from the indexed papers.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result as JSON.",
    )

    return parser.parse_args()


def get_retrieval_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Return the retrieval section from corpus.yaml."""

    retrieval_config = project_config.get("retrieval")

    if retrieval_config is None:
        raise KeyError("Missing retrieval configuration.")

    return retrieval_config


def print_readable_result(result) -> None:
    """Print the answer followed by its paper citations."""

    print()
    print("=" * 80)
    print("BASELINE CITED RAG ANSWER")
    print("=" * 80)
    print()
    print(result.answer)

    print()
    print("CITATIONS")
    print("-" * 80)

    if not result.citations:
        print("No citations were returned.")

    for citation_number, citation in enumerate(
        result.citations,
        start=1,
    ):
        print()
        print(str(citation_number) + ". " + citation.title)
        print("   Source ID: " + citation.source_id)
        print("   Paper ID: " + citation.paper_id)

        if citation.section_heading:
            print("   Section: " + citation.section_heading)

        if citation.doi:
            print("   DOI: " + citation.doi)

        if citation.url:
            print("   URL: " + citation.url)

    print()
    print("Elapsed seconds: " + str(round(result.elapsed_seconds, 2)))


def main() -> None:
    """Load the local models and execute baseline RAG."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    retrieval_config = get_retrieval_config(project_config)
    agent_config = load_agentic_rag_config()

    retriever = create_hybrid_retriever(retrieval_config)
    llm = create_llm_client(agent_config)

    try:
        result = run_baseline_rag(
            question=arguments.question,
            retriever=retriever,
            llm=llm,
            top_k=int(agent_config.get("retrieval_top_k", 10)),
            maximum_characters_per_document=int(
                agent_config.get(
                    "maximum_context_characters_per_document",
                    2400,
                )
            ),
        )
    finally:
        retriever.close()

    if arguments.json:
        print(
            json.dumps(
                result.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print_readable_result(result)


if __name__ == "__main__":
    main()
