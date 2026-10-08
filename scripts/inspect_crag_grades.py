import argparse
from typing import Any

from agentic_rag.config import (
    load_agentic_rag_config,
    load_corpus_config,
)
from agentic_rag.graph.crag import run_crag_assessment
from agentic_rag.graph.documents import convert_retrieval_results
from agentic_rag.llm.factory import create_llm_client
from agentic_rag.retrieval.factory import create_hybrid_retriever


def parse_arguments() -> argparse.Namespace:
    """Read the question used to inspect CRAG grading."""

    parser = argparse.ArgumentParser(description="Inspect CRAG grades for retrieved evidence.")
    parser.add_argument("question")
    return parser.parse_args()


def require_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Read one required YAML configuration section."""

    section = config.get(section_name)

    if section is None:
        raise KeyError("Missing configuration: " + section_name)

    return section


def main() -> None:
    """Retrieve chunks, grade them, and print the CRAG route."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    retrieval_config = require_section(
        project_config,
        "retrieval",
    )
    agent_config = load_agentic_rag_config()

    retriever = create_hybrid_retriever(retrieval_config)
    llm = create_llm_client(agent_config)

    try:
        candidates = retriever.search(
            query=arguments.question,
            top_k=int(agent_config.get("retrieval_top_k", 10)),
            use_reranker=True,
        )
    finally:
        retriever.close()

    documents = convert_retrieval_results(candidates)
    assessment = run_crag_assessment(
        question=arguments.question,
        documents=documents,
        llm=llm,
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
    print("CRAG EVIDENCE GRADES")
    print("=" * 80)

    for document in assessment.graded_documents:
        print()
        print("Title: " + document.title)
        print("Source ID: " + document.source_id)
        print("Grade: " + str(document.grade))
        print("Reason: " + document.grade_reason)
        print("-" * 80)

    print()
    print("CRAG route: " + assessment.route)
    print("Context status: " + assessment.context_status)
    print("Context reason: " + assessment.context_reason)

    if assessment.missing_information:
        print("Missing information: " + assessment.missing_information)


if __name__ == "__main__":
    main()
