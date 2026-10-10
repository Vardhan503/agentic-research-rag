import argparse
import json
from typing import Any

from agentic_rag.graph.runtime import AgenticRAGRuntime
from agentic_rag.graph.state import create_initial_state
from agentic_rag.graph.workflow import build_agentic_rag_graph


def parse_arguments() -> argparse.Namespace:
    """Read the question and optional state-output flag."""

    parser = argparse.ArgumentParser(description="Run the complete LangGraph Agentic RAG workflow.")
    parser.add_argument("question")
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Print the complete final graph state as JSON.",
    )
    return parser.parse_args()


def print_result(result: dict[str, Any]) -> None:
    """Print the final answer and its citation metadata."""

    print()
    print("=" * 80)
    print("AGENTIC RAG RESULT")
    print("=" * 80)
    print("Status: " + str(result.get("final_status", "unknown")))
    print("CRAG route: " + str(result.get("crag_route", "not_used")))
    print("Grounded: " + str(result.get("grounded", False)))
    print("Useful: " + str(result.get("useful", False)))
    print("Query rewrites: " + str(result.get("rewrite_count", 0)))
    print("Web search used: " + str(result.get("web_search_used", False)))
    print("Generations: " + str(result.get("generation_count", 0)))
    print()
    print(str(result.get("answer", "")))

    citations = result.get("citations", [])

    if citations:
        print()
        print("CITATIONS")
        print("-" * 80)

        for citation in citations:
            print("Source ID: " + str(citation.get("source_id", "")))
            print("Title: " + str(citation.get("title", "")))

            if citation.get("url"):
                print("URL: " + str(citation["url"]))

            print()


def main() -> None:
    """Build the graph, invoke it, and close local retrieval resources."""

    arguments = parse_arguments()
    runtime = AgenticRAGRuntime()
    graph = build_agentic_rag_graph(runtime)
    initial_state = create_initial_state(arguments.question)
    recursion_limit = int(runtime.agent_config.get("graph_recursion_limit", 40))

    try:
        result = graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
        )
    finally:
        runtime.close()

    print_result(result)

    if arguments.show_state:
        print()
        print("FINAL GRAPH STATE")
        print("-" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
