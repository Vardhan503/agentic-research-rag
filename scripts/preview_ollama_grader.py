import argparse

from agentic_rag.config import load_ollama_grading_config
from agentic_rag.processing.ollama_grader import (
    create_ollama_client,
    grade_paper,
    read_jsonl,
    route_grade,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        help="Path to corpus YAML (default: configs/corpus.yaml).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    grading_config = load_ollama_grading_config(arguments.config)
    papers = read_jsonl(grading_config["input_path"])
    client = create_ollama_client(grading_config)

    print("\n" + "=" * 80)
    print("OLLAMA AMBIGUOUS-PAPER PREVIEW")
    print("=" * 80)

    preview_count = min(arguments.limit, len(papers))

    for paper_number in range(preview_count):
        paper = papers[paper_number]
        grade = grade_paper(paper, client, grading_config)
        route = route_grade(grade, grading_config)
        selection = paper.get("selection", {})

        print("\nTitle:")
        print(paper.get("title") or "[Missing title]")
        print("Deterministic score:")
        print(selection.get("relevance_score", "unknown"))
        print("Ollama label:")
        print(grade.label)
        print("Ollama relevance score:")
        print(grade.relevance_score)
        print("Final route:")
        print(route)
        print("Reason:")
        print(grade.reason)
        print("-" * 80)


if __name__ == "__main__":
    main()
