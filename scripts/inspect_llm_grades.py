import argparse

from agentic_rag.config import load_ollama_grading_config
from agentic_rag.processing.ollama_grader import read_jsonl


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
        default=20,
    )

    return parser.parse_args()


def relevance_score(paper):
    return paper.get("ollama_grade", {}).get("relevance_score", 0.0)


def print_group(heading, papers, limit, reverse):
    print("\n" + "=" * 80)
    print(heading)
    print("=" * 80)

    papers.sort(key=relevance_score, reverse=reverse)
    selected_papers = papers[:limit]

    for paper in selected_papers:
        grade = paper.get("ollama_grade", {})
        selection = paper.get("selection", {})

        print("\nTitle:")
        print(paper.get("title") or "[Missing title]")
        print("Deterministic score:")
        print(selection.get("relevance_score", "unknown"))
        print("LLM label:")
        print(grade.get("label"))
        print("LLM relevance score:")
        print(grade.get("relevance_score"))
        print("Reason:")
        print(grade.get("reason"))
        print("-" * 80)


def main():
    arguments = parse_arguments()
    grading_config = load_ollama_grading_config(arguments.config)

    accepted = read_jsonl(grading_config["accepted_output_path"])
    review = read_jsonl(grading_config["review_output_path"])
    rejected = read_jsonl(grading_config["rejected_output_path"])

    print_group(
        "OLLAMA ACCEPTED - LOWEST SCORES FIRST",
        accepted,
        arguments.limit,
        reverse=False,
    )

    print_group(
        "OLLAMA NEEDS REVIEW - HIGHEST SCORES FIRST",
        review,
        arguments.limit,
        reverse=True,
    )

    print_group(
        "OLLAMA REJECTED - HIGHEST SCORES FIRST",
        rejected,
        arguments.limit,
        reverse=True,
    )


if __name__ == "__main__":
    main()
