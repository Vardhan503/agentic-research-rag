import argparse
from pathlib import Path

from agentic_rag.config import (
    PROJECT_ROOT,
    load_corpus_config,
)
from agentic_rag.processing.ollama_grader import (
    create_ollama_client,
    grade_paper,
    read_jsonl,
    route_grade,
)


POSITIVE_TITLE_TERMS = [
    "retrieval-augmented generation",
    "retrieval augmented generation",
    "corrective retrieval",
    "corrective rag",
    "self-rag",
    "self rag",
    "dense passage retrieval",
    "document retrieval",
    "passage retrieval",
    "cross-encoder reranking",
    "retrieval reranking",
]


def resolve_project_path(path_value):
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def title_matches(title):
    normalized_title = title.lower()

    for term in POSITIVE_TITLE_TERMS:
        if term in normalized_title:
            return True

    return False


def collect_positive_candidates(paths, limit):
    candidates = []
    seen_titles = set()

    for input_path in paths:
        papers = read_jsonl(input_path)

        for paper in papers:
            title = paper.get("title") or ""

            normalized_title = " ".join(
                title.lower().split()
            )

            if not title_matches(title):
                continue

            if normalized_title in seen_titles:
                continue

            candidate = dict(paper)

            candidate["_validation_source"] = str(
                input_path
            )

            candidates.append(candidate)
            seen_titles.add(normalized_title)

            if len(candidates) >= limit:
                return candidates

    return candidates


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Corpus configuration path. "
            "Defaults to configs/corpus.yaml."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    config = load_corpus_config(
        arguments.config
    )

    grading_config = config[
        "ollama_grading"
    ]

    selection_config = config[
        "selection"
    ]

    retained_path = resolve_project_path(
        selection_config["retained_output"]
    )

    ambiguous_path = resolve_project_path(
        selection_config["ambiguous_output"]
    )

    input_paths = [
        retained_path,
        ambiguous_path,
    ]

    print("Searching positive controls in:")

    for input_path in input_paths:
        print("-", input_path)

    candidates = collect_positive_candidates(
        input_paths,
        arguments.limit,
    )

    if not candidates:
        print(
            "\nNo positive-control candidates "
            "were found in the configured files."
        )
        return

    client = create_ollama_client(
        grading_config
    )

    accepted = 0
    needs_review = 0
    rejected = 0

    print("\n" + "=" * 80)
    print("POSITIVE-CONTROL VALIDATION")
    print("=" * 80)

    for paper in candidates:
        grade = grade_paper(
            paper,
            client,
            grading_config,
        )

        route = route_grade(
            grade,
            grading_config,
        )

        if route == "accepted":
            accepted += 1
        elif route == "needs_review":
            needs_review += 1
        else:
            rejected += 1

        print("\nTitle:")
        print(paper.get("title"))

        print("Source:")
        print(paper.get("_validation_source"))

        print("LLM label:")
        print(grade.label)

        print("Relevance score:")
        print(grade.relevance_score)

        print("Final route:")
        print(route)

        print("Reason:")
        print(grade.reason)

        print("-" * 80)

    print("\n" + "=" * 80)
    print("POSITIVE-CONTROL SUMMARY")
    print("=" * 80)
    print("Tested:", len(candidates))
    print("Accepted:", accepted)
    print("Needs review:", needs_review)
    print("Rejected:", rejected)


if __name__ == "__main__":
    main()