import argparse
from pathlib import Path

import yaml

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


def load_grading_config(config_path):
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    return config["ollama_grading"]


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
        default="configs/corpus.yaml",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    grading_config = load_grading_config(
        Path(arguments.config)
    )

    input_paths = [
        Path("data/interim/corpus_retained.jsonl"),
        Path("data/interim/corpus_ambiguous.jsonl"),
    ]

    candidates = collect_positive_candidates(
        input_paths,
        arguments.limit,
    )

    if not candidates:
        print("No positive-control candidates found.")
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