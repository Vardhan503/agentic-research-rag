"""Create pooled retrieval gold labels for the benchmark.

Usage:
    python scripts/judge_retrieval_candidates.py            # judge and write judgments
    python scripts/judge_retrieval_candidates.py --apply    # also write expected_paper_ids

Review data/benchmark/retrieval_judgments.jsonl before running with --apply.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_rag.config import PROJECT_ROOT
from agentic_rag.evaluation.dataset import load_evaluation_dataset
from agentic_rag.evaluation.gold_labels import (
    QuestionJudgments,
    apply_gold_labels,
    judge_candidates,
    load_judgments,
    write_judgments,
)
from agentic_rag.graph.runtime import AgenticRAGRuntime

GOLD_NOTE = (
    "expected_paper_ids pooled from hybrid top-20 and judged against the "
    "reference answer by gpt-4.1-mini, then reviewed."
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmark/agentic_rag_evaluation.jsonl")
    parser.add_argument("--output", default="data/benchmark/retrieval_judgments.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write relevant paper IDs into the benchmark's expected_paper_ids.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Skip judging and apply the judgments already saved in --output.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    arguments = parse_arguments()
    benchmark_path = resolve_path(arguments.benchmark)
    output_path = resolve_path(arguments.output)

    if arguments.reuse:
        judgments = load_judgments(output_path)
    else:
        examples = load_evaluation_dataset(benchmark_path, limit=arguments.limit)
        runtime = AgenticRAGRuntime()
        judgments: list[QuestionJudgments] = []

        try:
            retriever = runtime.get_retriever()
            llm = runtime.get_llm()

            for example in examples:
                if not example.requires_retrieval or example.reference_mode == "dynamic":
                    continue

                candidates = retriever.search(
                    query=example.question,
                    top_k=arguments.top_k,
                    use_reranker=True,
                )
                judged = judge_candidates(example, candidates, llm)
                judgments.append(judged)
                print(
                    example.question_id
                    + ": "
                    + str(len(judged.relevant_paper_ids))
                    + " relevant papers of "
                    + str(len({c.paper_id for c in judged.candidates}))
                )
                write_judgments(output_path, judgments)
        finally:
            runtime.close()

    print("Judgments: " + str(output_path))

    if arguments.apply:
        updated = apply_gold_labels(benchmark_path, judgments, GOLD_NOTE)
        print("Updated expected_paper_ids on " + str(updated) + " questions.")


if __name__ == "__main__":
    main()
