import argparse
import json
from pathlib import Path

from agentic_rag.config import PROJECT_ROOT
from agentic_rag.evaluation.dataset import load_evaluation_dataset
from agentic_rag.graph.runtime import AgenticRAGRuntime


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export retrieval candidates for human relevance judgments."
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/agentic_rag_evaluation.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark/retrieval_judgment_candidates.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    """Retrieve a candidate pool without pretending it is human gold data."""

    arguments = parse_arguments()
    examples = load_evaluation_dataset(
        resolve_path(arguments.benchmark),
        limit=arguments.limit,
    )
    output_path = resolve_path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = AgenticRAGRuntime()

    try:
        retriever = runtime.get_retriever()
        with output_path.open("w", encoding="utf-8") as output_file:
            for example in examples:
                if not example.requires_retrieval:
                    continue

                results = retriever.search(
                    query=example.question,
                    top_k=arguments.top_k,
                    use_reranker=True,
                )
                candidates = []
                for rank, result in enumerate(results, start=1):
                    candidates.append(
                        {
                            "rank": rank,
                            "source_id": result.chunk.chunk_id,
                            "paper_id": result.chunk.paper_id,
                            "title": result.chunk.title,
                            "section_heading": result.chunk.section_heading,
                            "text_preview": result.chunk.text[:500],
                            "relevant": None,
                        }
                    )

                record = {
                    "question_id": example.question_id,
                    "question": example.question,
                    "instructions": (
                        "Set relevant to true or false for every candidate. "
                        "Then copy relevant paper/source IDs into the benchmark."
                    ),
                    "candidates": candidates,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                print("Exported: " + example.question_id)
    finally:
        runtime.close()

    print("Output: " + str(output_path))


if __name__ == "__main__":
    main()
