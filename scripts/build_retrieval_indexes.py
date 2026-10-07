import argparse
import json
from pathlib import Path
from typing import Any

from agentic_rag.config import load_corpus_config
from agentic_rag.retrieval.index_builder import (
    build_retrieval_indexes,
)


def parse_arguments() -> argparse.Namespace:
    """Read the optional validation limit."""

    parser = argparse.ArgumentParser(description=("Build dense and BM25 retrieval indexes."))

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Build a temporary-sized index using only the first N chunks."),
    )

    return parser.parse_args()


def get_retrieval_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Read the retrieval configuration."""

    config = project_config.get("retrieval")

    if config is None:
        raise KeyError("Missing retrieval section in configs/corpus.yaml.")

    return config


def main() -> None:
    """Build both retrieval indexes."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    config = get_retrieval_config(project_config)

    report = build_retrieval_indexes(
        chunks_path=Path(config["chunks_path"]),
        faiss_index_path=Path(config["faiss_index_path"]),
        sqlite_index_path=Path(config["sqlite_index_path"]),
        report_path=Path(config["build_report_path"]),
        embedding_model_name=config["embedding_model"],
        embedding_batch_size=int(config["embedding_batch_size"]),
        minimum_index_tokens=int(config["minimum_index_tokens"]),
        limit=arguments.limit,
    )

    print()
    print("=" * 80)
    print("RETRIEVAL INDEX REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print()


if __name__ == "__main__":
    main()
