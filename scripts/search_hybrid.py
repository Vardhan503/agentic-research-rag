import argparse
from pathlib import Path
from typing import Any

from agentic_rag.config import load_corpus_config
from agentic_rag.retrieval.hybrid import HybridRetriever


def parse_arguments() -> argparse.Namespace:
    """Read the search query and options."""

    parser = argparse.ArgumentParser(description="Search the research corpus.")

    parser.add_argument(
        "query",
        help="Research question or search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of final results.",
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking.",
    )

    return parser.parse_args()


def get_retrieval_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Read retrieval configuration."""

    config = project_config.get("retrieval")

    if config is None:
        raise KeyError("Missing retrieval configuration.")

    return config


def main() -> None:
    """Run and display hybrid retrieval."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    config = get_retrieval_config(project_config)

    retriever = HybridRetriever(
        faiss_index_path=Path(config["faiss_index_path"]),
        sqlite_index_path=Path(config["sqlite_index_path"]),
        embedding_model_name=config["embedding_model"],
        reranker_model_name=config["reranker_model"],
        query_prefix=config.get(
            "query_prefix",
            "",
        ),
        dense_top_k=int(config["dense_top_k"]),
        sparse_top_k=int(config["sparse_top_k"]),
        fusion_top_k=int(config["fusion_top_k"]),
        final_top_k=int(config["final_top_k"]),
        rrf_k=int(config["rrf_k"]),
        reranker_batch_size=int(config["reranker_batch_size"]),
        max_chunks_per_paper=int(config["max_chunks_per_paper"]),
    )

    try:
        results = retriever.search(
            query=arguments.query,
            top_k=arguments.top_k,
            use_reranker=not arguments.no_rerank,
        )
    finally:
        retriever.close()

    print()
    print("=" * 80)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = chunk.text[:700]

        print()
        print(f"Rank: {rank}")
        print(f"Title: {chunk.title}")
        print(f"Paper ID: {chunk.paper_id}")
        print(f"Section: {chunk.section_heading}")
        print(f"Sources: {result.sources}")
        print(f"Dense score: {result.dense_score}")
        print(f"Sparse score: {result.sparse_score}")
        print(f"RRF score: {result.rrf_score}")
        print(f"Rerank score: {result.rerank_score}")
        print()
        print(preview)
        print("-" * 80)


if __name__ == "__main__":
    main()
