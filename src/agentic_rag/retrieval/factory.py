from pathlib import Path
from typing import Any

from agentic_rag.config import PROJECT_ROOT
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.paper_dates import load_paper_dates


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def create_hybrid_retriever(
    config: dict[str, Any],
) -> HybridRetriever:
    """Build the project's HybridRetriever from YAML settings."""

    paper_dates: dict[str, str] = {}
    corpus_path = config.get("paper_metadata_path")

    if corpus_path:
        paper_dates = load_paper_dates(resolve_project_path(str(corpus_path)))

    return HybridRetriever(
        paper_dates=paper_dates,
        faiss_index_path=resolve_project_path(config["faiss_index_path"]),
        sqlite_index_path=resolve_project_path(config["sqlite_index_path"]),
        embedding_model_name=str(config["embedding_model"]),
        reranker_model_name=str(config["reranker_model"]),
        query_prefix=str(config.get("query_prefix", "")),
        dense_top_k=int(config.get("dense_top_k", 50)),
        sparse_top_k=int(config.get("sparse_top_k", 50)),
        fusion_top_k=int(config.get("fusion_top_k", 30)),
        final_top_k=int(config.get("final_top_k", 10)),
        rrf_k=int(config.get("rrf_k", 60)),
        reranker_batch_size=int(config.get("reranker_batch_size", 16)),
        max_chunks_per_paper=int(config.get("max_chunks_per_paper", 3)),
    )
