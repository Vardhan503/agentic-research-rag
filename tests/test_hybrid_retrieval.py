import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from agentic_rag.processing.models import DocumentChunk
from agentic_rag.retrieval.hybrid import (
    HybridRetriever,
    apply_paper_diversity,
    build_fts_query,
    reciprocal_rank_fusion,
)
from agentic_rag.retrieval.index_builder import (
    build_retrieval_indexes,
)
from agentic_rag.retrieval.models import (
    RetrievalCandidate,
)
from agentic_rag.retrieval.paper_dates import load_paper_dates


class FakeEmbeddingModel:
    """Create deterministic vectors without downloading a model."""

    def encode(
        self,
        texts: list[str],
        **_kwargs: Any,
    ) -> np.ndarray:
        vectors: list[list[float]] = []

        for text in texts:
            lower_text = text.lower()

            dense_value = 0.1
            sparse_value = 0.1

            if "dense" in lower_text:
                dense_value = 1.0

            if "sparse" in lower_text:
                sparse_value = 1.0

            vector = np.array(
                [
                    dense_value,
                    sparse_value,
                    0.1,
                ],
                dtype="float32",
            )

            vector = vector / np.linalg.norm(vector)
            vectors.append(vector.tolist())

        return np.asarray(
            vectors,
            dtype="float32",
        )


def create_chunk(
    chunk_id: str,
    paper_id: str,
    text: str,
    token_count: int = 10,
) -> DocumentChunk:
    """Create one valid retrieval chunk."""

    return DocumentChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        publication_year=2024,
        section_id=f"{paper_id}-section-1",
        section_heading="Methods",
        section_type="methods",
        section_order=1,
        chunk_index=0,
        text=text,
        token_count=token_count,
        character_count=len(text),
        source_format="grobid_tei",
        source_path=f"{paper_id}.tei.xml",
    )


def write_chunks(
    path: Path,
    chunks: list[DocumentChunk],
) -> None:
    """Write chunks as JSONL."""

    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            json.dump(
                chunk.model_dump(mode="json"),
                file,
            )
            file.write("\n")


def test_build_fts_query_removes_query_syntax() -> None:
    query = build_fts_query('dense-retrieval: "RAG" dense')

    assert query == ('"dense" OR "retrieval" OR "rag"')


def test_reciprocal_rank_fusion_rewards_overlap() -> None:
    shared_chunk = create_chunk(
        "shared",
        "W1001",
        "Hybrid dense and sparse retrieval.",
    )
    dense_only_chunk = create_chunk(
        "dense-only",
        "W1002",
        "Dense semantic retrieval.",
    )
    sparse_only_chunk = create_chunk(
        "sparse-only",
        "W1003",
        "Sparse lexical retrieval.",
    )

    dense_results = [
        RetrievalCandidate(
            chunk=shared_chunk,
            dense_score=0.9,
            sources=["dense"],
        ),
        RetrievalCandidate(
            chunk=dense_only_chunk,
            dense_score=0.8,
            sources=["dense"],
        ),
    ]

    sparse_results = [
        RetrievalCandidate(
            chunk=shared_chunk,
            sparse_score=2.0,
            sources=["sparse"],
        ),
        RetrievalCandidate(
            chunk=sparse_only_chunk,
            sparse_score=1.5,
            sources=["sparse"],
        ),
    ]

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=60,
        top_k=3,
    )

    assert fused[0].chunk.chunk_id == "shared"
    assert "dense" in fused[0].sources
    assert "sparse" in fused[0].sources


def test_paper_diversity_limits_repetition() -> None:
    candidates: list[RetrievalCandidate] = []

    for index in range(5):
        chunk = create_chunk(
            chunk_id=f"same-{index}",
            paper_id="W1001",
            text=f"Repeated paper chunk {index}.",
        )

        candidates.append(
            RetrievalCandidate(
                chunk=chunk,
                rrf_score=1.0 - (index * 0.1),
            )
        )

    different_chunk = create_chunk(
        chunk_id="different",
        paper_id="W1002",
        text="Different paper.",
    )

    candidates.append(
        RetrievalCandidate(
            chunk=different_chunk,
            rrf_score=0.4,
        )
    )

    selected = apply_paper_diversity(
        candidates=candidates,
        top_k=4,
        max_chunks_per_paper=2,
    )

    same_paper_count = 0

    for candidate in selected:
        if candidate.chunk.paper_id == "W1001":
            same_paper_count += 1

    assert same_paper_count == 2
    assert selected[-1].chunk.paper_id == "W1002"


def test_build_and_search_small_index(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    faiss_path = tmp_path / "dense.faiss"
    sqlite_path = tmp_path / "retrieval.sqlite3"
    report_path = tmp_path / "report.json"

    chunks = [
        create_chunk(
            "dense-chunk",
            "W1001",
            "Dense retrieval uses vector embeddings.",
        ),
        create_chunk(
            "sparse-chunk",
            "W1002",
            "Sparse retrieval uses lexical terms.",
        ),
        create_chunk(
            "tiny-chunk",
            "W1003",
            "x",
            token_count=1,
        ),
    ]

    write_chunks(chunks_path, chunks)

    report = build_retrieval_indexes(
        chunks_path=chunks_path,
        faiss_index_path=faiss_path,
        sqlite_index_path=sqlite_path,
        report_path=report_path,
        embedding_model_name="fake-model",
        embedding_batch_size=2,
        minimum_index_tokens=5,
        embedding_model=FakeEmbeddingModel(),
    )

    assert report["input_chunks"] == 3
    assert report["indexed_chunks"] == 2
    assert report["filtered_small_chunks"] == 1

    index = faiss.read_index(str(faiss_path))
    assert index.ntotal == 2

    connection = sqlite3.connect(sqlite_path)
    row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
    connection.close()

    assert row is not None
    assert row[0] == 2

    retriever = HybridRetriever(
        faiss_index_path=faiss_path,
        sqlite_index_path=sqlite_path,
        embedding_model_name="fake-model",
        reranker_model_name=None,
        query_prefix="",
        dense_top_k=2,
        sparse_top_k=2,
        fusion_top_k=2,
        final_top_k=2,
        max_chunks_per_paper=2,
        embedding_model=FakeEmbeddingModel(),
        paper_dates={"W1001": "2026-10-03"},
    )

    try:
        results = retriever.search(
            query="dense retrieval",
            use_reranker=False,
        )

        # LangGraph runs nodes in worker threads while the retriever is
        # cached, so searches must work from a thread that did not open it.
        with ThreadPoolExecutor(max_workers=2) as executor:
            threaded_results = list(
                executor.map(
                    lambda query: retriever.search(
                        query=query,
                        use_reranker=False,
                    ),
                    ["dense retrieval", "sparse lexical terms"],
                )
            )
    finally:
        retriever.close()

    assert results
    assert results[0].chunk.chunk_id == "dense-chunk"
    assert all(threaded_results)

    # The index stores only the year; the full date comes from paper metadata.
    by_paper = {r.chunk.paper_id: r.chunk.publication_date for r in results}
    assert by_paper["W1001"] == "2026-10-03"
    assert by_paper.get("W1002") is None


def test_load_paper_dates_reads_iso_dates_only(tmp_path: Path) -> None:
    corpus_path = tmp_path / "final_corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "https://openalex.org/W1", "publication_date": "2026-10-03"}),
                json.dumps({"id": "W2", "publication_date": "2025"}),
                json.dumps({"id": "W3", "publication_date": None}),
            ]
        )
        + "\n"
    )

    assert load_paper_dates(corpus_path) == {"W1": "2026-10-03"}
    assert load_paper_dates(tmp_path / "missing.jsonl") == {}
