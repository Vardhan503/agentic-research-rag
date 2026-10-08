from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from agentic_rag.processing.models import DocumentChunk
from agentic_rag.retrieval.models import RetrievalCandidate


QUERY_TOKEN_PATTERN = re.compile(
    r"\w+",
    re.UNICODE,
)


# Common question words match nearly every chunk, so OR-ing them forces
# FTS5 to score most of the index while adding almost no BM25 signal.
QUERY_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be
    because been before being below between both but by can could did do does
    doing down during each either few for from further had has have having he
    her here hers him his how i if in into is it its itself just me more most
    my no nor not now of off on once only or other our ours out over own same
    she should so some such than that the their theirs them then there these
    they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours
    explain describe compare discuss list give tell show
    """.split()
)


def build_fts_query(query: str) -> str:
    """Convert a user question into a safe FTS5 query."""

    tokens = QUERY_TOKEN_PATTERN.findall(query.lower())

    content_tokens = [
        token for token in tokens if token not in QUERY_STOPWORDS
    ]

    # A query made only of stopwords still needs some lexical terms.
    if content_tokens:
        tokens = content_tokens

    terms: list[str] = []
    seen_tokens: set[str] = set()

    for token in tokens:
        if not token:
            continue

        if token in seen_tokens:
            continue

        seen_tokens.add(token)

        safe_token = token.replace('"', '""')
        terms.append(f'"{safe_token}"')

    return " OR ".join(terms)


def rrf_sort_score(
    candidate: RetrievalCandidate,
) -> float:
    """Return the score used for RRF sorting."""

    return candidate.rrf_score


def rerank_sort_score(
    candidate: RetrievalCandidate,
) -> float:
    """Return the score used after reranking."""

    if candidate.rerank_score is None:
        return candidate.rrf_score

    return candidate.rerank_score


def reciprocal_rank_fusion(
    dense_results: list[RetrievalCandidate],
    sparse_results: list[RetrievalCandidate],
    rrf_k: int = 60,
    top_k: int = 30,
) -> list[RetrievalCandidate]:
    """Fuse dense and sparse rankings using RRF."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    candidate_index: dict[
        str,
        RetrievalCandidate,
    ] = {}

    for rank, candidate in enumerate(
        dense_results,
        start=1,
    ):
        chunk_id = candidate.chunk.chunk_id

        fused_candidate = candidate.model_copy(deep=True)
        fused_candidate.rrf_score = 1.0 / (rrf_k + rank)

        if "dense" not in fused_candidate.sources:
            fused_candidate.sources.append("dense")

        candidate_index[chunk_id] = fused_candidate

    for rank, candidate in enumerate(
        sparse_results,
        start=1,
    ):
        chunk_id = candidate.chunk.chunk_id
        rrf_score = 1.0 / (rrf_k + rank)

        existing_candidate = candidate_index.get(chunk_id)

        if existing_candidate is None:
            fused_candidate = candidate.model_copy(deep=True)
            fused_candidate.rrf_score = rrf_score

            if "sparse" not in fused_candidate.sources:
                fused_candidate.sources.append("sparse")

            candidate_index[chunk_id] = fused_candidate
            continue

        existing_candidate.rrf_score += rrf_score
        existing_candidate.sparse_score = candidate.sparse_score

        if "sparse" not in existing_candidate.sources:
            existing_candidate.sources.append("sparse")

    fused_results: list[RetrievalCandidate] = []

    for candidate in candidate_index.values():
        fused_results.append(candidate)

    fused_results.sort(
        key=rrf_sort_score,
        reverse=True,
    )

    return fused_results[:top_k]


def apply_paper_diversity(
    candidates: list[RetrievalCandidate],
    top_k: int,
    max_chunks_per_paper: int,
) -> list[RetrievalCandidate]:
    """Prevent one paper from occupying every result."""

    selected: list[RetrievalCandidate] = []
    paper_counts: dict[str, int] = {}

    for candidate in candidates:
        paper_id = candidate.chunk.paper_id
        current_count = paper_counts.get(paper_id, 0)

        if current_count >= max_chunks_per_paper:
            continue

        selected.append(candidate)
        paper_counts[paper_id] = current_count + 1

        if len(selected) >= top_k:
            break

    return selected


class HybridRetriever:
    """Dense, BM25, RRF, and cross-encoder retrieval."""

    def __init__(
        self,
        faiss_index_path: Path,
        sqlite_index_path: Path,
        embedding_model_name: str,
        reranker_model_name: str | None,
        query_prefix: str = "",
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        fusion_top_k: int = 30,
        final_top_k: int = 10,
        rrf_k: int = 60,
        reranker_batch_size: int = 16,
        max_chunks_per_paper: int = 3,
        embedding_model: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        """Load indexes and retrieval models."""

        if not faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")

        if not sqlite_index_path.exists():
            raise FileNotFoundError(f"SQLite index not found: {sqlite_index_path}")

        self.dense_index = faiss.read_index(str(faiss_index_path))

        # The retriever is cached and shared by LangGraph worker threads, so
        # the read-only connection may be used outside the thread that opened it.
        self.connection = sqlite3.connect(
            f"file:{sqlite_index_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self._connection_lock = threading.Lock()

        self.query_prefix = query_prefix
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_top_k = fusion_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.reranker_batch_size = reranker_batch_size
        self.max_chunks_per_paper = max_chunks_per_paper

        if embedding_model is None:
            from sentence_transformers import (
                SentenceTransformer,
            )

            embedding_model = SentenceTransformer(embedding_model_name)

        self.embedding_model = embedding_model

        if reranker is not None:
            self.reranker = reranker
        elif reranker_model_name is not None:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(
                reranker_model_name,
                max_length=512,
            )
        else:
            self.reranker = None

        row = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()

        sqlite_count = 0

        if row is not None:
            sqlite_count = int(row[0])

        if sqlite_count != self.dense_index.ntotal:
            self.connection.close()

            raise ValueError("FAISS and SQLite index sizes do not match.")

    def load_chunk(
        self,
        vector_id: int,
    ) -> DocumentChunk | None:
        """Load one complete chunk from SQLite."""

        with self._connection_lock:
            row = self.connection.execute(
                """
                SELECT chunk_json
                FROM chunks
                WHERE vector_id = ?
                """,
                (vector_id,),
            ).fetchone()

        if row is None:
            return None

        return DocumentChunk.model_validate_json(row[0])

    def dense_search(
        self,
        query: str,
    ) -> list[RetrievalCandidate]:
        """Retrieve semantically similar chunks."""

        query_text = f"{self.query_prefix}{query}"

        embedding = self.embedding_model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embedding = np.asarray(
            embedding,
            dtype="float32",
        )
        embedding = np.ascontiguousarray(embedding)

        search_count = min(
            self.dense_top_k,
            int(self.dense_index.ntotal),
        )

        scores, vector_ids = self.dense_index.search(
            embedding,
            search_count,
        )

        results: list[RetrievalCandidate] = []

        for position in range(search_count):
            vector_id = int(vector_ids[0][position])

            if vector_id < 0:
                continue

            chunk = self.load_chunk(vector_id)

            if chunk is None:
                continue

            candidate = RetrievalCandidate(
                chunk=chunk,
                dense_score=float(scores[0][position]),
                sources=["dense"],
            )

            results.append(candidate)

        return results

    def sparse_search(
        self,
        query: str,
    ) -> list[RetrievalCandidate]:
        """Retrieve lexical matches using SQLite BM25."""

        fts_query = build_fts_query(query)

        if not fts_query:
            return []

        with self._connection_lock:
            rows = self.connection.execute(
                """
                SELECT
                    chunks.chunk_json,
                    bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks
                    ON chunks.vector_id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (
                    fts_query,
                    self.sparse_top_k,
                ),
            ).fetchall()

        results: list[RetrievalCandidate] = []

        for chunk_json, raw_score in rows:
            chunk = DocumentChunk.model_validate_json(chunk_json)

            candidate = RetrievalCandidate(
                chunk=chunk,
                sparse_score=-float(raw_score),
                sources=["sparse"],
            )

            results.append(candidate)

        return results

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """Rerank fused results with a CrossEncoder."""

        if self.reranker is None:
            return candidates

        if not candidates:
            return candidates

        pairs: list[list[str]] = []

        for candidate in candidates:
            pairs.append(
                [
                    query,
                    candidate.chunk.embedding_text(),
                ]
            )

        scores = self.reranker.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )

        for index, candidate in enumerate(candidates):
            candidate.rerank_score = float(scores[index])

        candidates.sort(
            key=rerank_sort_score,
            reverse=True,
        )

        return candidates

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = True,
    ) -> list[RetrievalCandidate]:
        """Run the complete hybrid retrieval pipeline."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Query cannot be empty.")

        if top_k is None:
            top_k = self.final_top_k

        dense_results = self.dense_search(clean_query)
        sparse_results = self.sparse_search(clean_query)

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=self.rrf_k,
            top_k=self.fusion_top_k,
        )

        if use_reranker:
            fused_results = self.rerank(
                query=clean_query,
                candidates=fused_results,
            )

        return apply_paper_diversity(
            candidates=fused_results,
            top_k=top_k,
            max_chunks_per_paper=(self.max_chunks_per_paper),
        )

    def close(self) -> None:
        """Close the SQLite connection."""

        with self._connection_lock:
            self.connection.close()
