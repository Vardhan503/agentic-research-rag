from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from agentic_rag.processing.corpus_audit import write_json
from agentic_rag.processing.models import DocumentChunk


def create_database_schema(
    connection: sqlite3.Connection,
) -> None:
    """Create metadata storage and the FTS5 BM25 index."""

    connection.execute(
        """
        CREATE TABLE chunks (
            vector_id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            paper_id TEXT NOT NULL,
            section_type TEXT NOT NULL,
            publication_year INTEGER,
            chunk_json TEXT NOT NULL,
            searchable_text TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX chunks_paper_id_index
        ON chunks(paper_id)
        """
    )

    connection.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            searchable_text,
            content='chunks',
            content_rowid='vector_id',
            tokenize='porter unicode61'
        )
        """
    )


def create_embedding_model(
    model_name: str,
) -> Any:
    """Load the configured Sentence Transformer."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_and_store_batch(
    chunks: list[DocumentChunk],
    embedding_model: Any,
    batch_size: int,
    connection: sqlite3.Connection,
    dense_index: Any | None,
    starting_vector_id: int,
) -> tuple[Any, int]:
    """Embed one batch and add it to FAISS and SQLite."""

    embedding_texts: list[str] = []

    for chunk in chunks:
        embedding_texts.append(chunk.embedding_text())

    embeddings = embedding_model.encode(
        embedding_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )
    embeddings = np.ascontiguousarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError("The embedding model returned an invalid shape.")

    embedding_dimension = int(embeddings.shape[1])

    if dense_index is None:
        dense_index = faiss.IndexFlatIP(embedding_dimension)

    if dense_index.d != embedding_dimension:
        raise ValueError("Embedding dimension changed during indexing.")

    dense_index.add(embeddings)

    vector_id = starting_vector_id

    for chunk in chunks:
        chunk_json = json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
        )

        connection.execute(
            """
            INSERT INTO chunks (
                vector_id,
                chunk_id,
                paper_id,
                section_type,
                publication_year,
                chunk_json,
                searchable_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vector_id,
                chunk.chunk_id,
                chunk.paper_id,
                chunk.section_type,
                chunk.publication_year,
                chunk_json,
                chunk.embedding_text(),
            ),
        )

        vector_id += 1

    connection.commit()

    return dense_index, vector_id


def build_retrieval_indexes(
    chunks_path: Path,
    faiss_index_path: Path,
    sqlite_index_path: Path,
    report_path: Path,
    embedding_model_name: str,
    embedding_batch_size: int,
    minimum_index_tokens: int,
    limit: int | None = None,
    embedding_model: Any | None = None,
) -> dict[str, Any]:
    """Build dense FAISS and sparse SQLite indexes."""

    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive.")

    if minimum_index_tokens < 1:
        raise ValueError("minimum_index_tokens must be at least one.")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {chunks_path}")

    faiss_index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sqlite_index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_faiss_path = faiss_index_path.with_suffix(f"{faiss_index_path.suffix}.tmp")
    temporary_sqlite_path = sqlite_index_path.with_suffix(f"{sqlite_index_path.suffix}.tmp")

    temporary_faiss_path.unlink(missing_ok=True)
    temporary_sqlite_path.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary_sqlite_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")

    try:
        create_database_schema(connection)
    except sqlite3.OperationalError as error:
        connection.close()

        raise RuntimeError("SQLite FTS5 is unavailable in this Python installation.") from error

    if embedding_model is None:
        embedding_model = create_embedding_model(embedding_model_name)

    dense_index: Any | None = None
    pending_chunks: list[DocumentChunk] = []

    input_chunks = 0
    indexed_chunks = 0
    filtered_small_chunks = 0
    vector_id = 0

    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if limit is not None and input_chunks >= limit:
                break

            clean_line = line.strip()

            if not clean_line:
                continue

            input_chunks += 1

            try:
                record = json.loads(clean_line)
                chunk = DocumentChunk.model_validate(record)
            except (json.JSONDecodeError, ValueError) as error:
                connection.close()

                raise ValueError(f"Invalid chunk on line {line_number}.") from error

            if chunk.token_count < minimum_index_tokens:
                filtered_small_chunks += 1
                continue

            pending_chunks.append(chunk)

            if len(pending_chunks) < embedding_batch_size:
                continue

            dense_index, vector_id = embed_and_store_batch(
                chunks=pending_chunks,
                embedding_model=embedding_model,
                batch_size=embedding_batch_size,
                connection=connection,
                dense_index=dense_index,
                starting_vector_id=vector_id,
            )

            indexed_chunks += len(pending_chunks)
            pending_chunks = []

            if indexed_chunks % 5000 == 0:
                print(f"Indexed chunks: {indexed_chunks}")

    if pending_chunks:
        dense_index, vector_id = embed_and_store_batch(
            chunks=pending_chunks,
            embedding_model=embedding_model,
            batch_size=embedding_batch_size,
            connection=connection,
            dense_index=dense_index,
            starting_vector_id=vector_id,
        )

        indexed_chunks += len(pending_chunks)

    if dense_index is None or indexed_chunks == 0:
        connection.close()

        raise ValueError("No chunks were eligible for indexing.")

    connection.execute(
        """
        INSERT INTO chunks_fts(chunks_fts)
        VALUES ('rebuild')
        """
    )
    connection.execute("PRAGMA optimize")
    connection.commit()
    connection.close()

    faiss.write_index(
        dense_index,
        str(temporary_faiss_path),
    )

    temporary_faiss_path.replace(faiss_index_path)
    temporary_sqlite_path.replace(sqlite_index_path)

    status = "complete"

    if limit is not None:
        status = "test"

    report = {
        "status": status,
        "built_at": datetime.now(UTC).isoformat(),
        "input_chunks": input_chunks,
        "indexed_chunks": indexed_chunks,
        "filtered_small_chunks": filtered_small_chunks,
        "minimum_index_tokens": minimum_index_tokens,
        "embedding_model": embedding_model_name,
        "embedding_dimension": int(dense_index.d),
        "faiss_vectors": int(dense_index.ntotal),
        "faiss_index_path": str(faiss_index_path),
        "sqlite_index_path": str(sqlite_index_path),
        "requested_limit": limit,
    }

    write_json(report_path, report)

    return report
