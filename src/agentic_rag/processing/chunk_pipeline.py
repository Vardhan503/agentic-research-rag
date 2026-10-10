from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_rag.processing.chunking import chunk_paper
from agentic_rag.processing.corpus_audit import (
    read_jsonl,
    write_json,
)
from agentic_rag.processing.models import ParsedPaper


def validate_chunking_settings(
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    """Validate chunk size and overlap settings."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")


def increment_count(
    counts: dict[str, int],
    name: str,
) -> None:
    """Increase one report counter."""

    current_count = counts.get(name, 0)
    counts[name] = current_count + 1


def chunk_corpus_file(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    max_tokens: int,
    overlap_tokens: int,
    include_abstract: bool,
    small_chunk_threshold: int = 40,
    progress_interval: int = 100,
) -> dict[str, Any]:
    """Convert a parsed paper corpus into searchable chunks."""

    validate_chunking_settings(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    paper_records = read_jsonl(input_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = output_path.with_suffix(f"{output_path.suffix}.tmp")

    papers_processed = 0
    papers_with_chunks = 0
    papers_without_chunks = 0

    total_chunks = 0
    abstract_chunks = 0
    body_chunks = 0
    small_chunks = 0

    total_token_count = 0
    minimum_token_count: int | None = None
    maximum_token_count = 0

    chunk_ids: set[str] = set()
    duplicate_chunk_ids = 0

    section_type_distribution: dict[str, int] = {}
    source_format_distribution: dict[str, int] = {}

    with temporary_output.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in paper_records:
            paper = ParsedPaper.model_validate(record)

            chunks = chunk_paper(
                paper=paper,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                include_abstract=include_abstract,
            )

            papers_processed += 1

            if chunks:
                papers_with_chunks += 1
            else:
                papers_without_chunks += 1

            for chunk in chunks:
                if chunk.chunk_id in chunk_ids:
                    duplicate_chunk_ids += 1

                    raise ValueError(f"Duplicate chunk ID detected: {chunk.chunk_id}")

                chunk_ids.add(chunk.chunk_id)

                output_record = chunk.model_dump(mode="json")

                json.dump(
                    output_record,
                    output_file,
                    ensure_ascii=False,
                )
                output_file.write("\n")

                total_chunks += 1
                total_token_count += chunk.token_count

                if minimum_token_count is None or chunk.token_count < minimum_token_count:
                    minimum_token_count = chunk.token_count

                maximum_token_count = max(maximum_token_count, chunk.token_count)

                if chunk.token_count < small_chunk_threshold:
                    small_chunks += 1

                if chunk.section_type == "abstract":
                    abstract_chunks += 1
                else:
                    body_chunks += 1

                increment_count(
                    section_type_distribution,
                    chunk.section_type,
                )
                increment_count(
                    source_format_distribution,
                    chunk.source_format,
                )

            if progress_interval > 0 and papers_processed % progress_interval == 0:
                print(
                    f"Processed papers: "
                    f"{papers_processed} / "
                    f"{len(paper_records)} | "
                    f"Chunks: {total_chunks}"
                )

    temporary_output.replace(output_path)

    average_token_count = 0.0

    if total_chunks > 0:
        average_token_count = total_token_count / total_chunks

    if minimum_token_count is None:
        minimum_token_count = 0

    report = {
        "status": "complete",
        "chunked_at": datetime.now(UTC).isoformat(),
        "input_papers": len(paper_records),
        "papers_processed": papers_processed,
        "papers_with_chunks": papers_with_chunks,
        "papers_without_chunks": papers_without_chunks,
        "total_chunks": total_chunks,
        "unique_chunk_ids": len(chunk_ids),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "abstract_chunks": abstract_chunks,
        "body_chunks": body_chunks,
        "max_tokens_setting": max_tokens,
        "overlap_tokens_setting": overlap_tokens,
        "minimum_chunk_tokens": minimum_token_count,
        "maximum_chunk_tokens": maximum_token_count,
        "average_chunk_tokens": average_token_count,
        "small_chunk_threshold": small_chunk_threshold,
        "chunks_below_small_threshold": small_chunks,
        "section_type_distribution": (section_type_distribution),
        "source_format_distribution": (source_format_distribution),
        "output_path": str(output_path),
    }

    write_json(report_path, report)

    return report
