import json
from pathlib import Path
from typing import Any

from agentic_rag.processing.chunk_pipeline import (
    chunk_corpus_file,
)
from agentic_rag.processing.corpus_audit import read_jsonl
from agentic_rag.processing.models import (
    PaperSection,
    ParsedPaper,
)


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write temporary JSONL test records."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file)
            file.write("\n")


def test_chunk_complete_corpus_file(
    tmp_path: Path,
) -> None:
    """Parsed papers should become unique document chunks."""

    section = PaperSection(
        section_id="W1001-section-1",
        heading="Methods",
        text=(
            "Dense retrieval finds semantic matches. "
            "Sparse retrieval finds lexical matches. "
            "Reciprocal rank fusion combines both lists."
        ),
        section_type="methods",
        order=1,
    )

    paper_with_content = ParsedPaper(
        paper_id="W1001",
        title="Hybrid Retrieval",
        abstract="A study of hybrid retrieval.",
        publication_year=2024,
        source_format="grobid_tei",
        source_path="W1001.tei.xml",
        sections=[section],
    )

    paper_without_content = ParsedPaper(
        paper_id="W1002",
        title="Empty Paper",
        source_format="grobid_tei",
        source_path="W1002.tei.xml",
    )

    input_path = tmp_path / "parsed.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    report_path = tmp_path / "report.json"

    input_records = [
        paper_with_content.model_dump(mode="json"),
        paper_without_content.model_dump(mode="json"),
    ]

    write_jsonl(input_path, input_records)

    report = chunk_corpus_file(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        max_tokens=20,
        overlap_tokens=5,
        include_abstract=True,
        small_chunk_threshold=5,
        progress_interval=0,
    )

    chunks = read_jsonl(output_path)

    assert report["status"] == "complete"
    assert report["input_papers"] == 2
    assert report["papers_processed"] == 2
    assert report["papers_with_chunks"] == 1
    assert report["papers_without_chunks"] == 1

    assert len(chunks) == report["total_chunks"]
    assert report["unique_chunk_ids"] == len(chunks)
    assert report["duplicate_chunk_ids"] == 0

    chunk_ids: set[str] = set()

    for chunk in chunks:
        chunk_ids.add(chunk["chunk_id"])
        assert chunk["paper_id"] == "W1001"
        assert chunk["token_count"] <= 20

    assert len(chunk_ids) == len(chunks)