import json
from pathlib import Path
from typing import Any

from agentic_rag.processing.parsed_corpus import (
    assemble_parsed_corpus,
)


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file)
            file.write("\n")


def create_parsed_record(
    paper_id: str,
    source_format: str,
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "Research paper abstract.",
        "doi": None,
        "publication_year": 2024,
        "source_format": source_format,
        "source_path": f"{paper_id}.xml",
        "sections": [
            {
                "section_id": f"{paper_id}-section-1",
                "heading": "Introduction",
                "text": "Research paper section text.",
                "section_type": "introduction",
                "order": 1,
                "level": 1,
                "page_start": None,
                "page_end": None,
            }
        ],
        "extraction_warnings": [],
    }


def test_assemble_parsed_corpus(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    jats_path = tmp_path / "jats.jsonl"
    tei_path = tmp_path / "tei.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    report_path = tmp_path / "report.json"

    write_jsonl(
        corpus_path,
        [
            {
                "id": "https://openalex.org/W1001",
                "title": "First paper",
            },
            {
                "id": "https://openalex.org/W1002",
                "title": "Second paper",
            },
        ],
    )

    write_jsonl(
        jats_path,
        [
            create_parsed_record(
                "W1001",
                "jats_xml",
            )
        ],
    )

    write_jsonl(
        tei_path,
        [
            create_parsed_record(
                "W1002",
                "grobid_tei",
            )
        ],
    )

    report = assemble_parsed_corpus(
        corpus_path=corpus_path,
        parsed_jats_path=jats_path,
        parsed_tei_path=tei_path,
        output_path=output_path,
        report_path=report_path,
    )

    assert report["status"] == "complete"
    assert report["expected_papers"] == 2
    assert report["parsed_jats_papers"] == 1
    assert report["parsed_tei_papers"] == 1
    assert report["final_parsed_papers"] == 2
    assert report["missing_parsed_papers"] == 0