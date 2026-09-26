import json
from pathlib import Path
from typing import Any

from agentic_rag.processing.corpus_audit import (
    audit_local_corpus,
    extract_openalex_id,
    inspect_xml,
    read_jsonl,
    validate_pdf,
)


def write_test_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Create a JSONL file for a test."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file)
            file.write("\n")


def create_test_pdf(path: Path) -> None:
    """Create a small file with a valid PDF signature."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\nTest PDF content")


def test_extract_openalex_id() -> None:
    """OpenAlex IDs should be extracted from URLs and filenames."""

    url_id = extract_openalex_id(
        "https://openalex.org/W4389984066"
    )
    filename_id = extract_openalex_id("W4406840433.pdf")
    missing_id = extract_openalex_id("paper-without-an-id.pdf")

    assert url_id == "W4389984066"
    assert filename_id == "W4406840433"
    assert missing_id is None


def test_validate_pdf_accepts_pdf_signature(
    tmp_path: Path,
) -> None:
    """A file beginning with %PDF- should be accepted."""

    pdf_path = tmp_path / "W1001.pdf"
    create_test_pdf(pdf_path)

    valid, reason = validate_pdf(
        pdf_path,
        minimum_bytes=10,
    )

    assert valid is True
    assert reason == "valid_pdf"


def test_validate_pdf_rejects_html_file(
    tmp_path: Path,
) -> None:
    """An HTML error page must not be accepted as a PDF."""

    pdf_path = tmp_path / "W1002.pdf"
    pdf_path.write_text(
        "<html>This is an error page</html>",
        encoding="utf-8",
    )

    valid, reason = validate_pdf(
        pdf_path,
        minimum_bytes=10,
    )

    assert valid is False
    assert reason == "invalid_pdf_signature"


def test_inspect_xml_identifies_jats_and_tei(
    tmp_path: Path,
) -> None:
    """The XML inspector should distinguish JATS from TEI."""

    jats_path = tmp_path / "W2001.xml"
    tei_path = tmp_path / "W2002.xml"

    jats_path.write_text(
        "<article><body><p>JATS text</p></body></article>",
        encoding="utf-8",
    )
    tei_path.write_text(
        (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text><body><p>TEI text</p></body></text>"
            "</TEI>"
        ),
        encoding="utf-8",
    )

    jats_valid, jats_type, jats_reason = inspect_xml(
        jats_path,
        minimum_bytes=10,
    )
    tei_valid, tei_type, tei_reason = inspect_xml(
        tei_path,
        minimum_bytes=10,
    )

    assert jats_valid is True
    assert jats_type == "jats"
    assert jats_reason == "valid_jats_xml"

    assert tei_valid is True
    assert tei_type == "tei"
    assert tei_reason == "valid_tei_xml"


def test_audit_local_corpus_creates_correct_report(
    tmp_path: Path,
) -> None:
    """The complete audit should count each content category."""

    corpus_path = tmp_path / "final_corpus.jsonl"
    manifest_path = tmp_path / "download_manifest.jsonl"

    pdf_directory = tmp_path / "pdf"
    xml_directory = tmp_path / "xml"

    inventory_output = tmp_path / "inventory.jsonl"
    report_output = tmp_path / "report.json"

    corpus_records = [
        {
            "id": "https://openalex.org/W3001",
            "title": "PDF only paper",
        },
        {
            "id": "https://openalex.org/W3002",
            "title": "JATS only paper",
        },
        {
            "id": "https://openalex.org/W3003",
            "title": "PDF and TEI paper",
        },
        {
            "id": "https://openalex.org/W3004",
            "title": "Invalid PDF paper",
        },
    ]

    manifest_records = [
        {"openalex_id": "W3001"},
        {"openalex_id": "W3001"},
        {"openalex_id": "W3002"},
    ]

    write_test_jsonl(corpus_path, corpus_records)
    write_test_jsonl(manifest_path, manifest_records)

    create_test_pdf(pdf_directory / "W3001.pdf")
    create_test_pdf(pdf_directory / "W3003.pdf")
    create_test_pdf(pdf_directory / "W9999.pdf")

    invalid_pdf_path = pdf_directory / "W3004.pdf"
    invalid_pdf_path.write_text(
        "This is not a PDF file",
        encoding="utf-8",
    )

    jats_path = xml_directory / "W3002.xml"
    jats_path.parent.mkdir(parents=True, exist_ok=True)
    jats_path.write_text(
        "<article><body><p>JATS paper</p></body></article>",
        encoding="utf-8",
    )

    tei_path = xml_directory / "W3003.xml"
    tei_path.write_text(
        (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text><body><p>TEI paper</p></body></text>"
            "</TEI>"
        ),
        encoding="utf-8",
    )

    audit_config = {
        "corpus_path": str(corpus_path),
        "manifest_path": str(manifest_path),
        "pdf_directory": str(pdf_directory),
        "xml_directory": str(xml_directory),
        "inventory_output": str(inventory_output),
        "report_output": str(report_output),
        "minimum_pdf_bytes": 10,
        "minimum_xml_bytes": 10,
    }

    report = audit_local_corpus(audit_config)

    assert report["expected_papers"] == 4
    assert report["valid_pdf_only"] == 1
    assert report["valid_jats_xml_only"] == 1
    assert report["valid_tei_xml_only"] == 0
    assert report["papers_with_pdf_and_xml"] == 1
    assert report["missing_local_content"] == 1
    assert report["invalid_pdfs"] == 1
    assert report["invalid_xml"] == 0
    assert report["orphan_files"] == 1
    assert report["duplicate_manifest_records"] == 1
    assert report["ready_for_processing"] == 3

    assert inventory_output.exists()
    assert report_output.exists()

    inventory_records = read_jsonl(inventory_output)

    assert len(inventory_records) == 4

    invalid_paper = None

    for record in inventory_records:
        if record["openalex_id"] == "W3004":
            invalid_paper = record
            break

    assert invalid_paper is not None
    assert invalid_paper["local_pdf_valid"] is False
    assert invalid_paper["ready_for_processing"] is False