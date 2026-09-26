import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.config import load_corpus_config
from agentic_rag.processing.corpus_audit import (
    get_record_work_id,
    read_jsonl,
    write_json,
    write_jsonl,
)
from agentic_rag.processing.tei_parser import (
    TEIParsingError,
    parse_tei_file,
)


def build_corpus_index(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Create a metadata lookup using OpenAlex IDs."""

    corpus_index: dict[str, dict[str, Any]] = {}

    for record in records:
        work_id = get_record_work_id(record)

        if work_id:
            corpus_index[work_id] = record

    return corpus_index


def main() -> None:
    """Parse all available GROBID TEI files."""

    project_config = load_corpus_config()
    extraction_config = project_config["text_extraction"]
    tei_config = extraction_config["tei"]

    minimum_characters = int(
        extraction_config.get(
            "minimum_text_characters",
            500,
        )
    )

    corpus_records = read_jsonl(
        Path(tei_config["corpus_path"])
    )
    inventory_records = read_jsonl(
        Path(tei_config["inventory_path"])
    )

    corpus_index = build_corpus_index(corpus_records)

    tei_directory = Path(tei_config["tei_directory"])
    parsed_output = Path(tei_config["parsed_output"])
    failures_output = Path(
        tei_config["failures_output"]
    )
    report_output = Path(tei_config["report_output"])

    parsed_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    pdf_candidates = 0
    available_tei = 0
    missing_tei = 0
    parse_failures = 0
    insufficient_text = 0
    total_sections = 0

    for inventory_record in inventory_records:
        if inventory_record.get("local_pdf_valid") is not True:
            continue

        work_id = inventory_record.get("openalex_id")

        if not work_id:
            continue

        pdf_candidates += 1

        tei_path = tei_directory / f"{work_id}.tei.xml"

        if not tei_path.exists():
            missing_tei += 1
            continue

        available_tei += 1
        metadata = corpus_index.get(work_id, {})

        try:
            paper = parse_tei_file(
                xml_path=tei_path,
                paper_id=work_id,
                metadata=metadata,
            )
        except (TEIParsingError, OSError, ValueError) as error:
            failure_records.append(
                {
                    "openalex_id": work_id,
                    "tei_path": str(tei_path),
                    "status": "parse_failed",
                    "error": str(error),
                }
            )
            parse_failures += 1
            continue

        if not paper.has_usable_text(minimum_characters):
            failure_records.append(
                {
                    "openalex_id": work_id,
                    "tei_path": str(tei_path),
                    "status": "insufficient_text",
                    "character_count": (
                        paper.content_character_count()
                    ),
                }
            )
            insufficient_text += 1
            continue

        parsed_record = paper.model_dump(mode="json")
        parsed_record["character_count"] = (
            paper.content_character_count()
        )

        parsed_records.append(parsed_record)
        total_sections += len(paper.sections)

        parsed_count = len(parsed_records)

        if parsed_count % 100 == 0:
            print(f"Parsed TEI papers: {parsed_count}")

    write_jsonl(parsed_output, parsed_records)
    write_jsonl(failures_output, failure_records)

    status = "partial"

    if missing_tei == 0:
        status = "complete"

    report = {
        "status": status,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "pdf_candidates": pdf_candidates,
        "available_tei_files": available_tei,
        "missing_tei_files": missing_tei,
        "successfully_parsed": len(parsed_records),
        "parse_failures": parse_failures,
        "insufficient_text": insufficient_text,
        "total_sections": total_sections,
        "minimum_text_characters": minimum_characters,
        "parsed_output": str(parsed_output),
    }

    write_json(report_output, report)

    print()
    print("=" * 80)
    print("TEI PARSING REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print()


if __name__ == "__main__":
    main()