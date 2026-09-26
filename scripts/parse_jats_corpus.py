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
from agentic_rag.processing.jats_parser import (
    JATSParsingError,
    parse_jats_file,
)


def get_jats_config(
    project_config: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Read and validate the JATS extraction configuration."""

    extraction_config = project_config.get("text_extraction")

    if extraction_config is None:
        raise KeyError(
            "The text_extraction section is missing from "
            "configs/corpus.yaml."
        )

    jats_config = extraction_config.get("jats")

    if jats_config is None:
        raise KeyError(
            "The text_extraction.jats section is missing from "
            "configs/corpus.yaml."
        )

    required_settings = (
        "corpus_path",
        "inventory_path",
        "parsed_output",
        "failures_output",
        "report_output",
    )

    for setting in required_settings:
        if setting not in jats_config:
            raise KeyError(
                f"Missing JATS configuration setting: {setting}"
            )

    minimum_characters = int(
        extraction_config.get(
            "minimum_text_characters",
            500,
        )
    )

    return jats_config, minimum_characters


def build_corpus_index(
    corpus_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Create a lookup table from OpenAlex ID to metadata."""

    corpus_index: dict[str, dict[str, Any]] = {}

    for record in corpus_records:
        work_id = get_record_work_id(record)

        if work_id is None:
            continue

        corpus_index[work_id] = record

    return corpus_index


def select_jats_inventory_records(
    inventory_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select papers that have valid local JATS XML."""

    selected_records: list[dict[str, Any]] = []

    for record in inventory_records:
        xml_valid = record.get("local_xml_valid")
        xml_type = record.get("local_xml_type")
        xml_path = record.get("local_xml_path")

        if xml_valid is not True:
            continue

        if xml_type != "jats":
            continue

        if not xml_path:
            continue

        selected_records.append(record)

    return selected_records


def run_jats_extraction(
    jats_config: dict[str, Any],
    minimum_characters: int,
) -> dict[str, Any]:
    """Parse every valid JATS paper in the local inventory."""

    corpus_path = Path(jats_config["corpus_path"])
    inventory_path = Path(jats_config["inventory_path"])

    parsed_output = Path(jats_config["parsed_output"])
    failures_output = Path(jats_config["failures_output"])
    report_output = Path(jats_config["report_output"])

    corpus_records = read_jsonl(corpus_path)
    inventory_records = read_jsonl(inventory_path)

    corpus_index = build_corpus_index(corpus_records)

    jats_records = select_jats_inventory_records(
        inventory_records
    )

    parsed_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    parse_failures = 0
    insufficient_text = 0
    total_sections = 0

    selected_count = len(jats_records)

    print()
    print("=" * 80)
    print("JATS XML EXTRACTION")
    print("=" * 80)
    print()
    print(f"JATS papers selected: {selected_count}")
    print()

    for position, inventory_record in enumerate(
        jats_records,
        start=1,
    ):
        work_id = inventory_record.get("openalex_id")
        xml_path_value = inventory_record.get("local_xml_path")

        if not work_id or not xml_path_value:
            failure_records.append(
                {
                    "openalex_id": work_id,
                    "status": "missing_inventory_fields",
                }
            )
            parse_failures += 1
            continue

        xml_path = Path(xml_path_value)
        metadata = corpus_index.get(work_id, {})

        try:
            paper = parse_jats_file(
                xml_path=xml_path,
                paper_id=work_id,
                metadata=metadata,
            )
        except (JATSParsingError, OSError, ValueError) as error:
            failure_records.append(
                {
                    "openalex_id": work_id,
                    "xml_path": str(xml_path),
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
                    "xml_path": str(xml_path),
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

        if position % 25 == 0 or position == selected_count:
            print(
                f"Processed: {position} / {selected_count} | "
                f"Accepted: {len(parsed_records)}"
            )

    write_jsonl(parsed_output, parsed_records)
    write_jsonl(failures_output, failure_records)

    report = {
        "status": "complete",
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "selected_jats_papers": selected_count,
        "successfully_parsed": len(parsed_records),
        "parse_failures": parse_failures,
        "insufficient_text": insufficient_text,
        "total_sections": total_sections,
        "minimum_text_characters": minimum_characters,
        "parsed_output": str(parsed_output),
        "failures_output": str(failures_output),
    }

    write_json(report_output, report)

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print the JATS extraction report."""

    print()
    print("=" * 80)
    print("JATS EXTRACTION REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print()


def main() -> None:
    """Load configuration and run JATS extraction."""

    project_config = load_corpus_config()

    jats_config, minimum_characters = get_jats_config(
        project_config
    )

    report = run_jats_extraction(
        jats_config=jats_config,
        minimum_characters=minimum_characters,
    )

    print_report(report)


if __name__ == "__main__":
    main()