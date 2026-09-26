from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPENALEX_ID_PATTERN = re.compile(r"W\d+", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from a JSONL file."""

    records: list[dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean_line = line.strip()

            if not clean_line:
                continue

            try:
                record = json.loads(clean_line)
            except json.JSONDecodeError as error:
                message = (
                    f"Invalid JSON in {path} on line "
                    f"{line_number}: {error}"
                )
                raise ValueError(message) from error

            records.append(record)

    return records


def extract_openalex_id(value: Any) -> str | None:
    """Extract an OpenAlex work ID such as W4389984066."""

    if value is None:
        return None

    text = str(value)
    match = OPENALEX_ID_PATTERN.search(text)

    if match is None:
        return None

    return match.group(0).upper()


def get_record_work_id(record: dict[str, Any]) -> str | None:
    """Find the OpenAlex ID in a paper or manifest record."""

    possible_keys = (
        "openalex_id",
        "work_id",
        "paper_id",
        "id",
    )

    for key in possible_keys:
        value = record.get(key)
        work_id = extract_openalex_id(value)

        if work_id is not None:
            return work_id

    return None


def build_file_index(
    directory: Path,
    file_suffix: str,
) -> tuple[dict[str, list[Path]], list[Path]]:
    """Group local files by their OpenAlex work ID."""

    file_index: dict[str, list[Path]] = {}
    unrecognized_files: list[Path] = []

    if not directory.exists():
        return file_index, unrecognized_files

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() != file_suffix.lower():
            continue

        work_id = extract_openalex_id(path.name)

        if work_id is None:
            unrecognized_files.append(path)
            continue

        if work_id not in file_index:
            file_index[work_id] = []

        file_index[work_id].append(path)

    for paths in file_index.values():
        paths.sort()

    unrecognized_files.sort()

    return file_index, unrecognized_files


def validate_pdf(
    path: Path,
    minimum_bytes: int,
) -> tuple[bool, str]:
    """Check whether a local file looks like a usable PDF."""

    if not path.exists():
        return False, "file_missing"

    if not path.is_file():
        return False, "not_a_file"

    file_size = path.stat().st_size

    if file_size < minimum_bytes:
        return False, "file_too_small"

    try:
        with path.open("rb") as file:
            signature = file.read(5)
    except OSError:
        return False, "file_cannot_be_read"

    if signature != b"%PDF-":
        return False, "invalid_pdf_signature"

    return True, "valid_pdf"


def inspect_xml(
    path: Path,
    minimum_bytes: int,
) -> tuple[bool, str | None, str]:
    """Validate XML and identify JATS or TEI format."""

    if not path.exists():
        return False, None, "file_missing"

    if not path.is_file():
        return False, None, "not_a_file"

    file_size = path.stat().st_size

    if file_size < minimum_bytes:
        return False, None, "file_too_small"

    try:
        parser = ET.iterparse(path, events=("start",))
        _, root = next(parser)
    except (ET.ParseError, OSError, StopIteration):
        return False, None, "invalid_xml"

    root_tag = root.tag

    if not isinstance(root_tag, str):
        return False, None, "invalid_root_element"

    local_name = root_tag

    if "}" in root_tag:
        local_name = root_tag.split("}", maxsplit=1)[1]

    local_name = local_name.lower()
    lower_root_tag = root_tag.lower()

    if local_name == "tei" or "tei-c.org" in lower_root_tag:
        return True, "tei", "valid_tei_xml"

    jats_root_names = (
        "article",
        "article-set",
        "pmc-articleset",
        "articles",
    )

    if local_name in jats_root_names:
        return True, "jats", "valid_jats_xml"

    return False, "unknown", f"unsupported_xml_root:{local_name}"


def choose_pdf(
    paths: list[Path],
    validation_results: dict[Path, tuple[bool, str]],
) -> tuple[Path | None, bool, str]:
    """Choose the first valid PDF available for one paper."""

    if not paths:
        return None, False, "missing"

    for path in paths:
        valid, reason = validation_results[path]

        if valid:
            return path, True, reason

    first_path = paths[0]
    valid, reason = validation_results[first_path]

    return first_path, valid, reason


def choose_xml(
    paths: list[Path],
    inspection_results: dict[
        Path,
        tuple[bool, str | None, str],
    ],
) -> tuple[Path | None, bool, str | None, str]:
    """Choose a valid JATS XML first, then a valid TEI XML."""

    if not paths:
        return None, False, None, "missing"

    preferred_types = ("jats", "tei")

    for preferred_type in preferred_types:
        for path in paths:
            valid, xml_type, reason = inspection_results[path]

            if valid and xml_type == preferred_type:
                return path, valid, xml_type, reason

    first_path = paths[0]
    valid, xml_type, reason = inspection_results[first_path]

    return first_path, valid, xml_type, reason


def count_duplicate_manifest_records(path: Path) -> int:
    """Count extra manifest records for the same OpenAlex paper."""

    if not path.exists():
        return 0

    manifest_records = read_jsonl(path)
    seen_work_ids: set[str] = set()
    duplicate_count = 0

    for record in manifest_records:
        work_id = get_record_work_id(record)

        if work_id is None:
            continue

        if work_id in seen_work_ids:
            duplicate_count += 1
            continue

        seen_work_ids.add(work_id)

    return duplicate_count


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write inventory records as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def write_json(
    path: Path,
    record: dict[str, Any],
) -> None:
    """Write the summary report as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def audit_local_corpus(
    audit_config: dict[str, Any],
) -> dict[str, Any]:
    """Audit local PDF and XML coverage for the final corpus."""

    corpus_path = Path(audit_config["corpus_path"])
    manifest_path = Path(audit_config["manifest_path"])
    pdf_directory = Path(audit_config["pdf_directory"])
    xml_directory = Path(audit_config["xml_directory"])
    inventory_output = Path(audit_config["inventory_output"])
    report_output = Path(audit_config["report_output"])

    minimum_pdf_bytes = int(
        audit_config.get("minimum_pdf_bytes", 10000)
    )
    minimum_xml_bytes = int(
        audit_config.get("minimum_xml_bytes", 1000)
    )

    corpus_records = read_jsonl(corpus_path)

    final_work_ids: set[str] = set()
    papers_without_work_id = 0

    for record in corpus_records:
        work_id = get_record_work_id(record)

        if work_id is None:
            papers_without_work_id += 1
            continue

        final_work_ids.add(work_id)

    pdf_index, unrecognized_pdfs = build_file_index(
        pdf_directory,
        ".pdf",
    )
    xml_index, unrecognized_xml = build_file_index(
        xml_directory,
        ".xml",
    )

    pdf_results: dict[Path, tuple[bool, str]] = {}
    xml_results: dict[
        Path,
        tuple[bool, str | None, str],
    ] = {}

    invalid_pdfs = 0
    invalid_xml = 0

    for work_id, paths in pdf_index.items():
        for path in paths:
            result = validate_pdf(path, minimum_pdf_bytes)
            pdf_results[path] = result

            valid, _ = result

            if work_id in final_work_ids and not valid:
                invalid_pdfs += 1

    for work_id, paths in xml_index.items():
        for path in paths:
            result = inspect_xml(path, minimum_xml_bytes)
            xml_results[path] = result

            valid, _, _ = result

            if work_id in final_work_ids and not valid:
                invalid_xml += 1

    inventory_records: list[dict[str, Any]] = []

    valid_pdf_only = 0
    valid_jats_xml_only = 0
    valid_tei_xml_only = 0
    papers_with_pdf_and_xml = 0
    missing_local_content = 0
    ready_for_processing = 0

    for record in corpus_records:
        work_id = get_record_work_id(record)

        pdf_paths: list[Path] = []
        xml_paths: list[Path] = []

        if work_id is not None:
            pdf_paths = pdf_index.get(work_id, [])
            xml_paths = xml_index.get(work_id, [])

        pdf_path, pdf_valid, pdf_reason = choose_pdf(
            pdf_paths,
            pdf_results,
        )
        xml_path, xml_valid, xml_type, xml_reason = choose_xml(
            xml_paths,
            xml_results,
        )

        paper_ready = pdf_valid or xml_valid

        if paper_ready:
            ready_for_processing += 1
        else:
            missing_local_content += 1

        if pdf_valid and xml_valid:
            papers_with_pdf_and_xml += 1
        elif pdf_valid:
            valid_pdf_only += 1
        elif xml_valid and xml_type == "jats":
            valid_jats_xml_only += 1
        elif xml_valid and xml_type == "tei":
            valid_tei_xml_only += 1

        inventory_record = {
            "id": record.get("id"),
            "openalex_id": work_id,
            "title": record.get("title"),
            "local_pdf_path": (
                str(pdf_path) if pdf_path is not None else None
            ),
            "local_pdf_valid": pdf_valid,
            "local_pdf_reason": pdf_reason,
            "local_xml_path": (
                str(xml_path) if xml_path is not None else None
            ),
            "local_xml_valid": xml_valid,
            "local_xml_type": xml_type,
            "local_xml_reason": xml_reason,
            "ready_for_processing": paper_ready,
        }

        inventory_records.append(inventory_record)

    orphan_files = len(unrecognized_pdfs)
    orphan_files += len(unrecognized_xml)

    for work_id, paths in pdf_index.items():
        if work_id not in final_work_ids:
            orphan_files += len(paths)

    for work_id, paths in xml_index.items():
        if work_id not in final_work_ids:
            orphan_files += len(paths)

    duplicate_manifest_records = count_duplicate_manifest_records(
        manifest_path
    )

    report = {
        "status": "complete",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "expected_papers": len(corpus_records),
        "unique_openalex_ids": len(final_work_ids),
        "papers_without_work_id": papers_without_work_id,
        "valid_pdf_only": valid_pdf_only,
        "valid_jats_xml_only": valid_jats_xml_only,
        "valid_tei_xml_only": valid_tei_xml_only,
        "papers_with_pdf_and_xml": papers_with_pdf_and_xml,
        "missing_local_content": missing_local_content,
        "invalid_pdfs": invalid_pdfs,
        "invalid_xml": invalid_xml,
        "orphan_files": orphan_files,
        "duplicate_manifest_records": duplicate_manifest_records,
        "ready_for_processing": ready_for_processing,
        "inventory_output": str(inventory_output),
    }

    write_jsonl(inventory_output, inventory_records)
    write_json(report_output, report)

    return report