import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.config import load_corpus_config
from agentic_rag.processing.corpus_audit import (
    read_jsonl,
    write_json,
    write_jsonl,
)
from agentic_rag.processing.grobid_client import (
    GrobidClient,
    GrobidClientError,
    validate_tei_file,
)


def parse_arguments() -> argparse.Namespace:
    """Read optional command-line settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert local research PDFs into GROBID TEI XML."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum new PDFs to process in this run.",
    )

    return parser.parse_args()


def get_grobid_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Read and validate the GROBID configuration."""

    extraction_config = project_config.get("text_extraction")

    if extraction_config is None:
        raise KeyError(
            "Missing text_extraction configuration."
        )

    grobid_config = extraction_config.get("grobid")

    if grobid_config is None:
        raise KeyError(
            "Missing text_extraction.grobid configuration."
        )

    return grobid_config


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    """Append one checkpoint record immediately."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")


def load_completed_ids(path: Path) -> set[str]:
    """Load successfully completed paper IDs."""

    completed_ids: set[str] = set()

    if not path.exists():
        return completed_ids

    records = read_jsonl(path)

    for record in records:
        work_id = record.get("openalex_id")

        if work_id:
            completed_ids.add(work_id)

    return completed_ids


def load_failure_map(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Load the latest failure for each paper."""

    failures: dict[str, dict[str, Any]] = {}

    if not path.exists():
        return failures

    records = read_jsonl(path)

    for record in records:
        work_id = record.get("openalex_id")

        if work_id:
            failures[work_id] = record

    return failures


def select_pdf_records(
    inventory_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select papers that have valid local PDFs."""

    selected_records: list[dict[str, Any]] = []

    for record in inventory_records:
        pdf_valid = record.get("local_pdf_valid")
        pdf_path = record.get("local_pdf_path")

        if pdf_valid is not True:
            continue

        if not pdf_path:
            continue

        selected_records.append(record)

    return selected_records


def main() -> None:
    """Convert all unprocessed local PDFs into TEI XML."""

    arguments = parse_arguments()
    project_config = load_corpus_config()
    grobid_config = get_grobid_config(project_config)

    inventory_path = Path(
        grobid_config["inventory_path"]
    )
    output_directory = Path(
        grobid_config["output_directory"]
    )
    manifest_path = Path(
        grobid_config["manifest_path"]
    )
    failures_path = Path(
        grobid_config["failures_path"]
    )
    report_path = Path(
        grobid_config["report_path"]
    )

    inventory_records = read_jsonl(inventory_path)
    pdf_records = select_pdf_records(inventory_records)

    completed_ids = load_completed_ids(manifest_path)
    failure_map = load_failure_map(failures_path)

    client = GrobidClient(
        base_url=grobid_config["base_url"],
        timeout_seconds=int(
            grobid_config.get("timeout_seconds", 180)
        ),
        max_retries=int(
            grobid_config.get("max_retries", 3)
        ),
        retry_delay_seconds=float(
            grobid_config.get(
                "retry_delay_seconds",
                5,
            )
        ),
        minimum_tei_bytes=int(
            grobid_config.get(
                "minimum_tei_bytes",
                1000,
            )
        ),
        consolidate_header=bool(
            grobid_config.get(
                "consolidate_header",
                False,
            )
        ),
        consolidate_citations=bool(
            grobid_config.get(
                "consolidate_citations",
                False,
            )
        ),
    )

    request_delay = float(
        grobid_config.get(
            "request_delay_seconds",
            0.25,
        )
    )
    minimum_tei_bytes = int(
        grobid_config.get("minimum_tei_bytes", 1000)
    )

    grobid_version = client.version()

    print()
    print("=" * 80)
    print("GROBID PDF PROCESSING")
    print("=" * 80)
    print()
    print(f"GROBID version: {grobid_version}")
    print(f"PDF candidates: {len(pdf_records)}")
    print(f"Previously completed: {len(completed_ids)}")
    print(f"Limit for this run: {arguments.limit}")
    print()

    attempted_this_run = 0
    processed_this_run = 0
    failed_this_run = 0
    recovered_existing = 0

    for record in pdf_records:
        work_id = record.get("openalex_id")
        pdf_path_value = record.get("local_pdf_path")

        if not work_id or not pdf_path_value:
            continue

        if work_id in completed_ids:
            continue

        if (
            arguments.limit is not None
            and attempted_this_run >= arguments.limit
        ):
            break

        pdf_path = Path(pdf_path_value)
        tei_path = output_directory / f"{work_id}.tei.xml"

        existing_valid, existing_reason = validate_tei_file(
            path=tei_path,
            minimum_bytes=minimum_tei_bytes,
        )

        if existing_valid:
            manifest_record = {
                "openalex_id": work_id,
                "pdf_path": str(pdf_path),
                "tei_path": str(tei_path),
                "status": "recovered_existing",
                "validation": existing_reason,
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            append_jsonl(manifest_path, manifest_record)
            completed_ids.add(work_id)
            failure_map.pop(work_id, None)
            recovered_existing += 1
            continue

        attempted_this_run += 1

        try:
            result = client.process_pdf(
                pdf_path=pdf_path,
                output_path=tei_path,
            )

            manifest_record = {
                "openalex_id": work_id,
                "pdf_path": str(pdf_path),
                "tei_path": str(tei_path),
                "status": "success",
                "tei_bytes": result["tei_bytes"],
                "attempts": result["attempts"],
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            append_jsonl(manifest_path, manifest_record)

            completed_ids.add(work_id)
            failure_map.pop(work_id, None)
            processed_this_run += 1

            print(
                f"Processed: {work_id} | "
                f"New: {processed_this_run} | "
                f"Attempted: {attempted_this_run}"
            )

        except GrobidClientError as error:
            failure_record = {
                "openalex_id": work_id,
                "pdf_path": str(pdf_path),
                "tei_path": str(tei_path),
                "status": "failed",
                "error": str(error),
                "failed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            failure_map[work_id] = failure_record
            failed_this_run += 1

            print(f"Failed: {work_id} | {error}")

        if request_delay > 0:
            time.sleep(request_delay)

    failure_records: list[dict[str, Any]] = []

    for failure_record in failure_map.values():
        failure_records.append(failure_record)

    write_jsonl(failures_path, failure_records)

    completed_candidates = 0

    for record in pdf_records:
        work_id = record.get("openalex_id")

        if work_id in completed_ids:
            completed_candidates += 1

    remaining = len(pdf_records) - completed_candidates

    status = "partial"

    if remaining == 0:
        status = "complete"

    report = {
        "status": status,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "grobid_version": grobid_version,
        "total_pdf_candidates": len(pdf_records),
        "completed_pdf_papers": completed_candidates,
        "remaining_pdf_papers": remaining,
        "attempted_this_run": attempted_this_run,
        "processed_this_run": processed_this_run,
        "failed_this_run": failed_this_run,
        "recovered_existing": recovered_existing,
        "current_failures": len(failure_records),
        "output_directory": str(output_directory),
    }

    write_json(report_path, report)

    print()
    print("=" * 80)
    print("GROBID PROCESSING REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print()


if __name__ == "__main__":
    main()