import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agentic_rag.ingestion.openalex_client import (
    OpenAlexClient,
    normalize_work_id,
)


ENRICHMENT_FIELDS = [
    "id",
    "doi",
    "open_access",
    "best_oa_location",
    "primary_location",
    "locations",
]


def read_jsonl(path):
    records = []
    input_path = Path(path)

    if not input_path.exists():
        return records

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            records.append(
                json.loads(stripped_line)
            )

    return records


def append_jsonl(record, path):
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "a",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            record,
            output_file,
            ensure_ascii=False,
        )

        output_file.write("\n")
        output_file.flush()


def write_json(data, path):
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            ensure_ascii=False,
        )


def write_jsonl(records, path):
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            json.dump(
                record,
                output_file,
                ensure_ascii=False,
            )

            output_file.write("\n")


def resolve_project_path(
    path_value,
    project_root,
):
    path = Path(path_value)

    if not path.is_absolute():
        path = Path(project_root) / path

    return path


def paper_key(paper):
    value = (
        paper.get("id")
        or paper.get("paper_id")
    )

    return normalize_work_id(value)


def collect_oa_pdf_urls(paper):
    pdf_urls = []

    best_location = paper.get(
        "best_oa_location"
    ) or {}

    best_pdf_url = best_location.get(
        "pdf_url"
    )

    best_is_oa = best_location.get(
        "is_oa"
    )

    if best_pdf_url and best_is_oa is not False:
        pdf_urls.append(best_pdf_url)

    locations = paper.get(
        "locations"
    ) or []

    for location in locations:
        if not location.get("is_oa"):
            continue

        pdf_url = location.get("pdf_url")

        if not pdf_url:
            continue

        if pdf_url not in pdf_urls:
            pdf_urls.append(pdf_url)

    return pdf_urls


def merge_oa_location_data(
    paper,
    fresh_work,
):
    enriched_paper = dict(paper)

    for field_name in ENRICHMENT_FIELDS:
        if field_name in fresh_work:
            enriched_paper[field_name] = (
                fresh_work[field_name]
            )

    enriched_paper[
        "oa_location_enrichment"
    ] = {
        "source": "openalex_singleton",
        "enriched_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    enriched_paper[
        "oa_pdf_urls"
    ] = collect_oa_pdf_urls(
        enriched_paper
    )

    return enriched_paper


def load_processed_keys(path):
    processed_keys = set()

    for paper in read_jsonl(path):
        key = paper_key(paper)

        if key:
            processed_keys.add(key)

    return processed_keys


def create_enrichment_report(
    input_records,
    enriched_records,
    failures,
    report_path,
):
    unique_enriched = {}

    for paper in enriched_records:
        key = paper_key(paper)

        if key:
            unique_enriched[key] = paper

    with_best_oa_pdf = 0
    with_any_oa_pdf = 0

    for paper in unique_enriched.values():
        best_location = paper.get(
            "best_oa_location"
        ) or {}

        if best_location.get("pdf_url"):
            with_best_oa_pdf += 1

        pdf_urls = paper.get(
            "oa_pdf_urls"
        ) or []

        if pdf_urls:
            with_any_oa_pdf += 1

    total_input = len(input_records)
    total_enriched = len(unique_enriched)

    remaining = max(
        total_input - total_enriched,
        0,
    )

    status = "partial"

    if remaining == 0:
        status = "complete"

    report = {
        "status": status,
        "total_final_corpus": total_input,
        "total_enriched": total_enriched,
        "remaining": remaining,
        "with_best_oa_pdf": (
            with_best_oa_pdf
        ),
        "with_any_oa_pdf": (
            with_any_oa_pdf
        ),
        "without_direct_oa_pdf": (
            total_enriched
            - with_any_oa_pdf
        ),
        "failures_this_run": len(
            failures
        ),
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    write_json(
        report,
        report_path,
    )

    return report


def enrich_oa_locations(
    config,
    project_root,
    limit=None,
    client=None,
):
    enrichment_config = config[
        "oa_location_enrichment"
    ]

    if limit is None:
        limit = int(
            enrichment_config[
                "default_limit"
            ]
        )

    if limit < 1:
        raise ValueError(
            "Enrichment limit must be "
            "at least 1."
        )

    input_path = resolve_project_path(
        enrichment_config["input_path"],
        project_root,
    )

    output_path = resolve_project_path(
        enrichment_config["output_path"],
        project_root,
    )

    report_path = resolve_project_path(
        enrichment_config["report_path"],
        project_root,
    )

    failures_path = resolve_project_path(
        enrichment_config[
            "failures_path"
        ],
        project_root,
    )

    input_records = read_jsonl(
        input_path
    )

    processed_keys = load_processed_keys(
        output_path
    )

    pending_records = []

    for paper in input_records:
        key = paper_key(paper)

        if not key:
            continue

        if key in processed_keys:
            continue

        pending_records.append(paper)

    selected_records = pending_records[
        :limit
    ]

    if client is None:
        client = OpenAlexClient(
            max_retries=enrichment_config[
                "max_retries"
            ]
        )

    failures = []
    newly_enriched = 0

    for paper in selected_records:
        work_id = paper_key(paper)

        try:
            fresh_work = client.get_work(
                work_id,
                select_fields=(
                    ENRICHMENT_FIELDS
                ),
            )

            enriched_paper = (
                merge_oa_location_data(
                    paper,
                    fresh_work,
                )
            )

            append_jsonl(
                enriched_paper,
                output_path,
            )

            processed_keys.add(work_id)
            newly_enriched += 1

            print(
                "Enriched:",
                work_id,
                "| New:",
                newly_enriched,
                "/",
                len(selected_records),
            )

        except Exception as error:
            failure = {
                "work_id": work_id,
                "title": paper.get(
                    "title"
                ),
                "error": str(error),
            }

            failures.append(failure)

            print(
                "Failed:",
                work_id,
                "|",
                str(error),
            )

        time.sleep(
            enrichment_config[
                "request_delay_seconds"
            ]
        )

    write_jsonl(
        failures,
        failures_path,
    )

    enriched_records = read_jsonl(
        output_path
    )

    return create_enrichment_report(
        input_records,
        enriched_records,
        failures,
        report_path,
    )