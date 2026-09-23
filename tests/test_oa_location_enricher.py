import json

from agentic_rag.ingestion.oa_location_enricher import (
    collect_oa_pdf_urls,
    enrich_oa_locations,
    merge_oa_location_data,
)
from agentic_rag.ingestion.openalex_client import (
    normalize_work_id,
)


class FakeOpenAlexClient:
    def __init__(self):
        self.calls = []

    def get_work(
        self,
        work_id,
        select_fields=None,
    ):
        self.calls.append(work_id)

        return {
            "id": (
                "https://openalex.org/"
                + work_id
            ),
            "doi": (
                "https://doi.org/"
                "10.1000/"
                + work_id.lower()
            ),
            "open_access": {
                "is_oa": True,
            },
            "best_oa_location": {
                "is_oa": True,
                "pdf_url": (
                    "https://repository.example/"
                    + work_id
                    + ".pdf"
                ),
            },
            "primary_location": {},
            "locations": [],
        }


def write_jsonl(records, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            json.dump(
                record,
                output_file,
            )
            output_file.write("\n")


def test_normalize_work_id():
    assert normalize_work_id(
        "https://openalex.org/W123456"
    ) == "W123456"

    assert normalize_work_id(
        "w123456"
    ) == "W123456"

    assert normalize_work_id(
        "not-an-openalex-id"
    ) == ""


def test_collect_oa_pdf_urls():
    paper = {
        "best_oa_location": {
            "is_oa": True,
            "pdf_url": (
                "https://example.org/best.pdf"
            ),
        },
        "locations": [
            {
                "is_oa": True,
                "pdf_url": (
                    "https://example.org/best.pdf"
                ),
            },
            {
                "is_oa": True,
                "pdf_url": (
                    "https://repository.org/"
                    "copy.pdf"
                ),
            },
            {
                "is_oa": False,
                "pdf_url": (
                    "https://closed.org/paper.pdf"
                ),
            },
        ],
    }

    assert collect_oa_pdf_urls(paper) == [
        "https://example.org/best.pdf",
        "https://repository.org/copy.pdf",
    ]


def test_merge_oa_location_data():
    original = {
        "id": (
            "https://openalex.org/W123"
        ),
        "title": "Original title",
        "final_corpus": {
            "source": (
                "deterministic_retained"
            )
        },
    }

    fresh = {
        "id": (
            "https://openalex.org/W123"
        ),
        "best_oa_location": {
            "is_oa": True,
            "pdf_url": (
                "https://example.org/paper.pdf"
            ),
        },
        "locations": [],
    }

    enriched = merge_oa_location_data(
        original,
        fresh,
    )

    assert enriched["title"] == (
        "Original title"
    )

    assert (
        enriched["final_corpus"]["source"]
        == "deterministic_retained"
    )

    assert enriched["oa_pdf_urls"] == [
        "https://example.org/paper.pdf"
    ]

    assert (
        enriched[
            "oa_location_enrichment"
        ]["source"]
        == "openalex_singleton"
    )


def test_enrichment_resumes(tmp_path):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "enriched.jsonl"
    )

    report_path = (
        tmp_path / "report.json"
    )

    failures_path = (
        tmp_path / "failures.jsonl"
    )

    write_jsonl(
        [
            {
                "id": (
                    "https://openalex.org/"
                    "W100"
                ),
                "title": "Paper one",
            },
            {
                "id": (
                    "https://openalex.org/"
                    "W200"
                ),
                "title": "Paper two",
            },
        ],
        input_path,
    )

    config = {
        "oa_location_enrichment": {
            "input_path": str(
                input_path
            ),
            "output_path": str(
                output_path
            ),
            "report_path": str(
                report_path
            ),
            "failures_path": str(
                failures_path
            ),
            "default_limit": 1,
            "max_retries": 1,
            "request_delay_seconds": 0,
        }
    }

    client = FakeOpenAlexClient()

    first_report = enrich_oa_locations(
        config,
        tmp_path,
        limit=1,
        client=client,
    )

    assert first_report[
        "total_enriched"
    ] == 1

    second_report = enrich_oa_locations(
        config,
        tmp_path,
        limit=1,
        client=client,
    )

    assert second_report["status"] == (
        "complete"
    )

    assert second_report[
        "total_enriched"
    ] == 2

    assert client.calls == [
        "W100",
        "W200",
    ]