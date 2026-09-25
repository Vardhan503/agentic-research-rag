import json

import pytest

from agentic_rag.ingestion.direct_pdf_downloader import (
    classify_pdf_failure,
    download_direct_oa_pdfs,
)
from agentic_rag.ingestion.europe_pmc_fallback import (
    EuropePmcFallbackError,
    download_europe_pmc_xml,
    existing_xml_is_valid,
    normalize_doi,
)


class FakeResponse:
    def __init__(
        self,
        status_code,
        content=b"",
        url="https://example.org/response",
        json_data=None,
    ):
        self.status_code = status_code
        self.content = content
        self.url = url
        self.json_data = json_data
        self.headers = {
            "Content-Length": str(
                len(content)
            )
        }
        self.closed = False

    def json(self):
        return self.json_data

    def iter_content(self, chunk_size):
        for start in range(
            0,
            len(self.content),
            chunk_size,
        ):
            yield self.content[
                start:start + chunk_size
            ]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.calls.append(
            {
                "url": url,
                "kwargs": kwargs,
            }
        )

        return self.responses.pop(0)


def fallback_config(tmp_path):
    return {
        "enabled": True,
        "output_directory": str(
            tmp_path / "xml"
        ),
        "search_url": (
            "https://www.ebi.ac.uk/"
            "europepmc/webservices/rest/search"
        ),
        "full_text_base_url": (
            "https://www.ebi.ac.uk/"
            "europepmc/webservices/rest"
        ),
        "connect_timeout_seconds": 10,
        "read_timeout_seconds": 30,
        "max_retries": 1,
        "retry_delay_seconds": 0,
        "chunk_size_bytes": 32,
        "minimum_xml_bytes": 20,
        "maximum_xml_bytes": 10000,
    }


def europe_pmc_search_payload():
    return {
        "resultList": {
            "result": [
                {
                    "doi": "10.2196/66098",
                    "pmcid": "PMC12079058",
                    "isOpenAccess": "Y",
                    "inEPMC": "Y",
                }
            ]
        }
    }


def test_normalize_doi():
    assert normalize_doi(
        "https://doi.org/10.2196/66098"
    ) == "10.2196/66098"

    assert normalize_doi(
        "DOI:10.2196/66098"
    ) == "10.2196/66098"


def test_download_europe_pmc_xml(tmp_path):
    xml_content = (
        b"<?xml version='1.0'?>"
        b"<article><body><p>RAG text</p>"
        b"</body></article>"
    )

    session = FakeSession(
        [
            FakeResponse(
                status_code=200,
                json_data=(
                    europe_pmc_search_payload()
                ),
            ),
            FakeResponse(
                status_code=200,
                content=xml_content,
                url=(
                    "https://www.ebi.ac.uk/"
                    "europepmc/webservices/rest/"
                    "PMC12079058/fullTextXML"
                ),
            ),
        ]
    )

    output_path = (
        tmp_path / "W123.xml"
    )

    result = download_europe_pmc_xml(
        {
            "doi": (
                "https://doi.org/"
                "10.2196/66098"
            )
        },
        session,
        output_path,
        fallback_config(tmp_path),
    )

    assert output_path.read_bytes() == (
        xml_content
    )

    assert result["pmcid"] == (
        "PMC12079058"
    )

    assert len(result["sha256"]) == 64

    assert existing_xml_is_valid(
        output_path,
        minimum_xml_bytes=20,
    )


def test_invalid_europe_pmc_content_is_rejected(
    tmp_path,
):
    session = FakeSession(
        [
            FakeResponse(
                status_code=200,
                json_data=(
                    europe_pmc_search_payload()
                ),
            ),
            FakeResponse(
                status_code=200,
                content=(
                    b"<html>Not article XML</html>"
                ),
            ),
        ]
    )

    output_path = (
        tmp_path / "W123.xml"
    )

    with pytest.raises(
        EuropePmcFallbackError
    ):
        download_europe_pmc_xml(
            {
                "doi": "10.2196/66098"
            },
            session,
            output_path,
            fallback_config(tmp_path),
        )

    assert not output_path.exists()
    assert not (
        tmp_path / "W123.xml.part"
    ).exists()


def test_direct_download_uses_xml_fallback(
    tmp_path,
):
    input_path = tmp_path / "input.jsonl"

    paper = {
        "id": "https://openalex.org/W123",
        "title": "RAG medical paper",
        "doi": (
            "https://doi.org/10.2196/66098"
        ),
        "oa_pdf_urls": [
            "https://publisher.example/paper.pdf"
        ],
    }

    input_path.write_text(
        json.dumps(paper) + "\n",
        encoding="utf-8",
    )

    xml_content = (
        b"<article><body><p>Structured full text"
        b" for retrieval.</p></body></article>"
    )

    session = FakeSession(
        [
            FakeResponse(
                status_code=403,
                url=(
                    "https://publisher.example/"
                    "paper.pdf"
                ),
            ),
            FakeResponse(
                status_code=200,
                json_data=(
                    europe_pmc_search_payload()
                ),
            ),
            FakeResponse(
                status_code=200,
                content=xml_content,
            ),
        ]
    )

    config = {
        "direct_pdf_download": {
            "input_path": str(input_path),
            "output_directory": str(
                tmp_path / "pdf"
            ),
            "manifest_path": str(
                tmp_path / "manifest.jsonl"
            ),
            "report_path": str(
                tmp_path / "report.json"
            ),
            "failures_path": str(
                tmp_path / "failures.jsonl"
            ),
            "default_limit": 1,
            "connect_timeout_seconds": 10,
            "read_timeout_seconds": 30,
            "max_retries_per_url": 1,
            "retry_delay_seconds": 0,
            "request_delay_seconds": 0,
            "fallback_delay_seconds": 0,
            "chunk_size_bytes": 32,
            "minimum_pdf_bytes": 20,
            "maximum_pdf_bytes": 10000,
            "europe_pmc_fallback": (
                fallback_config(tmp_path)
            ),
        }
    }

    report = download_direct_oa_pdfs(
        config,
        tmp_path,
        limit=1,
        session=session,
    )

    assert report[
        "new_pdfs_downloaded"
    ] == 0

    assert report[
        "new_europe_pmc_xml_downloaded"
    ] == 1

    assert report[
        "local_valid_full_text"
    ] == 1

    assert report[
        "remaining_without_local_full_text"
    ] == 0

    assert (
        tmp_path / "xml" / "W123.xml"
    ).exists()


def test_failure_categories():
    assert classify_pdf_failure(
        "Remote host returned HTTP 403."
    ) == "blocked_by_remote_host"

    assert classify_pdf_failure(
        "Downloaded response is too small."
    ) == "invalid_or_intermediate_content"
