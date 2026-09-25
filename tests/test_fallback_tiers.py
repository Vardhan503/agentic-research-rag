import json

import pytest

from agentic_rag.ingestion.direct_pdf_downloader import (
    classify_pdf_failure,
    download_direct_oa_pdfs,
)
from agentic_rag.ingestion.europe_pmc_fallback import (
    existing_xml_is_valid,
)
from agentic_rag.ingestion.mdpi_cdn import (
    MdpiSlugResolver,
    build_mdpi_cdn_url,
    mdpi_article_parts,
    mdpi_cdn_url_for_paper,
    mdpi_slug_candidates,
)
from agentic_rag.ingestion.openalex_content import (
    OpenAlexBudget,
    OpenAlexBudgetDeferred,
    openalex_content_url,
    redact_api_key,
)


PDF_BYTES = b"%PDF-1.4\n" + (b"valid-pdf-content" * 20)

TEI_BYTES = (
    b"<?xml version='1.0'?>"
    b'<TEI xmlns="http://www.tei-c.org/ns/1.0">'
    b"<text><body><p>GROBID full text</p></body></text>"
    b"</TEI>"
)


class FakeResponse:
    def __init__(
        self,
        status_code,
        content=b"",
        url="https://example.org/response",
        headers=None,
        content_type=None,
        json_data=None,
    ):
        self.status_code = status_code
        self.content = content
        self.url = url
        self.json_data = json_data
        self.headers = {
            "Content-Length": str(len(content)),
        }

        if content_type:
            self.headers["Content-Type"] = content_type

        if headers:
            self.headers.update(headers)

        self.closed = False

    def json(self):
        return self.json_data

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.calls.append(url)

        if not self.responses:
            raise AssertionError(
                "Unexpected request: " + url
            )

        return self.responses.pop(0)


def base_download_config(tmp_path, input_path):
    return {
        "input_path": str(input_path),
        "output_directory": str(tmp_path / "pdf"),
        "manifest_path": str(tmp_path / "manifest.jsonl"),
        "report_path": str(tmp_path / "report.json"),
        "failures_path": str(tmp_path / "failures.jsonl"),
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
    }


def openalex_config(tmp_path):
    return {
        "enabled": True,
        "xml_output_directory": str(tmp_path / "xml"),
        "max_wait_seconds": 7200,
        "request_delay_seconds": 0,
        "max_retries": 1,
        "retry_delay_seconds": 0,
        "minimum_xml_bytes": 20,
        "maximum_xml_bytes": 10000,
    }


def write_paper(tmp_path, paper):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(paper) + "\n",
        encoding="utf-8",
    )
    return input_path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def openalex_paper():
    return {
        "id": "https://openalex.org/W123",
        "title": "Blocked publisher paper",
        "doi": "https://doi.org/10.3390/info17020133",
        "oa_pdf_urls": [
            "https://publisher.example/paper.pdf",
        ],
        "has_content": {"pdf": True, "grobid_xml": True},
        "content_urls": {
            "pdf": "https://content.openalex.org/works/W123.pdf",
            "grobid_xml": (
                "https://content.openalex.org/works/W123.grobid-xml"
            ),
        },
    }


# --- MDPI CDN -------------------------------------------------------------


def test_mdpi_article_parts_and_cdn_url():
    parts = mdpi_article_parts(
        "https://www.mdpi.com/2078-2489/17/2/133/pdf?version=1"
    )

    assert parts == ("2078-2489", 17, 133)

    assert build_mdpi_cdn_url("information", 17, 133) == (
        "https://mdpi-res.com/d_attachment/information/"
        "information-17-00133/article_deploy/"
        "information-17-00133.pdf"
    )

    # Single-digit volumes are zero-padded to two digits.
    assert build_mdpi_cdn_url("ai", 6, 226) == (
        "https://mdpi-res.com/d_attachment/ai/"
        "ai-06-00226/article_deploy/ai-06-00226.pdf"
    )

    assert mdpi_article_parts(
        "https://arxiv.org/pdf/2312.10997"
    ) is None


def test_mdpi_slug_candidates_prefer_known_table():
    paper = {
        "doi": "https://doi.org/10.3390/app15010001",
        "primary_location": {
            "source": {"display_name": "Applied Sciences"}
        },
    }

    assert mdpi_slug_candidates(paper) == [
        "applsci",
        "appliedsciences",
        "app",
    ]


def test_mdpi_resolver_verifies_once_per_issn():
    session = FakeSession(
        [
            FakeResponse(403, content_type="text/html"),
            FakeResponse(200, content_type="application/pdf"),
        ]
    )

    resolver = MdpiSlugResolver(
        session,
        {"base_url": "https://mdpi-res.com/d_attachment"},
        timeout=(1, 1),
    )

    paper = {
        "doi": "https://doi.org/10.3390/app15010001",
        "oa_pdf_urls": [
            "https://www.mdpi.com/2076-3417/15/1/1/pdf"
        ],
        "primary_location": {
            "source": {"display_name": "Applied Sciences"}
        },
    }

    # First candidate (applsci) fails, second verifies.
    url = mdpi_cdn_url_for_paper(paper, resolver)

    assert url.endswith(
        "/appliedsciences/appliedsciences-15-00001/"
        "article_deploy/appliedsciences-15-00001.pdf"
    )
    assert len(session.calls) == 2

    # Second paper in the same journal uses the cache.
    second = dict(paper)
    second["oa_pdf_urls"] = [
        "https://www.mdpi.com/2076-3417/15/2/250/pdf"
    ]

    url = mdpi_cdn_url_for_paper(second, resolver)

    assert url.endswith("appliedsciences-15-00250.pdf")
    assert len(session.calls) == 2


def test_mdpi_cdn_tier_downloads(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    paper = {
        "id": "https://openalex.org/W7126397257",
        "title": "MDPI paper",
        "doi": "https://doi.org/10.3390/info17020133",
        "oa_pdf_urls": [
            "https://www.mdpi.com/2078-2489/17/2/133/pdf"
        ],
        "primary_location": {
            "source": {"display_name": "Information"}
        },
    }

    input_path = write_paper(tmp_path, paper)

    session = FakeSession(
        [
            FakeResponse(403, content_type="text/html"),
            FakeResponse(200, content_type="application/pdf"),
            FakeResponse(
                200,
                content=PDF_BYTES,
                content_type="application/pdf",
            ),
        ]
    )

    config = {
        "direct_pdf_download": {
            **base_download_config(tmp_path, input_path),
            "mdpi_cdn_fallback": {
                "enabled": True,
                "base_url": "https://mdpi-res.com/d_attachment",
            },
        }
    }

    report = download_direct_oa_pdfs(
        config, tmp_path, limit=1, session=session
    )

    assert report["new_mdpi_cdn_pdfs_downloaded"] == 1
    assert report["new_publisher_pdfs_downloaded"] == 0
    assert report["remaining_without_local_full_text"] == 0

    assert session.calls[0].startswith("https://www.mdpi.com/")
    assert session.calls[1].startswith("https://mdpi-res.com/")
    assert session.calls[2] == session.calls[1]

    manifest = read_jsonl(tmp_path / "manifest.jsonl")
    assert manifest[0]["tier"] == "mdpi_cdn"
    assert manifest[0]["source_host"] == "mdpi-res.com"


# --- OpenAlex content -----------------------------------------------------


def test_openalex_content_url_appends_key_and_gates():
    paper = openalex_paper()

    url = openalex_content_url(paper, "pdf", "SECRET")
    assert url == (
        "https://content.openalex.org/works/W123.pdf?api_key=SECRET"
    )

    assert openalex_content_url(paper, "pdf", "") is None

    paper["has_content"]["grobid_xml"] = False
    assert openalex_content_url(paper, "grobid_xml", "SECRET") is None

    paper["content_urls"]["pdf"] = "https://evil.example/W123.pdf"
    assert openalex_content_url(paper, "pdf", "SECRET") is None


def test_redact_api_key():
    assert redact_api_key(
        "https://content.openalex.org/works/W1.pdf?api_key=abc123&x=1"
    ) == "https://content.openalex.org/works/W1.pdf?api_key=REDACTED&x=1"

    assert "abc123" not in redact_api_key(
        "HTTP 403 for ?api_key=abc123."
    )


def test_budget_waits_then_proceeds():
    sleeps = []

    budget = OpenAlexBudget(
        max_wait_seconds=7200,
        sleep=sleeps.append,
        clock=lambda: 0.0,
    )

    budget.update_from_headers(
        {
            "x-ratelimit-remaining-usd": "0.00",
            "x-ratelimit-reset": "120",
            "x-ratelimit-cost-usd": "0.01",
        }
    )

    assert not budget.has_budget()

    budget.ensure_budget()

    assert sleeps == [125.0]
    assert budget.waits == 1
    assert budget.has_budget()


def test_budget_defers_when_wait_exceeds_maximum():
    budget = OpenAlexBudget(
        max_wait_seconds=60,
        sleep=lambda seconds: None,
        clock=lambda: 0.0,
    )

    budget.update_from_headers(
        {
            "x-ratelimit-remaining-usd": "0",
            "x-ratelimit-reset": "3000",
        }
    )

    with pytest.raises(OpenAlexBudgetDeferred):
        budget.ensure_budget()

    assert budget.deferred == 1
    assert classify_pdf_failure(
        "OpenAlex free-tier budget exhausted; reset in 3005 seconds"
    ) == "deferred_openalex_budget"


def test_publisher_403_then_openalex_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRETKEY")

    input_path = write_paper(tmp_path, openalex_paper())

    session = FakeSession(
        [
            FakeResponse(403, content_type="text/html"),
            FakeResponse(
                200,
                content=PDF_BYTES,
                content_type="application/pdf",
                headers={
                    "x-ratelimit-remaining-usd": "0.95",
                    "x-ratelimit-reset": "5000",
                    "x-ratelimit-cost-usd": "0.01",
                },
            ),
        ]
    )

    config = {
        "direct_pdf_download": {
            **base_download_config(tmp_path, input_path),
            "openalex_content_fallback": openalex_config(tmp_path),
        }
    }

    report = download_direct_oa_pdfs(
        config, tmp_path, limit=1, session=session
    )

    assert report["new_openalex_pdfs_downloaded"] == 1
    assert report["remaining_without_local_full_text"] == 0
    assert report["openalex_budget"]["remaining_usd"] == 0.95
    assert report["openalex_budget"]["metered_downloads"] == 1
    assert report["tiers_enabled"]["openalex_content"] is True

    assert "api_key=SECRETKEY" in session.calls[1]

    manifest = read_jsonl(tmp_path / "manifest.jsonl")
    assert manifest[0]["tier"] == "openalex_content_pdf"
    assert "SECRETKEY" not in manifest[0]["source_url"]
    assert "api_key=REDACTED" in manifest[0]["source_url"]
    assert manifest[0]["source_host"] == "content.openalex.org"


def test_grobid_xml_tier_after_pdf_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRETKEY")

    input_path = write_paper(tmp_path, openalex_paper())

    session = FakeSession(
        [
            FakeResponse(403, content_type="text/html"),
            FakeResponse(404, content_type="application/json"),
            FakeResponse(
                200,
                content=TEI_BYTES,
                content_type="application/xml",
                headers={
                    "x-ratelimit-remaining-usd": "0.90",
                    "x-ratelimit-reset": "5000",
                },
            ),
        ]
    )

    config = {
        "direct_pdf_download": {
            **base_download_config(tmp_path, input_path),
            "openalex_content_fallback": openalex_config(tmp_path),
        }
    }

    report = download_direct_oa_pdfs(
        config, tmp_path, limit=1, session=session
    )

    assert report["new_openalex_pdfs_downloaded"] == 0
    assert report["new_openalex_grobid_xml_downloaded"] == 1
    assert report["local_valid_xml"] == 1
    assert report["remaining_without_local_full_text"] == 0

    xml_path = tmp_path / "xml" / "W123.xml"
    assert xml_path.read_bytes() == TEI_BYTES

    manifest = read_jsonl(tmp_path / "manifest.jsonl")
    assert manifest[0]["content_format"] == "grobid_tei_xml"
    assert "SECRETKEY" not in json.dumps(manifest)


def test_no_api_key_skips_openalex_tier(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    input_path = write_paper(tmp_path, openalex_paper())

    session = FakeSession(
        [FakeResponse(403, content_type="text/html")]
    )

    config = {
        "direct_pdf_download": {
            **base_download_config(tmp_path, input_path),
            "openalex_content_fallback": openalex_config(tmp_path),
        }
    }

    report = download_direct_oa_pdfs(
        config, tmp_path, limit=1, session=session
    )

    assert len(session.calls) == 1
    assert report["tiers_enabled"]["openalex_content"] is False
    assert report["failures_this_run"] == 1

    failures = read_jsonl(tmp_path / "failures.jsonl")
    assert failures[0]["attempted_tiers"] == ["publisher_pdf"]
    assert failures[0]["failure_category"] == "blocked_by_remote_host"


def test_budget_exhaustion_defers_paper(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRETKEY")

    input_path = write_paper(tmp_path, openalex_paper())

    # First metered response reports the budget is gone with a long
    # reset; the paper's XML tier must be deferred, not attempted.
    session = FakeSession(
        [
            FakeResponse(403, content_type="text/html"),
            FakeResponse(
                402,
                content_type="application/json",
                headers={
                    "x-ratelimit-remaining-usd": "0",
                    "x-ratelimit-reset": "5000",
                },
            ),
        ]
    )

    config = {
        "direct_pdf_download": {
            **base_download_config(tmp_path, input_path),
            "openalex_content_fallback": {
                **openalex_config(tmp_path),
                "max_wait_seconds": 60,
            },
        }
    }

    report = download_direct_oa_pdfs(
        config, tmp_path, limit=1, session=session
    )

    assert len(session.calls) == 2
    assert report["deferred_openalex_budget"] == 1
    assert report["failure_categories"] == {
        "deferred_openalex_budget": 1
    }

    failures = read_jsonl(tmp_path / "failures.jsonl")
    assert failures[0]["failure_category"] == "deferred_openalex_budget"
    assert "openalex_grobid_xml" in failures[0]["attempted_tiers"]


# --- XML validation -------------------------------------------------------


def test_existing_xml_is_valid_accepts_tei(tmp_path):
    xml_path = tmp_path / "W1.xml"
    xml_path.write_bytes(TEI_BYTES)

    assert existing_xml_is_valid(xml_path, minimum_xml_bytes=20)

    xml_path.write_bytes(b"<html><body>Denied</body></html>")

    assert not existing_xml_is_valid(xml_path, minimum_xml_bytes=20)
