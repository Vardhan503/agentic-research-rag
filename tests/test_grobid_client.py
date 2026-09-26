from pathlib import Path
from typing import Any

import requests

from agentic_rag.processing.grobid_client import (
    GrobidClient,
    validate_pdf_input,
    validate_tei_content,
)


class FakeResponse:
    """A small fake requests response for testing."""

    def __init__(
        self,
        status_code: int,
        content: bytes = b"",
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}"
            )


class FakeSession:
    """A fake HTTP session that returns prepared responses."""

    def __init__(
        self,
        post_responses: list[FakeResponse] | None = None,
        get_response: FakeResponse | None = None,
    ) -> None:
        if post_responses is None:
            post_responses = []

        self.post_responses = post_responses
        self.get_response = get_response
        self.post_calls = 0

    def get(
        self,
        _url: str,
        **_kwargs: Any,
    ) -> FakeResponse:
        if self.get_response is None:
            return FakeResponse(200, text="0.9.1")

        return self.get_response

    def post(
        self,
        _url: str,
        **_kwargs: Any,
    ) -> FakeResponse:
        self.post_calls += 1

        return self.post_responses.pop(0)


def create_pdf(path: Path) -> None:
    """Create a small file with a valid PDF signature."""

    path.write_bytes(
        b"%PDF-1.7\nTest research paper content"
    )


def create_tei_content() -> bytes:
    """Create a small valid TEI response."""

    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <div>
            <head>Introduction</head>
            <p>This is extracted research paper text.</p>
          </div>
        </body>
      </text>
    </TEI>
    """

    return xml.encode("utf-8")


def test_validate_pdf_input_accepts_pdf(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "W1001.pdf"
    create_pdf(pdf_path)

    validate_pdf_input(pdf_path)


def test_validate_tei_content_accepts_tei() -> None:
    valid, reason = validate_tei_content(
        create_tei_content(),
        minimum_bytes=10,
    )

    assert valid is True
    assert reason == "valid_tei_xml"


def test_grobid_client_returns_version() -> None:
    session = FakeSession(
        get_response=FakeResponse(
            status_code=200,
            text="0.9.1",
        )
    )

    client = GrobidClient(
        base_url="http://localhost:8070",
        session=session,
    )

    assert client.version() == "0.9.1"


def test_grobid_client_processes_pdf(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "W1001.pdf"
    output_path = tmp_path / "W1001.tei.xml"

    create_pdf(pdf_path)

    session = FakeSession(
        post_responses=[
            FakeResponse(
                status_code=200,
                content=create_tei_content(),
            )
        ]
    )

    client = GrobidClient(
        base_url="http://localhost:8070",
        minimum_tei_bytes=10,
        retry_delay_seconds=0,
        session=session,
    )

    result = client.process_pdf(
        pdf_path=pdf_path,
        output_path=output_path,
    )

    assert result["status"] == "success"
    assert result["attempts"] == 1
    assert output_path.exists()


def test_grobid_client_retries_temporary_failure(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "W1002.pdf"
    output_path = tmp_path / "W1002.tei.xml"

    create_pdf(pdf_path)

    session = FakeSession(
        post_responses=[
            FakeResponse(status_code=503),
            FakeResponse(
                status_code=200,
                content=create_tei_content(),
            ),
        ]
    )

    client = GrobidClient(
        base_url="http://localhost:8070",
        minimum_tei_bytes=10,
        max_retries=2,
        retry_delay_seconds=0,
        session=session,
    )

    result = client.process_pdf(
        pdf_path=pdf_path,
        output_path=output_path,
    )

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert session.post_calls == 2