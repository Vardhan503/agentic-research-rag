from agentic_rag.ingestion.direct_pdf_downloader import (
    download_paper_pdf,
    existing_pdf_is_valid,
    is_safe_remote_url,
    ordered_pdf_urls,
)


class FakeResponse:
    def __init__(
        self,
        status_code,
        content,
        url,
    ):
        self.status_code = status_code
        self.content = content
        self.url = url
        self.headers = {
            "Content-Length": str(
                len(content)
            )
        }
        self.closed = False

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

    def get(
        self,
        url,
        stream,
        allow_redirects,
        timeout,
    ):
        self.calls.append(url)
        return self.responses.pop(0)


def test_safe_remote_urls():
    assert is_safe_remote_url(
        "https://arxiv.org/pdf/1234"
    )

    assert not is_safe_remote_url(
        "file:///tmp/paper.pdf"
    )

    assert not is_safe_remote_url(
        "http://localhost/paper.pdf"
    )

    assert not is_safe_remote_url(
        "http://127.0.0.1/paper.pdf"
    )


def test_ordered_pdf_urls():
    paper = {
        "oa_pdf_urls": [
            "https://arxiv.org/a.pdf",
            "https://arxiv.org/a.pdf",
            "file:///tmp/private.pdf",
            "https://example.org/b.pdf",
        ]
    }

    assert ordered_pdf_urls(paper) == [
        "https://arxiv.org/a.pdf",
        "https://example.org/b.pdf",
    ]


def test_existing_pdf_validation(
    tmp_path,
):
    pdf_path = tmp_path / "paper.pdf"

    pdf_path.write_bytes(
        b"%PDF-1.4\n" + (b"x" * 100)
    )

    assert existing_pdf_is_valid(
        pdf_path,
        minimum_pdf_bytes=20,
    )


def test_pdf_fallback(tmp_path):
    invalid_response = FakeResponse(
        status_code=200,
        content=(
            b"<html>Access denied</html>"
        ),
        url=(
            "https://publisher.example/"
            "paper.pdf"
        ),
    )

    pdf_content = (
        b"%PDF-1.4\n"
        + (b"valid-pdf-content" * 20)
    )

    valid_response = FakeResponse(
        status_code=200,
        content=pdf_content,
        url=(
            "https://repository.example/"
            "paper.pdf"
        ),
    )

    session = FakeSession(
        [
            invalid_response,
            valid_response,
        ]
    )

    paper = {
        "oa_pdf_urls": [
            (
                "https://publisher.example/"
                "paper.pdf"
            ),
            (
                "https://repository.example/"
                "paper.pdf"
            ),
        ]
    }

    output_path = (
        tmp_path / "W123.pdf"
    )

    config = {
        "connect_timeout_seconds": 10,
        "read_timeout_seconds": 30,
        "max_retries_per_url": 1,
        "retry_delay_seconds": 0,
        "fallback_delay_seconds": 0,
        "chunk_size_bytes": 32,
        "minimum_pdf_bytes": 20,
        "maximum_pdf_bytes": 10000,
    }

    result = download_paper_pdf(
        paper,
        session,
        output_path,
        config,
    )

    assert output_path.read_bytes() == (
        pdf_content
    )

    assert result["size_bytes"] == len(
        pdf_content
    )

    assert len(result["sha256"]) == 64

    assert result["source_host"] == (
        "repository.example"
    )

    assert session.calls == [
        (
            "https://publisher.example/"
            "paper.pdf"
        ),
        (
            "https://repository.example/"
            "paper.pdf"
        ),
    ]

    assert not (
        tmp_path / "W123.pdf.part"
    ).exists()