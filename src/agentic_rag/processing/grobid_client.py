from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


class GrobidClientError(Exception):
    """Raised when GROBID cannot process a PDF."""


def local_xml_name(tag: str) -> str:
    """Remove an XML namespace from an element name."""

    if "}" in tag:
        return tag.split("}", maxsplit=1)[1].lower()

    return tag.lower()


def validate_pdf_input(path: Path) -> None:
    """Confirm that the input exists and looks like a PDF."""

    if not path.exists():
        raise GrobidClientError(
            f"PDF file does not exist: {path}"
        )

    if not path.is_file():
        raise GrobidClientError(
            f"PDF path is not a file: {path}"
        )

    try:
        with path.open("rb") as file:
            signature = file.read(5)
    except OSError as error:
        raise GrobidClientError(
            f"PDF file cannot be read: {path}"
        ) from error

    if signature != b"%PDF-":
        raise GrobidClientError(
            f"File does not have a valid PDF signature: {path}"
        )


def validate_tei_content(
    content: bytes,
    minimum_bytes: int,
) -> tuple[bool, str]:
    """Check whether GROBID returned usable TEI XML."""

    if len(content) < minimum_bytes:
        return False, "tei_response_too_small"

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return False, "invalid_xml_response"

    root_name = local_xml_name(root.tag)

    if root_name != "tei":
        return False, f"unexpected_xml_root:{root_name}"

    return True, "valid_tei_xml"

def validate_tei_file(
    path: Path,
    minimum_bytes: int,
) -> tuple[bool, str]:
    """Validate a TEI XML file already stored locally."""

    if not path.exists():
        return False, "tei_file_missing"

    if not path.is_file():
        return False, "tei_path_is_not_a_file"

    try:
        content = path.read_bytes()
    except OSError:
        return False, "tei_file_cannot_be_read"

    return validate_tei_content(
        content=content,
        minimum_bytes=minimum_bytes,
    )

def boolean_form_value(value: bool) -> str:
    """Convert a Python boolean into a GROBID form value."""

    if value:
        return "1"

    return "0"


class GrobidClient:
    """Send research PDFs to a local GROBID service."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 180,
        max_retries: int = 3,
        retry_delay_seconds: float = 5,
        minimum_tei_bytes: int = 1000,
        consolidate_header: bool = False,
        consolidate_citations: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        """Store the GROBID connection and retry settings."""

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.minimum_tei_bytes = minimum_tei_bytes
        self.consolidate_header = consolidate_header
        self.consolidate_citations = consolidate_citations

        if session is None:
            self.session = requests.Session()
        else:
            self.session = session

    def version(self) -> str:
        """Return the running GROBID version."""

        url = f"{self.base_url}/api/version"

        try:
            response = self.session.get(
                url,
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise GrobidClientError(
                f"GROBID is not available at {self.base_url}"
            ) from error

        return response.text.strip()

    def process_pdf(
        self,
        pdf_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        """Convert one PDF into TEI XML and save it locally."""

        validate_pdf_input(pdf_path)

        url = (
            f"{self.base_url}/api/"
            "processFulltextDocument"
        )

        form_data = {
            "consolidateHeader": boolean_form_value(
                self.consolidate_header
            ),
            "consolidateCitations": boolean_form_value(
                self.consolidate_citations
            ),
            "includeRawAffiliations": "1",
            "includeRawCitations": "1",
        }

        last_error = "unknown_error"

        for attempt in range(1, self.max_retries + 1):
            try:
                with pdf_path.open("rb") as pdf_file:
                    files = {
                        "input": (
                            pdf_path.name,
                            pdf_file,
                            "application/pdf",
                        )
                    }

                    response = self.session.post(
                        url,
                        data=form_data,
                        files=files,
                        headers={
                            "Accept": "application/xml",
                        },
                        timeout=self.timeout_seconds,
                    )
            except requests.RequestException as error:
                last_error = (
                    f"request_failed:{type(error).__name__}"
                )

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue

                raise GrobidClientError(
                    f"GROBID request failed for {pdf_path}: "
                    f"{error}"
                ) from error
            except OSError as error:
                raise GrobidClientError(
                    f"Could not open PDF file: {pdf_path}"
                ) from error

            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = (
                    f"retryable_http_status:"
                    f"{response.status_code}"
                )

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue

                raise GrobidClientError(
                    f"GROBID returned HTTP "
                    f"{response.status_code} for {pdf_path}"
                )

            if response.status_code != 200:
                raise GrobidClientError(
                    f"GROBID returned HTTP "
                    f"{response.status_code} for {pdf_path}"
                )

            valid, reason = validate_tei_content(
                response.content,
                self.minimum_tei_bytes,
            )

            if not valid:
                last_error = reason

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue

                raise GrobidClientError(
                    f"Invalid GROBID TEI response for "
                    f"{pdf_path}: {reason}"
                )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = output_path.with_suffix(
                f"{output_path.suffix}.tmp"
            )

            try:
                temporary_path.write_bytes(response.content)
                temporary_path.replace(output_path)
            except OSError as error:
                raise GrobidClientError(
                    f"Could not save TEI XML to {output_path}"
                ) from error

            return {
                "status": "success",
                "pdf_path": str(pdf_path),
                "tei_path": str(output_path),
                "tei_bytes": len(response.content),
                "attempts": attempt,
                "validation": reason,
            }

        raise GrobidClientError(
            f"GROBID processing failed for {pdf_path}: "
            f"{last_error}"
        )