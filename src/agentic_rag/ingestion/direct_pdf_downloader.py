import hashlib
import ipaddress
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from agentic_rag.ingestion.openalex_client import (
    normalize_work_id,
)


class PdfDownloadError(RuntimeError):
    pass


class RetryablePdfDownloadError(
    PdfDownloadError
):
    pass


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


def build_user_agent():
    contact_email = os.getenv(
        "RESEARCH_CONTACT_EMAIL",
        "",
    ).strip()

    user_agent = (
        "AgenticResearchRAG/0.1 "
        "(open-access research downloader"
    )

    if contact_email:
        user_agent += (
            "; contact="
            + contact_email
        )

    user_agent += ")"

    return user_agent


def is_safe_remote_url(url):
    if not url:
        return False

    parsed = urlparse(str(url))

    if parsed.scheme not in {
        "http",
        "https",
    }:
        return False

    hostname = parsed.hostname

    if not hostname:
        return False

    normalized_hostname = hostname.lower()

    if normalized_hostname in {
        "localhost",
        "localhost.localdomain",
    }:
        return False

    if normalized_hostname.endswith(
        ".local"
    ):
        return False

    try:
        address = ipaddress.ip_address(
            normalized_hostname
        )

        if not address.is_global:
            return False

    except ValueError:
        pass

    return True


def ordered_pdf_urls(paper):
    urls = []
    candidate_urls = paper.get(
        "oa_pdf_urls"
    ) or []

    for url in candidate_urls:
        normalized_url = str(url).strip()

        if not is_safe_remote_url(
            normalized_url
        ):
            continue

        if normalized_url not in urls:
            urls.append(normalized_url)

    return urls


def existing_pdf_is_valid(
    path,
    minimum_pdf_bytes,
):
    pdf_path = Path(path)

    if not pdf_path.exists():
        return False

    if (
        pdf_path.stat().st_size
        < minimum_pdf_bytes
    ):
        return False

    with open(
        pdf_path,
        "rb",
    ) as input_file:
        signature = input_file.read(5)

    return signature == b"%PDF-"


def write_validated_pdf(
    response,
    temporary_path,
    download_config,
):
    maximum_bytes = download_config[
        "maximum_pdf_bytes"
    ]

    minimum_bytes = download_config[
        "minimum_pdf_bytes"
    ]

    content_length = response.headers.get(
        "Content-Length"
    )

    if content_length:
        try:
            declared_size = int(
                content_length
            )

            if declared_size > maximum_bytes:
                raise PdfDownloadError(
                    "Server declared a PDF larger "
                    "than the maximum allowed size."
                )

        except ValueError:
            pass

    digest = hashlib.sha256()
    total_bytes = 0

    with open(
        temporary_path,
        "wb",
    ) as output_file:
        for chunk in response.iter_content(
            chunk_size=download_config[
                "chunk_size_bytes"
            ]
        ):
            if not chunk:
                continue

            total_bytes += len(chunk)

            if total_bytes > maximum_bytes:
                raise PdfDownloadError(
                    "PDF exceeded the maximum "
                    "allowed size during download."
                )

            output_file.write(chunk)
            digest.update(chunk)

        output_file.flush()

    if total_bytes < minimum_bytes:
        raise PdfDownloadError(
            "Downloaded response is too small "
            "to be a usable research PDF."
        )

    with open(
        temporary_path,
        "rb",
    ) as input_file:
        signature = input_file.read(5)

    if signature != b"%PDF-":
        raise PdfDownloadError(
            "Downloaded response does not "
            "have a valid PDF signature."
        )

    return {
        "size_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def request_pdf_to_temporary_file(
    session,
    url,
    temporary_path,
    download_config,
):
    last_error = None

    for attempt_number in range(
        1,
        download_config[
            "max_retries_per_url"
        ] + 1,
    ):
        response = None

        if temporary_path.exists():
            temporary_path.unlink()

        try:
            response = session.get(
                url,
                stream=True,
                allow_redirects=True,
                timeout=(
                    download_config[
                        "connect_timeout_seconds"
                    ],
                    download_config[
                        "read_timeout_seconds"
                    ],
                ),
            )

            status_code = (
                response.status_code
            )

            if status_code == 429:
                raise (
                    RetryablePdfDownloadError(
                        "Remote host returned "
                        "HTTP 429."
                    )
                )

            if status_code >= 500:
                raise (
                    RetryablePdfDownloadError(
                        "Remote host returned HTTP "
                        + str(status_code)
                        + "."
                    )
                )

            if status_code >= 400:
                raise PdfDownloadError(
                    "Remote host returned HTTP "
                    + str(status_code)
                    + "."
                )

            final_url = getattr(
                response,
                "url",
                url,
            )

            if not is_safe_remote_url(
                final_url
            ):
                raise PdfDownloadError(
                    "Remote host redirected to "
                    "an unsafe URL."
                )

            result = write_validated_pdf(
                response,
                temporary_path,
                download_config,
            )

            return result

        except (
            requests.RequestException,
            RetryablePdfDownloadError,
        ) as error:
            last_error = error

            if temporary_path.exists():
                temporary_path.unlink()

        except (
            PdfDownloadError,
            OSError,
        ) as error:
            if temporary_path.exists():
                temporary_path.unlink()

            raise PdfDownloadError(
                str(error)
            ) from error

        finally:
            if response is not None:
                response.close()

        if attempt_number < download_config[
            "max_retries_per_url"
        ]:
            wait_seconds = (
                download_config[
                    "retry_delay_seconds"
                ]
                * attempt_number
            )

            time.sleep(wait_seconds)

    raise PdfDownloadError(
        "PDF request failed after retries: "
        + str(last_error)
    )


def download_paper_pdf(
    paper,
    session,
    output_path,
    download_config,
):
    urls = ordered_pdf_urls(paper)

    if not urls:
        raise PdfDownloadError(
            "Paper has no safe direct OA PDF URL."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    errors = []

    for index, url in enumerate(urls):
        try:
            result = (
                request_pdf_to_temporary_file(
                    session,
                    url,
                    temporary_path,
                    download_config,
                )
            )

            temporary_path.replace(
                output_path
            )

            result["source_url"] = url
            result["source_host"] = (
                urlparse(url).hostname
            )

            return result

        except PdfDownloadError as error:
            hostname = (
                urlparse(url).hostname
                or "unknown-host"
            )

            errors.append(
                hostname
                + ": "
                + str(error)
            )

        if index < len(urls) - 1:
            time.sleep(
                download_config[
                    "fallback_delay_seconds"
                ]
            )

    if temporary_path.exists():
        temporary_path.unlink()

    raise PdfDownloadError(
        "All direct OA PDF locations "
        "failed. "
        + " | ".join(errors)
    )


def count_valid_local_pdfs(
    papers,
    output_directory,
    minimum_pdf_bytes,
):
    valid_count = 0

    for paper in papers:
        work_id = paper_key(paper)

        if not work_id:
            continue

        output_path = (
            Path(output_directory)
            / (work_id + ".pdf")
        )

        if existing_pdf_is_valid(
            output_path,
            minimum_pdf_bytes,
        ):
            valid_count += 1

    return valid_count


def download_direct_oa_pdfs(
    config,
    project_root,
    limit=None,
    session=None,
):
    download_config = config[
        "direct_pdf_download"
    ]

    if limit is None:
        limit = int(
            download_config[
                "default_limit"
            ]
        )

    if limit < 1:
        raise ValueError(
            "PDF download limit must be "
            "at least 1."
        )

    input_path = resolve_project_path(
        download_config["input_path"],
        project_root,
    )

    output_directory = (
        resolve_project_path(
            download_config[
                "output_directory"
            ],
            project_root,
        )
    )

    manifest_path = resolve_project_path(
        download_config[
            "manifest_path"
        ],
        project_root,
    )

    report_path = resolve_project_path(
        download_config["report_path"],
        project_root,
    )

    failures_path = resolve_project_path(
        download_config[
            "failures_path"
        ],
        project_root,
    )

    papers = read_jsonl(input_path)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if session is None:
        session = requests.Session()

    session.headers.update(
        {
            "User-Agent": build_user_agent(),
            "Accept": (
                "application/pdf,"
                "application/octet-stream;"
                "q=0.9,*/*;q=0.1"
            ),
        }
    )

    attempted = 0
    downloaded = 0
    skipped_existing = 0
    missing_urls = 0
    failures = []

    for paper in papers:
        work_id = paper_key(paper)

        if not work_id:
            failures.append(
                {
                    "work_id": "",
                    "title": paper.get(
                        "title"
                    ),
                    "error": (
                        "Invalid OpenAlex "
                        "work ID."
                    ),
                }
            )
            continue

        output_path = (
            output_directory
            / (work_id + ".pdf")
        )

        if existing_pdf_is_valid(
            output_path,
            download_config[
                "minimum_pdf_bytes"
            ],
        ):
            skipped_existing += 1
            continue

        urls = ordered_pdf_urls(paper)

        if not urls:
            missing_urls += 1
            continue

        if attempted >= limit:
            break

        attempted += 1

        try:
            result = download_paper_pdf(
                paper,
                session,
                output_path,
                download_config,
            )

            manifest_record = {
                "work_id": work_id,
                "title": paper.get(
                    "title"
                ),
                "local_path": str(
                    output_path
                ),
                "source_url": result[
                    "source_url"
                ],
                "source_host": result[
                    "source_host"
                ],
                "size_bytes": result[
                    "size_bytes"
                ],
                "sha256": result[
                    "sha256"
                ],
                "downloaded_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

            append_jsonl(
                manifest_record,
                manifest_path,
            )

            downloaded += 1

            print(
                "Downloaded:",
                work_id,
                "| New:",
                downloaded,
                "| Attempted:",
                attempted,
                "/",
                limit,
            )

        except PdfDownloadError as error:
            failure_record = {
                "work_id": work_id,
                "title": paper.get(
                    "title"
                ),
                "error": str(error),
                "attempted_hosts": [
                    urlparse(url).hostname
                    for url in urls
                ],
            }

            failures.append(
                failure_record
            )

            print(
                "Failed:",
                work_id,
                "|",
                str(error),
            )

        time.sleep(
            download_config[
                "request_delay_seconds"
            ]
        )

    write_jsonl(
        failures,
        failures_path,
    )

    local_valid_pdfs = (
        count_valid_local_pdfs(
            papers,
            output_directory,
            download_config[
                "minimum_pdf_bytes"
            ],
        )
    )

    remaining = max(
        len(papers) - local_valid_pdfs,
        0,
    )

    status = "partial"

    if remaining == 0:
        status = "complete"

    report = {
        "status": status,
        "total_enriched_papers": len(
            papers
        ),
        "requested_limit": limit,
        "papers_attempted_this_run": (
            attempted
        ),
        "new_pdfs_downloaded": (
            downloaded
        ),
        "existing_pdfs_skipped": (
            skipped_existing
        ),
        "missing_safe_pdf_url": (
            missing_urls
        ),
        "failures_this_run": len(
            failures
        ),
        "local_valid_pdfs": (
            local_valid_pdfs
        ),
        "remaining_without_local_pdf": (
            remaining
        ),
        "output_directory": str(
            output_directory
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