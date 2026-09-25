import hashlib
import ipaddress
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from agentic_rag.ingestion.europe_pmc_fallback import (
    EuropePmcFallbackError,
    download_europe_pmc_xml,
    existing_xml_is_valid,
)
from agentic_rag.ingestion.mdpi_cdn import (
    MdpiSlugResolver,
    mdpi_cdn_url_for_paper,
)
from agentic_rag.ingestion.openalex_client import (
    normalize_work_id,
)
from agentic_rag.ingestion.openalex_content import (
    OPENALEX_CONTENT_HOST,
    OpenAlexBudget,
    OpenAlexBudgetDeferred,
    OpenAlexContentError,
    download_openalex_grobid_xml,
    get_openalex_api_key,
    openalex_content_url,
    redact_api_key,
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
    on_response=None,
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

            if on_response is not None:
                on_response(response)

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
        + redact_api_key(str(last_error))
    )


def download_pdf_from_url(
    session,
    url,
    output_path,
    download_config,
    source_host=None,
    on_response=None,
):
    """
    Download a single PDF URL straight to output_path.
    The returned source_url never contains an API key.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    try:
        result = request_pdf_to_temporary_file(
            session,
            url,
            temporary_path,
            download_config,
            on_response=on_response,
        )

    except PdfDownloadError as error:
        if temporary_path.exists():
            temporary_path.unlink()

        raise PdfDownloadError(
            redact_api_key(str(error))
        ) from error

    temporary_path.replace(output_path)

    result["source_url"] = redact_api_key(url)
    result["source_host"] = (
        source_host
        or urlparse(url).hostname
    )

    return result


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


def count_valid_local_xml(
    papers,
    output_directory,
    minimum_xml_bytes,
):
    valid_count = 0

    for paper in papers:
        work_id = paper_key(paper)

        if not work_id:
            continue

        output_path = (
            Path(output_directory)
            / (work_id + ".xml")
        )

        if existing_xml_is_valid(
            output_path,
            minimum_xml_bytes,
        ):
            valid_count += 1

    return valid_count


def count_valid_local_full_text(
    papers,
    pdf_output_directory,
    xml_output_directory,
    minimum_pdf_bytes,
    minimum_xml_bytes,
):
    valid_count = 0

    for paper in papers:
        work_id = paper_key(paper)

        if not work_id:
            continue

        pdf_path = (
            Path(pdf_output_directory)
            / (work_id + ".pdf")
        )

        if existing_pdf_is_valid(
            pdf_path,
            minimum_pdf_bytes,
        ):
            valid_count += 1
            continue

        if xml_output_directory is None:
            continue

        xml_path = (
            Path(xml_output_directory)
            / (work_id + ".xml")
        )

        if existing_xml_is_valid(
            xml_path,
            minimum_xml_bytes,
        ):
            valid_count += 1

    return valid_count


def classify_pdf_failure(error):
    error_text = str(error).lower()

    if "budget exhausted" in error_text:
        return "deferred_openalex_budget"

    if (
        "http 401" in error_text
        or "http 403" in error_text
    ):
        return "blocked_by_remote_host"

    if "http 429" in error_text:
        return "rate_limited"

    if "too small" in error_text:
        return "invalid_or_intermediate_content"

    if "valid pdf signature" in error_text:
        return "invalid_or_intermediate_content"

    if "http 5" in error_text:
        return "temporary_remote_error"

    if "no safe direct" in error_text:
        return "missing_direct_pdf_url"

    return "direct_pdf_download_failed"


def add_failure_category(
    counts,
    category,
):
    current_count = counts.get(
        category,
        0,
    )

    counts[category] = current_count + 1


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

    fallback_config = download_config.get(
        "europe_pmc_fallback"
    ) or {}

    fallback_enabled = bool(
        fallback_config.get(
            "enabled",
            False,
        )
    )

    mdpi_config = download_config.get(
        "mdpi_cdn_fallback"
    ) or {}

    mdpi_enabled = bool(
        mdpi_config.get(
            "enabled",
            False,
        )
    )

    openalex_config = download_config.get(
        "openalex_content_fallback"
    ) or {}

    openalex_api_key = get_openalex_api_key()

    openalex_enabled = bool(
        openalex_config.get(
            "enabled",
            False,
        )
    ) and bool(openalex_api_key)

    xml_output_directory = None

    if fallback_enabled:
        xml_output_directory = (
            resolve_project_path(
                fallback_config[
                    "output_directory"
                ],
                project_root,
            )
        )

    if openalex_enabled:
        openalex_xml_directory = (
            openalex_config.get(
                "xml_output_directory"
            )
            or fallback_config.get(
                "output_directory"
            )
        )

        if openalex_xml_directory:
            xml_output_directory = (
                resolve_project_path(
                    openalex_xml_directory,
                    project_root,
                )
            )

    # XML validation limits are shared by the Europe PMC
    # and OpenAlex GROBID tiers.
    xml_config = {
        "connect_timeout_seconds": (
            openalex_config.get(
                "connect_timeout_seconds",
                download_config[
                    "connect_timeout_seconds"
                ],
            )
        ),
        "read_timeout_seconds": (
            openalex_config.get(
                "read_timeout_seconds",
                download_config[
                    "read_timeout_seconds"
                ],
            )
        ),
        "max_retries": openalex_config.get(
            "max_retries",
            download_config[
                "max_retries_per_url"
            ],
        ),
        "retry_delay_seconds": (
            openalex_config.get(
                "retry_delay_seconds",
                download_config[
                    "retry_delay_seconds"
                ],
            )
        ),
        "chunk_size_bytes": download_config[
            "chunk_size_bytes"
        ],
        "minimum_xml_bytes": (
            openalex_config.get(
                "minimum_xml_bytes",
                fallback_config.get(
                    "minimum_xml_bytes",
                    1000,
                ),
            )
        ),
        "maximum_xml_bytes": (
            openalex_config.get(
                "maximum_xml_bytes",
                fallback_config.get(
                    "maximum_xml_bytes",
                    52428800,
                ),
            )
        ),
    }

    minimum_xml_bytes = xml_config[
        "minimum_xml_bytes"
    ]

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if xml_output_directory is not None:
        xml_output_directory.mkdir(
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

    mdpi_resolver = None

    if mdpi_enabled:
        mdpi_resolver = MdpiSlugResolver(
            session,
            mdpi_config,
            timeout=(
                download_config[
                    "connect_timeout_seconds"
                ],
                download_config[
                    "read_timeout_seconds"
                ],
            ),
        )

    budget = OpenAlexBudget(
        max_wait_seconds=openalex_config.get(
            "max_wait_seconds",
            7200,
        )
    )

    openalex_delay = openalex_config.get(
        "request_delay_seconds",
        download_config[
            "request_delay_seconds"
        ],
    )

    attempted = 0
    downloaded = 0
    mdpi_cdn_downloaded = 0
    xml_fallbacks_downloaded = 0
    openalex_pdfs_downloaded = 0
    openalex_xml_downloaded = 0
    deferred_budget = 0
    skipped_existing = 0
    skipped_existing_xml = 0
    missing_urls = 0
    failures = []
    failure_categories = {}

    def record_manifest(
        work_id,
        paper,
        content_format,
        tier,
        local_path,
        result,
    ):
        manifest_record = {
            "work_id": work_id,
            "title": paper.get("title"),
            "content_format": content_format,
            "tier": tier,
            "local_path": str(local_path),
            "source_url": redact_api_key(
                result.get("source_url")
            ),
            "source_host": result.get(
                "source_host"
            ),
            "size_bytes": result.get(
                "size_bytes"
            ),
            "sha256": result.get("sha256"),
            "downloaded_at": datetime.now(
                UTC
            ).isoformat(),
        }

        for key in ("doi", "pmcid"):
            if key in result:
                manifest_record[key] = result[
                    key
                ]

        append_jsonl(
            manifest_record,
            manifest_path,
        )

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

        xml_output_path = None

        if xml_output_directory is not None:
            xml_output_path = (
                xml_output_directory
                / (work_id + ".xml")
            )

            if existing_xml_is_valid(
                xml_output_path,
                minimum_xml_bytes,
            ):
                skipped_existing_xml += 1
                continue

        urls = ordered_pdf_urls(paper)

        if attempted >= limit:
            break

        attempted += 1

        attempted_tiers = []
        tier_errors = []
        resolved = False

        # Tier 1: publisher / repository OA PDF URLs.
        if not urls:
            missing_urls += 1
            tier_errors.append(
                "Paper has no safe direct OA PDF URL."
            )
        else:
            attempted_tiers.append("publisher_pdf")

            try:
                result = download_paper_pdf(
                    paper,
                    session,
                    output_path,
                    download_config,
                )

                record_manifest(
                    work_id,
                    paper,
                    "pdf",
                    "publisher_pdf",
                    output_path,
                    result,
                )

                downloaded += 1
                resolved = True

                print(
                    "Downloaded PDF:",
                    work_id,
                    "| New PDFs:",
                    downloaded,
                    "| Attempted:",
                    attempted,
                    "/",
                    limit,
                )

            except PdfDownloadError as error:
                tier_errors.append(str(error))

        # Tier 2: MDPI static CDN rewrite.
        if not resolved and mdpi_resolver is not None:
            cdn_url = mdpi_cdn_url_for_paper(
                paper,
                mdpi_resolver,
            )

            if cdn_url:
                attempted_tiers.append("mdpi_cdn")

                try:
                    result = download_pdf_from_url(
                        session,
                        cdn_url,
                        output_path,
                        download_config,
                    )

                    record_manifest(
                        work_id,
                        paper,
                        "pdf",
                        "mdpi_cdn",
                        output_path,
                        result,
                    )

                    downloaded += 1
                    mdpi_cdn_downloaded += 1
                    resolved = True

                    print(
                        "Downloaded MDPI CDN PDF:",
                        work_id,
                        "| New MDPI CDN PDFs:",
                        mdpi_cdn_downloaded,
                        "| Attempted:",
                        attempted,
                        "/",
                        limit,
                    )

                except PdfDownloadError as error:
                    tier_errors.append(
                        "MDPI CDN: " + str(error)
                    )

        # Tier 3: Europe PMC JATS XML.
        if (
            not resolved
            and fallback_enabled
            and xml_output_path is not None
        ):
            attempted_tiers.append("europe_pmc_xml")

            try:
                fallback_result = (
                    download_europe_pmc_xml(
                        paper,
                        session,
                        xml_output_path,
                        fallback_config,
                    )
                )

                record_manifest(
                    work_id,
                    paper,
                    "jats_xml",
                    "europe_pmc_xml",
                    xml_output_path,
                    fallback_result,
                )

                xml_fallbacks_downloaded += 1
                resolved = True

                print(
                    "Downloaded Europe PMC XML:",
                    work_id,
                    "| New XML fallbacks:",
                    xml_fallbacks_downloaded,
                    "| Attempted:",
                    attempted,
                    "/",
                    limit,
                )

            except EuropePmcFallbackError as error:
                tier_errors.append(
                    "Europe PMC fallback: "
                    + str(error)
                )

        # Tier 4: OpenAlex content PDF (metered).
        deferred = False

        if not resolved and openalex_enabled:
            content_pdf_url = openalex_content_url(
                paper,
                "pdf",
                openalex_api_key,
            )

            if content_pdf_url:
                attempted_tiers.append(
                    "openalex_content_pdf"
                )

                try:
                    budget.ensure_budget()

                    result = download_pdf_from_url(
                        session,
                        content_pdf_url,
                        output_path,
                        download_config,
                        source_host=(
                            OPENALEX_CONTENT_HOST
                        ),
                        on_response=(
                            lambda response: (
                                budget.update_from_headers(
                                    getattr(
                                        response,
                                        "headers",
                                        None,
                                    )
                                )
                            )
                        ),
                    )

                    budget.record_download()

                    record_manifest(
                        work_id,
                        paper,
                        "pdf",
                        "openalex_content_pdf",
                        output_path,
                        result,
                    )

                    downloaded += 1
                    openalex_pdfs_downloaded += 1
                    resolved = True

                    print(
                        "Downloaded OpenAlex PDF:",
                        work_id,
                        "| New OpenAlex PDFs:",
                        openalex_pdfs_downloaded,
                        "| Budget USD left:",
                        budget.remaining_usd,
                        "| Attempted:",
                        attempted,
                        "/",
                        limit,
                    )

                    time.sleep(openalex_delay)

                except OpenAlexBudgetDeferred as error:
                    deferred = True
                    tier_errors.append(str(error))

                except PdfDownloadError as error:
                    tier_errors.append(
                        "OpenAlex content PDF: "
                        + str(error)
                    )

        # Tier 5: OpenAlex GROBID TEI XML (metered).
        if (
            not resolved
            and not deferred
            and openalex_enabled
            and xml_output_path is not None
        ):
            content_xml_url = openalex_content_url(
                paper,
                "grobid_xml",
                openalex_api_key,
            )

            if content_xml_url:
                attempted_tiers.append(
                    "openalex_grobid_xml"
                )

                try:
                    xml_result = (
                        download_openalex_grobid_xml(
                            content_xml_url,
                            session,
                            xml_output_path,
                            xml_config,
                            budget=budget,
                        )
                    )

                    record_manifest(
                        work_id,
                        paper,
                        "grobid_tei_xml",
                        "openalex_grobid_xml",
                        xml_output_path,
                        xml_result,
                    )

                    openalex_xml_downloaded += 1
                    resolved = True

                    print(
                        "Downloaded OpenAlex GROBID XML:",
                        work_id,
                        "| New GROBID XML:",
                        openalex_xml_downloaded,
                        "| Budget USD left:",
                        budget.remaining_usd,
                        "| Attempted:",
                        attempted,
                        "/",
                        limit,
                    )

                    time.sleep(openalex_delay)

                except OpenAlexBudgetDeferred as error:
                    deferred = True
                    tier_errors.append(str(error))

                except OpenAlexContentError as error:
                    tier_errors.append(
                        "OpenAlex GROBID XML: "
                        + str(error)
                    )

        if resolved:
            time.sleep(
                download_config[
                    "request_delay_seconds"
                ]
            )
            continue

        if deferred:
            deferred_budget += 1

        error_message = redact_api_key(
            " | ".join(tier_errors)
        )

        if deferred:
            category = "deferred_openalex_budget"
        else:
            category = classify_pdf_failure(
                tier_errors[0]
                if tier_errors
                else "direct PDF download failed"
            )

        add_failure_category(
            failure_categories,
            category,
        )

        failure_record = {
            "work_id": work_id,
            "title": paper.get("title"),
            "failure_category": category,
            "error": error_message,
            "attempted_tiers": attempted_tiers,
            "attempted_hosts": [],
        }

        for url in urls:
            failure_record[
                "attempted_hosts"
            ].append(
                urlparse(url).hostname
            )

        failures.append(failure_record)

        print(
            "Failed:",
            work_id,
            "| Category:",
            category,
            "|",
            error_message,
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

    local_valid_xml = 0

    if xml_output_directory is not None:
        local_valid_xml = (
            count_valid_local_xml(
                papers,
                xml_output_directory,
                minimum_xml_bytes,
            )
        )

    local_valid_full_text = (
        count_valid_local_full_text(
            papers,
            output_directory,
            xml_output_directory,
            download_config[
                "minimum_pdf_bytes"
            ],
            minimum_xml_bytes,
        )
    )

    remaining = max(
        len(papers) - local_valid_full_text,
        0,
    )

    remaining_without_pdf = max(
        len(papers) - local_valid_pdfs,
        0,
    )

    status = "partial"

    if remaining == 0:
        status = "complete"

    xml_output_directory_value = None

    if xml_output_directory is not None:
        xml_output_directory_value = str(
            xml_output_directory
        )

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
        "new_publisher_pdfs_downloaded": (
            downloaded
            - mdpi_cdn_downloaded
            - openalex_pdfs_downloaded
        ),
        "new_mdpi_cdn_pdfs_downloaded": (
            mdpi_cdn_downloaded
        ),
        "new_europe_pmc_xml_downloaded": (
            xml_fallbacks_downloaded
        ),
        "new_openalex_pdfs_downloaded": (
            openalex_pdfs_downloaded
        ),
        "new_openalex_grobid_xml_downloaded": (
            openalex_xml_downloaded
        ),
        "deferred_openalex_budget": (
            deferred_budget
        ),
        "openalex_budget": budget.snapshot(),
        "tiers_enabled": {
            "publisher_pdf": True,
            "mdpi_cdn": mdpi_enabled,
            "europe_pmc_xml": fallback_enabled,
            "openalex_content": openalex_enabled,
        },
        "existing_pdfs_skipped": (
            skipped_existing
        ),
        "existing_xml_skipped": (
            skipped_existing_xml
        ),
        "missing_safe_pdf_url": (
            missing_urls
        ),
        "failures_this_run": len(
            failures
        ),
        "failure_categories": (
            failure_categories
        ),
        "local_valid_pdfs": (
            local_valid_pdfs
        ),
        "local_valid_xml": (
            local_valid_xml
        ),
        "local_valid_europe_pmc_xml": (
            local_valid_xml
        ),
        "local_valid_full_text": (
            local_valid_full_text
        ),
        "remaining_without_local_pdf": (
            remaining_without_pdf
        ),
        "remaining_without_local_full_text": (
            remaining
        ),
        "output_directory": str(
            output_directory
        ),
        "xml_output_directory": (
            xml_output_directory_value
        ),
        "updated_at": datetime.now(
            UTC
        ).isoformat(),
    }

    write_json(
        report,
        report_path,
    )

    return report
