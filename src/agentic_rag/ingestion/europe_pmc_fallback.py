import hashlib
import re
import time
import xml.etree.ElementTree as element_tree
from pathlib import Path

import requests


class EuropePmcFallbackError(RuntimeError):
    pass


# JATS (Europe PMC) uses <article>; GROBID output
# (OpenAlex content) uses a TEI root element.
VALID_XML_ROOTS = {
    "article",
    "article-set",
    "TEI",
}


def xml_root_name(tree):
    root_name = tree.getroot().tag

    if "}" in root_name:
        root_name = root_name.split("}")[-1]

    return root_name


def normalize_doi(value):
    if not value:
        return ""

    normalized_value = str(value).strip().lower()

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ]

    for prefix in prefixes:
        if normalized_value.startswith(prefix):
            normalized_value = normalized_value[
                len(prefix):
            ]
            break

    return normalized_value.strip()


def valid_pmcid(value):
    if not value:
        return False

    return bool(
        re.fullmatch(
            r"PMC[0-9]+",
            str(value).strip().upper(),
        )
    )


def existing_xml_is_valid(
    path,
    minimum_xml_bytes,
):
    xml_path = Path(path)

    if not xml_path.exists():
        return False

    if xml_path.stat().st_size < minimum_xml_bytes:
        return False

    try:
        tree = element_tree.parse(xml_path)
    except (
        element_tree.ParseError,
        OSError,
    ):
        return False

    return xml_root_name(tree) in VALID_XML_ROOTS


def find_europe_pmc_record(
    paper,
    session,
    fallback_config,
):
    doi = normalize_doi(
        paper.get("doi")
    )

    if not doi:
        raise EuropePmcFallbackError(
            "Paper has no DOI for Europe PMC lookup."
        )

    response = None

    try:
        response = session.get(
            fallback_config["search_url"],
            params={
                "query": "DOI:" + doi,
                "format": "json",
                "pageSize": 5,
            },
            headers={
                "Accept": "application/json",
            },
            timeout=(
                fallback_config[
                    "connect_timeout_seconds"
                ],
                fallback_config[
                    "read_timeout_seconds"
                ],
            ),
        )

        if response.status_code >= 400:
            raise EuropePmcFallbackError(
                "Europe PMC search returned HTTP "
                + str(response.status_code)
                + "."
            )

        payload = response.json()

        if not isinstance(payload, dict):
            raise EuropePmcFallbackError(
                "Europe PMC JSON payload must be an object."
            )

    except requests.RequestException as error:
        raise EuropePmcFallbackError(
            "Europe PMC search request failed: "
            + str(error)
        ) from error

    except ValueError as error:
        raise EuropePmcFallbackError(
            "Europe PMC search returned invalid JSON."
        ) from error

    finally:
        if response is not None:
            response.close()

    result_list = payload.get(
        "resultList"
    ) or {}

    results = result_list.get(
        "result"
    ) or []

    for result in results:
        result_doi = normalize_doi(
            result.get("doi")
        )

        if result_doi != doi:
            continue

        pmcid = str(
            result.get("pmcid") or ""
        ).strip().upper()

        if not valid_pmcid(pmcid):
            continue

        if result.get("isOpenAccess") != "Y":
            continue

        if result.get("inEPMC") != "Y":
            continue

        return {
            "doi": doi,
            "pmcid": pmcid,
        }

    raise EuropePmcFallbackError(
        "No open-access Europe PMC full-text record was found."
    )


def write_validated_xml(
    response,
    temporary_path,
    fallback_config,
):
    maximum_bytes = fallback_config[
        "maximum_xml_bytes"
    ]

    minimum_bytes = fallback_config[
        "minimum_xml_bytes"
    ]

    content_length = response.headers.get(
        "Content-Length"
    )

    if content_length:
        try:
            declared_size = int(content_length)

            if declared_size > maximum_bytes:
                raise EuropePmcFallbackError(
                    "Europe PMC XML is larger than the configured maximum."
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
            chunk_size=fallback_config[
                "chunk_size_bytes"
            ]
        ):
            if not chunk:
                continue

            total_bytes += len(chunk)

            if total_bytes > maximum_bytes:
                raise EuropePmcFallbackError(
                    "Europe PMC XML exceeded the configured maximum."
                )

            output_file.write(chunk)
            digest.update(chunk)

        output_file.flush()

    if total_bytes < minimum_bytes:
        raise EuropePmcFallbackError(
            "Europe PMC response is too small to be full-text XML."
        )

    try:
        tree = element_tree.parse(
            temporary_path
        )
    except element_tree.ParseError as error:
        raise EuropePmcFallbackError(
            "Europe PMC response is not valid XML."
        ) from error

    if xml_root_name(tree) not in VALID_XML_ROOTS:
        raise EuropePmcFallbackError(
            "XML does not contain a JATS article or TEI root."
        )

    return {
        "size_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def download_europe_pmc_xml(
    paper,
    session,
    output_path,
    fallback_config,
):
    record = find_europe_pmc_record(
        paper,
        session,
        fallback_config,
    )

    pmcid = record["pmcid"]

    full_text_url = (
        fallback_config[
            "full_text_base_url"
        ].rstrip("/")
        + "/"
        + pmcid
        + "/fullTextXML"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    last_error = None

    for attempt_number in range(
        1,
        fallback_config["max_retries"] + 1,
    ):
        response = None

        if temporary_path.exists():
            temporary_path.unlink()

        try:
            response = session.get(
                full_text_url,
                stream=True,
                headers={
                    "Accept": (
                        "application/xml,text/xml;"
                        "q=0.9,*/*;q=0.1"
                    ),
                },
                timeout=(
                    fallback_config[
                        "connect_timeout_seconds"
                    ],
                    fallback_config[
                        "read_timeout_seconds"
                    ],
                ),
            )

            if response.status_code >= 400:
                raise EuropePmcFallbackError(
                    "Europe PMC full text returned HTTP "
                    + str(response.status_code)
                    + "."
                )

            result = write_validated_xml(
                response,
                temporary_path,
                fallback_config,
            )

            temporary_path.replace(
                output_path
            )

            result["source_url"] = full_text_url
            result["source_host"] = (
                "www.ebi.ac.uk"
            )
            result["doi"] = record["doi"]
            result["pmcid"] = pmcid

            return result

        except (
            requests.RequestException,
            EuropePmcFallbackError,
            OSError,
        ) as error:
            last_error = error

            if temporary_path.exists():
                temporary_path.unlink()

        finally:
            if response is not None:
                response.close()

        if attempt_number < fallback_config[
            "max_retries"
        ]:
            time.sleep(
                fallback_config[
                    "retry_delay_seconds"
                ] * attempt_number
            )

    raise EuropePmcFallbackError(
        "Europe PMC full-text download failed after retries: "
        + str(last_error)
    )
