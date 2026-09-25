import os
import re
import time
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

import requests

from agentic_rag.ingestion.europe_pmc_fallback import (
    EuropePmcFallbackError,
    write_validated_xml,
)


OPENALEX_CONTENT_HOST = "content.openalex.org"

DEFAULT_COST_USD = 0.01

CONTENT_KINDS = {
    "pdf": "pdf",
    "grobid_xml": "grobid_xml",
}


class OpenAlexContentError(RuntimeError):
    pass


class OpenAlexBudgetDeferred(OpenAlexContentError):
    """
    Raised when the free-tier budget is exhausted and the
    wait until reset exceeds the configured maximum.
    """


def get_openalex_api_key():
    return os.getenv(
        "OPENALEX_API_KEY",
        "",
    ).strip()


def redact_api_key(text):
    if text is None:
        return text

    return re.sub(
        r"(api_key=)[^&\s'\"]+",
        r"\1REDACTED",
        str(text),
    )


def append_api_key(url, api_key):
    parsed = urlparse(str(url))

    query_items = [
        (key, value)
        for key, value in parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
        if key != "api_key"
    ]

    query_items.append(
        ("api_key", api_key)
    )

    return urlunparse(
        parsed._replace(
            query=urlencode(query_items)
        )
    )


def openalex_content_url(
    paper,
    kind,
    api_key,
):
    """
    Return the api-key-authenticated content.openalex.org
    URL for `kind` ("pdf" or "grobid_xml"), or None when
    the paper has no such content or no key is available.
    """
    if kind not in CONTENT_KINDS:
        raise ValueError(
            "Unknown OpenAlex content kind: "
            + str(kind)
        )

    if not api_key:
        return None

    has_content = paper.get(
        "has_content"
    ) or {}

    if not has_content.get(kind):
        return None

    content_urls = paper.get(
        "content_urls"
    ) or {}

    url = content_urls.get(kind)

    if not url:
        return None

    hostname = urlparse(str(url)).hostname

    if not hostname:
        return None

    if hostname.lower() != OPENALEX_CONTENT_HOST:
        return None

    return append_api_key(url, api_key)


def parse_float_header(headers, name):
    value = headers.get(name)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class OpenAlexBudget:
    """
    Tracks the metered free-tier budget reported by
    content.openalex.org via x-ratelimit-* headers.
    """

    def __init__(
        self,
        max_wait_seconds,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.max_wait_seconds = float(
            max_wait_seconds
        )
        self.sleep = sleep
        self.clock = clock

        self.remaining_usd = None
        self.reset_seconds = None
        self.cost_usd = DEFAULT_COST_USD
        self.observed_at = None

        self.metered_downloads = 0
        self.waits = 0
        self.deferred = 0

    def update_from_headers(self, headers):
        if headers is None:
            return

        remaining = parse_float_header(
            headers,
            "x-ratelimit-remaining-usd",
        )

        if remaining is not None:
            self.remaining_usd = remaining

        reset = parse_float_header(
            headers,
            "x-ratelimit-reset",
        )

        if reset is not None:
            self.reset_seconds = reset

        cost = parse_float_header(
            headers,
            "x-ratelimit-cost-usd",
        )

        if cost is not None and cost > 0:
            self.cost_usd = cost

        self.observed_at = self.clock()

    def seconds_until_reset(self):
        if self.reset_seconds is None:
            return None

        if self.observed_at is None:
            return self.reset_seconds

        elapsed = self.clock() - self.observed_at

        return max(
            self.reset_seconds - elapsed,
            0.0,
        )

    def has_budget(self):
        if self.remaining_usd is None:
            return True

        return self.remaining_usd >= self.cost_usd

    def ensure_budget(self):
        """
        Block until the budget allows one more metered
        call. Raises OpenAlexBudgetDeferred when the wait
        would exceed max_wait_seconds.
        """
        if self.has_budget():
            return

        wait_seconds = self.seconds_until_reset()

        if wait_seconds is None:
            wait_seconds = self.max_wait_seconds

        wait_seconds = wait_seconds + 5.0

        if wait_seconds > self.max_wait_seconds:
            self.deferred += 1

            raise OpenAlexBudgetDeferred(
                "OpenAlex free-tier budget exhausted; "
                "reset in "
                + str(int(wait_seconds))
                + " seconds exceeds the configured "
                "maximum wait."
            )

        print(
            "OpenAlex budget exhausted. Waiting",
            int(wait_seconds),
            "seconds for the free-tier window to reset...",
        )

        self.waits += 1
        self.sleep(wait_seconds)

        # After the reset the remaining budget is
        # unknown until the next response arrives.
        self.remaining_usd = None
        self.reset_seconds = None

    def record_download(self):
        self.metered_downloads += 1

    def snapshot(self):
        return {
            "remaining_usd": self.remaining_usd,
            "reset_seconds": (
                self.seconds_until_reset()
            ),
            "cost_usd": self.cost_usd,
            "metered_downloads": (
                self.metered_downloads
            ),
            "waits": self.waits,
            "deferred": self.deferred,
        }


def download_openalex_grobid_xml(
    url,
    session,
    output_path,
    xml_config,
    budget=None,
):
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
        xml_config["max_retries"] + 1,
    ):
        response = None

        if temporary_path.exists():
            temporary_path.unlink()

        if budget is not None:
            budget.ensure_budget()

        try:
            response = session.get(
                url,
                stream=True,
                allow_redirects=True,
                timeout=(
                    xml_config[
                        "connect_timeout_seconds"
                    ],
                    xml_config[
                        "read_timeout_seconds"
                    ],
                ),
            )

            if budget is not None:
                budget.update_from_headers(
                    getattr(
                        response,
                        "headers",
                        None,
                    )
                )

            if response.status_code >= 400:
                raise OpenAlexContentError(
                    "OpenAlex content returned HTTP "
                    + str(response.status_code)
                    + "."
                )

            result = write_validated_xml(
                response,
                temporary_path,
                xml_config,
            )

            temporary_path.replace(
                output_path
            )

            if budget is not None:
                budget.record_download()

            result["source_url"] = redact_api_key(
                url
            )
            result["source_host"] = (
                OPENALEX_CONTENT_HOST
            )

            return result

        except (
            requests.RequestException,
            EuropePmcFallbackError,
            OpenAlexContentError,
            OSError,
        ) as error:
            last_error = error

            if temporary_path.exists():
                temporary_path.unlink()

        finally:
            if response is not None:
                response.close()

        if attempt_number < xml_config[
            "max_retries"
        ]:
            time.sleep(
                xml_config[
                    "retry_delay_seconds"
                ] * attempt_number
            )

    raise OpenAlexContentError(
        "OpenAlex GROBID XML download failed "
        "after retries: "
        + redact_api_key(str(last_error))
    )
