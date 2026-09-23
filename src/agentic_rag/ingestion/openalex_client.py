import time
import re
import requests

from agentic_rag.config import get_openalex_api_key


BASE_URL = "https://api.openalex.org"


SELECT_FIELDS = [
    "id",
    "doi",
    "title",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "cited_by_count",
    "abstract_inverted_index",
    "authorships",
    "open_access",
    "primary_topic",
    "topics",
    "keywords",
    "has_content",
    "has_fulltext",
    "content_urls",
    "referenced_works",
    "is_retracted",
]

def normalize_work_id(value):
    if not value:
        return ""

    normalized = str(value).strip()

    if "/" in normalized:
        normalized = normalized.rsplit(
            "/",
            1,
        )[-1]

    normalized = normalized.upper()

    if not re.fullmatch(
        r"W\d+",
        normalized,
    ):
        return ""

    return normalized

class OpenAlexClient:
    def __init__(self, max_retries=5):
        self.api_key = get_openalex_api_key()
        self.max_retries = max_retries

        self.session = requests.Session()

    def _get(self, endpoint, params):
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    endpoint,
                    params=params,
                    timeout=30,
                )

                if response.status_code == 429:
                    wait_seconds = 2 ** (attempt - 1)

                    print(f"OpenAlex rate limit reached. Retrying in {wait_seconds} seconds...")

                    time.sleep(wait_seconds)
                    continue

                if response.status_code >= 500:
                    wait_seconds = 2 ** (attempt - 1)

                    print(f"OpenAlex server error. Retrying in {wait_seconds} seconds...")

                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()

                return response.json()

            except requests.RequestException as error:
                last_error = error

                wait_seconds = 2 ** (attempt - 1)

                print(
                    "Request failed. "
                    f"Attempt {attempt}/{self.max_retries}. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

        raise RuntimeError("OpenAlex request failed after maximum retries.") from last_error

    def search_works_page(
        self,
        query,
        filters,
        cursor="*",
        per_page=100,
    ):
        endpoint = f"{BASE_URL}/works"

        params = {
            "api_key": self.api_key,
            "search": query,
            "filter": filters,
            "per_page": per_page,
            "cursor": cursor,
            "select": ",".join(SELECT_FIELDS),
        }

        return self._get(
            endpoint=endpoint,
            params=params,
        )
    def get_work(
        self,
        work_id,
        select_fields=None,
    ):
        normalized_id = normalize_work_id(
            work_id
        )

        if not normalized_id:
            raise ValueError(
                "Invalid OpenAlex work ID: "
                + str(work_id)
            )

        endpoint = (
            BASE_URL
            + "/works/"
            + normalized_id
        )

        params = {
            "api_key": self.api_key,
        }

        if select_fields:
            params["select"] = ",".join(
                select_fields
            )

        return self._get(
            endpoint=endpoint,
            params=params,
        )
