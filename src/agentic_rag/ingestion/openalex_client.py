import requests

from agentic_rag.config import get_openalex_api_key


OPENALEX_HOST = "api.openalex.org"
BASE_URL = f"https://{OPENALEX_HOST}"


class OpenAlexClient:
    def __init__(self):
        self.api_key = get_openalex_api_key()
        self.session = requests.Session()

    def search_works(
        self,
        query,
        per_page=10,
    ):
        endpoint = f"{BASE_URL}/works"

        filters = [
            "language:en",
            "open_access.is_oa:true",
            "has_fulltext:true",
            "type:article|preprint",
            "from_publication_date:2017-01-01",
            "to_publication_date:2026-12-31",
        ]

        params = {
            "api_key": self.api_key,
            "search": query,
            "filter": ",".join(filters),
            "per_page": per_page,
            "select": ",".join(
                [
                    "id",
                    "doi",
                    "title",
                    "publication_year",
                    "publication_date",
                    "type",
                    "language",
                    "cited_by_count",
                    "open_access",
                    "primary_topic",
                    "topics",
                    "keywords",
                    "has_content",
                    "has_fulltext",
                    "content_urls",
                    "referenced_works",
                ]
            ),
        }

        response = self.session.get(
            endpoint,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()