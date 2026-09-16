from agentic_rag.ingestion.discovery import build_filters


def test_build_filters():
    config = {
        "corpus": {
            "language": "en",
            "publication_year": {
                "min": 2017,
                "max": 2026,
            },
            "work_types": [
                "article",
                "preprint",
            ],
            "require_open_access": True,
            "require_full_text": True,
        }
    }

    filters = build_filters(config)

    assert "language:en" in filters

    assert (
        "from_publication_date:2017-01-01"
        in filters
    )

    assert (
        "to_publication_date:2026-12-31"
        in filters
    )

    assert "open_access.is_oa:true" in filters

    assert "has_fulltext:true" in filters