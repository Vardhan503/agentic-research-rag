import argparse
import json

from agentic_rag.config import (
    PROJECT_ROOT,
    load_corpus_config,
)
from agentic_rag.ingestion.oa_location_enricher import (
    enrich_oa_locations,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Enrich the final corpus with "
            "open-access PDF locations."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of new papers "
            "to enrich."
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = load_corpus_config()

    report = enrich_oa_locations(
        config,
        PROJECT_ROOT,
        limit=arguments.limit,
    )

    print()
    print("=" * 80)
    print("OA LOCATION ENRICHMENT REPORT")
    print("=" * 80)
    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()