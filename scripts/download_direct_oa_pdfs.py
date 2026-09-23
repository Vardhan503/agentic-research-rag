import argparse
import json

from agentic_rag.config import (
    PROJECT_ROOT,
    load_corpus_config,
)
from agentic_rag.ingestion.direct_pdf_downloader import (
    download_direct_oa_pdfs,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Download PDFs from original "
            "open-access locations."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of new papers "
            "to attempt."
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = load_corpus_config()

    configured_limit = config[
        "direct_pdf_download"
    ]["default_limit"]

    effective_limit = arguments.limit

    if effective_limit is None:
        effective_limit = configured_limit

    print()
    print("=" * 80)
    print("DIRECT OA PDF DOWNLOAD")
    print("=" * 80)
    print()
    print("Maximum papers to attempt:")
    print(effective_limit)
    print()
    print(
        "Existing valid PDFs will be "
        "skipped automatically."
    )

    report = download_direct_oa_pdfs(
        config,
        PROJECT_ROOT,
        limit=arguments.limit,
    )

    print()
    print("=" * 80)
    print("DIRECT OA PDF REPORT")
    print("=" * 80)
    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()