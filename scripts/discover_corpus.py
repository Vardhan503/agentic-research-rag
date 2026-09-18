import argparse
import json

from agentic_rag.config import load_corpus_config
from agentic_rag.ingestion.discovery import (
    create_discovery_report,
    deduplicate_candidates,
    discover_raw_candidates,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Discover research papers from OpenAlex."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a small discovery test.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_corpus_config()

    discovery_config = config["discovery"]

    if args.test:
        candidate_target = 15000
        max_results_per_query = 50

        raw_output = (
            "data/raw/metadata/"
            "discovery_test_raw.jsonl"
        )

        unique_output = (
            "data/interim/"
            "candidates_test_unique.jsonl"
        )

        report_output = (
            "data/interim/"
            "discovery_test_report.json"
        )

        print()
        print("TEST MODE")

    else:
        candidate_target = discovery_config[
            "candidate_target"
        ]

        max_results_per_query = (
            discovery_config[
                "max_results_per_query"
            ]
        )

        raw_output = discovery_config[
            "raw_output"
        ]

        unique_output = discovery_config[
            "unique_output"
        ]

        report_output = discovery_config[
            "report_output"
        ]

        print()
        print("FULL DISCOVERY MODE")

    discovery_result = (
        discover_raw_candidates(
            config=config,
            raw_output_path=raw_output,
            candidate_target=candidate_target,
            max_results_per_query=max_results_per_query,
        )
    )

    print()
    print("=" * 80)
    print("RAW DISCOVERY COMPLETE")
    print("=" * 80)

    print(
        json.dumps(
            discovery_result,
            indent=2,
        )
    )

    deduplication_result = (
        deduplicate_candidates(
            raw_input_path=raw_output,
            unique_output_path=unique_output,
        )
    )

    print()
    print("=" * 80)
    print("DEDUPLICATION COMPLETE")
    print("=" * 80)

    print(
        json.dumps(
            deduplication_result,
            indent=2,
        )
    )

    report = create_discovery_report(
        unique_input_path=unique_output,
        report_output_path=report_output,
    )

    print()
    print("=" * 80)
    print("CORPUS REPORT")
    print("=" * 80)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()