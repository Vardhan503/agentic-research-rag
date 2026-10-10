import argparse
import json

from agentic_rag.config import (
    load_corpus_config,
)
from agentic_rag.processing.corpus_selection import (
    filter_candidates,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=("Score and filter OpenAlex candidates."))

    parser.add_argument(
        "--test",
        action="store_true",
        help="Filter the small test dataset.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_corpus_config()

    discovery_config = config["discovery"]
    selection_config = config["selection"]

    if args.test:
        input_path = "data/interim/candidates_test_unique.jsonl"

        output_paths = {
            "retained": ("data/processed/corpus_test_selected.jsonl"),
            "ambiguous": ("data/interim/corpus_test_ambiguous.jsonl"),
            "rejected": ("data/interim/corpus_test_rejected.jsonl"),
            "report": ("data/interim/selection_test_report.json"),
        }

        print()
        print("TEST FILTERING MODE")

    else:
        input_path = discovery_config["unique_output"]

        output_paths = {
            "retained": selection_config["retained_output"],
            "ambiguous": selection_config["ambiguous_output"],
            "rejected": selection_config["rejected_output"],
            "report": selection_config["report_output"],
        }

        print()
        print("FULL FILTERING MODE")

    report = filter_candidates(
        config=config,
        input_path=input_path,
        output_paths=output_paths,
    )

    print()
    print("=" * 80)
    print("CORPUS FILTERING COMPLETE")
    print("=" * 80)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
