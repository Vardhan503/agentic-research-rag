import argparse
import json

from agentic_rag.config import (
    load_ollama_screening_config,
)
from agentic_rag.processing.batch_screener import (
    screen_ambiguous_corpus,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        help="Path to corpus YAML.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Screen only the next N unprocessed papers.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    screening_config = (
        load_ollama_screening_config(
            arguments.config
        )
    )

    report = screen_ambiguous_corpus(
        screening_config,
        limit=arguments.limit,
    )

    print("\n" + "=" * 80)
    print("AMBIGUOUS-PAPER SCREENING REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()