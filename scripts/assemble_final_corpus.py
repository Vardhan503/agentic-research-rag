import argparse
import json

from agentic_rag.config import (
    PROJECT_ROOT,
    load_corpus_config,
)
from agentic_rag.processing.final_corpus import (
    assemble_final_corpus,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to corpus configuration. "
            "Defaults to configs/corpus.yaml."
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    config = load_corpus_config(
        arguments.config
    )

    report = assemble_final_corpus(
        config,
        PROJECT_ROOT,
    )

    print("\n" + "=" * 80)
    print("FINAL CORPUS REPORT")
    print("=" * 80)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()