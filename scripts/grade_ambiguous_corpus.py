import argparse
import json

from agentic_rag.config import load_ollama_grading_config
from agentic_rag.processing.ollama_grader import grade_ambiguous_corpus


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        help="Path to corpus YAML (default: configs/corpus.yaml).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Grade only the first N ambiguous papers.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    grading_config = load_ollama_grading_config(arguments.config)

    report = grade_ambiguous_corpus(
        grading_config,
        limit=arguments.limit,
    )

    print("\n" + "=" * 80)
    print("GRADING REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
