import argparse
import json
from pathlib import Path

import yaml

from agentic_rag.config import PROJECT_ROOT
from agentic_rag.evaluation.dataset import load_evaluation_dataset
from agentic_rag.evaluation.langsmith_sync import upload_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the local evaluation benchmark to LangSmith."
    )
    parser.add_argument("--config", default="configs/evaluation.yaml")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    arguments = parse_arguments()
    config = yaml.safe_load(resolve_path(arguments.config).read_text(encoding="utf-8"))
    evaluation_config = config["evaluation"]
    langsmith_config = evaluation_config["langsmith"]
    examples = load_evaluation_dataset(
        resolve_path(evaluation_config["benchmark_path"])
    )
    result = upload_dataset(
        examples=examples,
        dataset_name=langsmith_config["dataset_name"],
        description=langsmith_config["description"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
