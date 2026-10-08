import argparse
import json
from pathlib import Path

import yaml

from agentic_rag.config import PROJECT_ROOT
from agentic_rag.evaluation.ragas_evaluator import RagasEvaluator
from agentic_rag.evaluation.ragas_runner import score_saved_records


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add RAGAS scores to saved pipeline evaluation records."
    )
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--pipeline", default=None)
    parser.add_argument("--limit", type=int, default=None)
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
    evaluator = RagasEvaluator(evaluation_config["ragas"])
    benchmark_path = resolve_path(evaluation_config["benchmark_path"])

    report = score_saved_records(
        records_path=resolve_path(evaluation_config["records_output"]),
        report_path=resolve_path(evaluation_config["report_output"]),
        dataset_path=str(benchmark_path),
        evaluator=evaluator,
        pipeline=arguments.pipeline,
        limit=arguments.limit,
    )

    print()
    print("=" * 80)
    print("RAGAS SCORING REPORT")
    print("=" * 80)
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
