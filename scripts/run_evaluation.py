import argparse
import json
from pathlib import Path

import yaml

from agentic_rag.config import PROJECT_ROOT
from agentic_rag.evaluation.dataset import load_evaluation_dataset
from agentic_rag.evaluation.pipelines import (
    SUPPORTED_PIPELINES,
    EvaluationPipelineRunner,
)
from agentic_rag.evaluation.ragas_evaluator import RagasEvaluator
from agentic_rag.evaluation.runner import EvaluationRunner


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline, CRAG, web CRAG, and Agentic RAG."
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
    )
    parser.add_argument(
        "--pipelines",
        default=None,
        help="Comma-separated pipeline names.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_path(arguments.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluation_config = config["evaluation"]

    if arguments.pipelines:
        pipelines = []
        for name in arguments.pipelines.split(","):
            pipelines.append(name.strip())
    else:
        pipelines = list(evaluation_config["pipelines"])

    for pipeline in pipelines:
        if pipeline not in SUPPORTED_PIPELINES:
            raise ValueError("Unsupported pipeline: " + pipeline)

    benchmark_path = resolve_path(evaluation_config["benchmark_path"])
    examples = load_evaluation_dataset(
        path=benchmark_path,
        split=arguments.split,
        category=arguments.category,
        limit=arguments.limit,
    )

    ragas_config = evaluation_config.get("ragas", {})
    ragas_evaluator = None
    if bool(ragas_config.get("enabled", True)) and not arguments.skip_ragas:
        ragas_evaluator = RagasEvaluator(ragas_config)

    pipeline_runner = EvaluationPipelineRunner()
    runner = EvaluationRunner(
        pipeline_runner=pipeline_runner,
        ragas_evaluator=ragas_evaluator,
        retrieval_k=int(evaluation_config.get("retrieval_k", 10)),
    )

    try:
        report = runner.run(
            examples=examples,
            pipelines=pipelines,
            records_path=resolve_path(evaluation_config["records_output"]),
            report_path=resolve_path(evaluation_config["report_output"]),
            dataset_path=str(benchmark_path),
            resume=bool(evaluation_config.get("resume", True))
            and not arguments.no_resume,
        )
    finally:
        pipeline_runner.close()

    print()
    print("=" * 80)
    print("AGENTIC RAG EVALUATION REPORT")
    print("=" * 80)
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
