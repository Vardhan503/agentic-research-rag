from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentic_rag.evaluation.models import EvaluationRecord, EvaluationReport
from agentic_rag.evaluation.ragas_evaluator import RagasEvaluator
from agentic_rag.evaluation.runner import (
    build_evaluation_report,
    record_needs_ragas,
    write_report,
)


def load_records(path: str | Path) -> list[EvaluationRecord]:
    """Load normalized evaluation outputs that are ready for semantic judging."""

    records: list[EvaluationRecord] = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(EvaluationRecord.model_validate_json(line))
            except Exception as error:
                raise ValueError(
                    "Invalid evaluation record on line " + str(line_number)
                ) from error
    return records


def write_records(path: str | Path, records: list[EvaluationRecord]) -> None:
    """Atomically checkpoint records after each expensive RAGAS judgment."""

    output_path = Path(path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )
    temporary_path.replace(output_path)


def score_saved_records(
    records_path: str | Path,
    report_path: str | Path,
    dataset_path: str,
    evaluator: RagasEvaluator,
    pipeline: str | None = None,
    limit: int | None = None,
) -> EvaluationReport:
    """Add RAGAS scores to saved answers without rerunning a RAG pipeline."""

    records = load_records(records_path)
    newly_scored = 0

    for record in records:
        if not record_needs_ragas(record):
            continue
        if pipeline is not None and record.output.pipeline != pipeline:
            continue
        if limit is not None and limit > 0 and newly_scored >= limit:
            break

        print(
            "RAGAS scoring "
            + record.output.pipeline
            + " | "
            + record.example.question_id
        )
        record.ragas = asyncio.run(evaluator.evaluate(record.example, record.output))
        newly_scored += 1
        write_records(records_path, records)

    examples_by_id = {}
    pipelines: list[str] = []
    for record in records:
        examples_by_id[record.example.question_id] = record.example
        if record.output.pipeline not in pipelines:
            pipelines.append(record.output.pipeline)

    examples = list(examples_by_id.values())
    report = build_evaluation_report(
        records=records,
        examples=examples,
        pipelines=pipelines,
        dataset_path=dataset_path,
        ragas_enabled=True,
    )
    write_report(report_path, report)
    return report
