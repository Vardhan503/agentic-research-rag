from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_rag.evaluation.metrics import score_deterministic_metrics
from agentic_rag.evaluation.models import (
    EvaluationExample,
    EvaluationRecord,
    EvaluationReport,
    PipelineSummary,
    RagasScores,
)
from agentic_rag.evaluation.pipelines import EvaluationPipelineRunner
from agentic_rag.evaluation.ragas_evaluator import RagasEvaluator


class EvaluationRunner:
    """Execute, checkpoint, resume, and summarize a pipeline comparison."""

    def __init__(
        self,
        pipeline_runner: EvaluationPipelineRunner,
        ragas_evaluator: RagasEvaluator | None = None,
        retrieval_k: int = 10,
    ) -> None:
        self.pipeline_runner = pipeline_runner
        self.ragas_evaluator = ragas_evaluator
        self.retrieval_k = retrieval_k

    def run(
        self,
        examples: list[EvaluationExample],
        pipelines: list[str],
        records_path: str | Path,
        report_path: str | Path,
        dataset_path: str,
        resume: bool = True,
    ) -> EvaluationReport:
        """Run all requested pairs and save every completed record."""

        output_path = Path(records_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_records = self._load_records(output_path) if resume else []
        existing_records = self._deduplicate_records(existing_records)
        if resume and output_path.exists():
            self._rewrite_records(output_path, existing_records)
        completed = self._completed_keys(existing_records)

        file_mode = "a" if resume else "w"
        with output_path.open(file_mode, encoding="utf-8") as output_file:
            for pipeline in pipelines:
                for example in examples:
                    key = (pipeline, example.question_id)
                    if key in completed:
                        continue

                    print("Evaluating " + pipeline + " | " + example.question_id)
                    pipeline_output = self.pipeline_runner.run(
                        pipeline=pipeline,
                        example=example,
                    )
                    deterministic = score_deterministic_metrics(
                        example=example,
                        output=pipeline_output,
                        k=self.retrieval_k,
                    )
                    ragas_scores = None
                    if self.ragas_evaluator is not None:
                        ragas_scores = asyncio.run(
                            self.ragas_evaluator.evaluate(
                                example=example,
                                output=pipeline_output,
                            )
                        )

                    record = EvaluationRecord(
                        example=example,
                        output=pipeline_output,
                        deterministic=deterministic,
                        ragas=ragas_scores,
                    )
                    output_file.write(
                        json.dumps(
                            record.model_dump(mode="json"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    output_file.flush()
                    existing_records.append(record)
                    completed.add(key)

        # A retried failure leaves its old record before the new one.
        existing_records = self._deduplicate_records(existing_records)
        self._rewrite_records(output_path, existing_records)

        report = build_evaluation_report(
            records=existing_records,
            examples=examples,
            pipelines=pipelines,
            dataset_path=dataset_path,
            ragas_enabled=self.ragas_evaluator is not None,
        )
        write_report(report_path, report)
        return report

    @staticmethod
    def _deduplicate_records(
        records: list[EvaluationRecord],
    ) -> list[EvaluationRecord]:
        """Keep the newest checkpoint for each pipeline-question pair."""

        newest: dict[tuple[str, str], EvaluationRecord] = {}
        order: list[tuple[str, str]] = []
        for record in records:
            key = (record.output.pipeline, record.example.question_id)
            if key not in newest:
                order.append(key)
            newest[key] = record

        deduplicated: list[EvaluationRecord] = []
        for key in order:
            deduplicated.append(newest[key])
        return deduplicated

    @staticmethod
    def _rewrite_records(
        path: Path,
        records: list[EvaluationRecord],
    ) -> None:
        """Clean duplicate checkpoints left by an interrupted upgrade run."""

        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(
                        record.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _completed_keys(
        self,
        records: list[EvaluationRecord],
    ) -> set[tuple[str, str]]:
        """Return pairs that need no rerun.

        Records whose pipeline raised are retried, so an outage such as an
        unreachable LLM does not freeze zero scores into later reports.
        """

        keys: set[tuple[str, str]] = set()
        for record in records:
            if record.output.error:
                continue
            if self.ragas_evaluator is not None and record_needs_ragas(record):
                continue
            keys.add((record.output.pipeline, record.example.question_id))
        return keys

    @staticmethod
    def _load_records(path: Path) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        if not path.exists():
            return records

        with path.open("r", encoding="utf-8") as input_file:
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


def record_needs_ragas(record: EvaluationRecord) -> bool:
    """Return True when a produced answer still lacks usable RAGAS scores.

    A judge outage stores errors with every metric empty; such records are
    rescored instead of being treated as finished.
    """

    if record.output.error or not record.output.answer.strip():
        return False
    if record.ragas is None:
        return True

    values = record.ragas.model_dump(exclude={"errors"}).values()
    return bool(record.ragas.errors) and all(value is None for value in values)


def missing_ragas_metrics(
    record: EvaluationRecord,
    metric_names: list[str],
) -> list[str]:
    """Return the configured metrics this record still has to be judged on.

    A record without scores needs every metric. A record that was scored but
    lost some metrics to judge errors (for example a truncated faithfulness
    response) needs only those, so rescoring costs one call per gap rather
    than a full re-judgement.
    """

    if record.output.error or not record.output.answer.strip():
        return []
    if record.ragas is None:
        return list(metric_names)

    return [name for name in metric_names if name in record.ragas.errors]


def merge_ragas_scores(existing: RagasScores | None, update: RagasScores) -> RagasScores:
    """Overlay freshly judged metrics on saved scores, clearing fixed errors."""

    if existing is None:
        return update

    merged = existing.model_copy(deep=True)

    for name, value in update.model_dump(exclude={"errors"}).items():
        if value is not None:
            setattr(merged, name, value)
            merged.errors.pop(name, None)

    merged.errors.update(update.errors)
    return merged


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metric_means(
    records: list[EvaluationRecord], attribute: str
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Average each metric over the records where it is defined."""

    values_by_name: dict[str, list[float]] = {}
    for record in records:
        model = getattr(record, attribute)
        if model is None:
            continue
        for metric_name, value in model.model_dump().items():
            if metric_name == "errors" or value is None:
                continue
            values_by_name.setdefault(metric_name, []).append(float(value))

    means: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for metric_name, values in values_by_name.items():
        means[metric_name] = _mean(values)
        counts[metric_name] = len(values)
    return means, counts


def build_evaluation_report(
    records: list[EvaluationRecord],
    examples: list[EvaluationExample],
    pipelines: list[str],
    dataset_path: str,
    ragas_enabled: bool,
) -> EvaluationReport:
    """Aggregate comparable pipeline metrics while ignoring unavailable values."""

    expected_records = len(examples) * len(pipelines)
    relevant_question_ids: set[str] = set()
    for example in examples:
        relevant_question_ids.add(example.question_id)

    summaries: list[PipelineSummary] = []
    records_completed = 0
    ragas_records_completed = 0
    for pipeline in pipelines:
        selected: list[EvaluationRecord] = []
        for record in records:
            if record.output.pipeline != pipeline:
                continue
            if record.example.question_id not in relevant_question_ids:
                continue
            selected.append(record)

        records_completed += len(selected)
        failures = 0
        ragas_completed = 0
        latencies: list[float] = []
        status_distribution: dict[str, int] = {}
        for record in selected:
            if record.output.error:
                failures += 1
            if record.ragas is not None:
                ragas_completed += 1
            latencies.append(record.output.latency_seconds)
            status = record.output.final_status
            status_distribution[status] = status_distribution.get(status, 0) + 1

        # Quality metrics describe answers that were produced; a crashed run
        # has no retrieval or answer to judge. Failures are still reflected
        # in success, which is averaged over every record.
        answered = [record for record in selected if not record.output.error]
        deterministic_means, deterministic_counts = _metric_means(
            answered, "deterministic"
        )
        success_values = [record.deterministic.success for record in selected]
        deterministic_means["success"] = _mean(success_values)
        deterministic_counts["success"] = len(success_values)
        ragas_means, ragas_counts = _metric_means(answered, "ragas")

        summaries.append(
            PipelineSummary(
                pipeline=pipeline,
                total_examples=len(examples),
                completed_examples=len(selected),
                ragas_completed_examples=ragas_completed,
                failed_examples=failures,
                mean_latency_seconds=_mean(latencies),
                status_distribution=status_distribution,
                deterministic_means=deterministic_means,
                deterministic_counts=deterministic_counts,
                ragas_means=ragas_means,
                ragas_counts=ragas_counts,
            )
        )
        ragas_records_completed += ragas_completed

    records_are_complete = records_completed >= expected_records
    ragas_is_complete = not ragas_enabled or ragas_records_completed >= expected_records
    status = "complete" if records_are_complete and ragas_is_complete else "partial"
    return EvaluationReport(
        status=status,
        evaluated_at=datetime.now(UTC).isoformat(),
        dataset_path=dataset_path,
        requested_pipelines=pipelines,
        total_examples=len(examples),
        records_completed=records_completed,
        ragas_enabled=ragas_enabled,
        ragas_records_completed=ragas_records_completed,
        summaries=summaries,
    )


def write_report(path: str | Path, report: EvaluationReport) -> None:
    """Atomically replace the human-readable aggregate JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
