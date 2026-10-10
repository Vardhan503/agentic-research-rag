import inspect

from agentic_rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    score_deterministic_metrics,
)
from agentic_rag.evaluation.models import (
    EvaluationExample,
    EvaluationRecord,
    PipelineOutput,
    RagasScores,
)
from agentic_rag.evaluation.runner import (
    EvaluationRunner,
    build_evaluation_report,
    record_needs_ragas,
)
from agentic_rag.graph.query_rewriting import run_corrective_retrieval


def make_example(question_id="q1", expected_paper_ids=None):
    return EvaluationExample(
        question_id=question_id,
        question="What is RAG?",
        reference_answer="RAG retrieves evidence.",
        category="rag",
        difficulty="easy",
        expected_paper_ids=expected_paper_ids or ["W1"],
        must_contain=["retrieves"],
    )


def answered_output(question_id="q1"):
    return PipelineOutput(
        pipeline="baseline",
        question_id=question_id,
        answer="RAG retrieves evidence [s1].",
        contexts=["Evidence."],
        retrieved_source_ids=["s1"],
        retrieved_paper_ids=["W1"],
        cited_source_ids=["s1"],
        cited_paper_ids=["W1"],
        final_status="accepted",
        retrieval_used=True,
    )


def failed_output(question_id="q1"):
    return PipelineOutput(
        pipeline="baseline",
        question_id=question_id,
        final_status="failed",
        error="RuntimeError: Failed to connect to Ollama.",
    )


def make_record(example, output, ragas=None):
    return EvaluationRecord(
        example=example,
        output=output,
        deterministic=score_deterministic_metrics(example, output),
        ragas=ragas,
    )


def test_repeated_paper_ids_do_not_inflate_ranking_metrics():
    # Three chunks of one relevant paper used to give nDCG of about 2.13.
    retrieved = ["W1", "W1", "W1", "W2"]

    assert ndcg_at_k(retrieved, ["W1"], k=10) == 1.0
    assert precision_at_k(retrieved, ["W1"], k=10) == 0.5
    assert reciprocal_rank(["W2", "W2", "W1"], ["W1"]) == 0.5


def test_failed_records_are_retried_on_resume(tmp_path):
    class RecoveringRunner:
        def __init__(self):
            self.calls = 0

        def run(self, pipeline, example):
            self.calls += 1
            if self.calls == 1:
                return failed_output(example.question_id)
            return answered_output(example.question_id)

    pipeline_runner = RecoveringRunner()
    runner = EvaluationRunner(pipeline_runner=pipeline_runner)
    paths = {
        "records_path": tmp_path / "records.jsonl",
        "report_path": tmp_path / "report.json",
        "dataset_path": "benchmark.jsonl",
    }

    first = runner.run(examples=[make_example()], pipelines=["baseline"], **paths)
    second = runner.run(examples=[make_example()], pipelines=["baseline"], **paths)

    assert pipeline_runner.calls == 2
    assert first.summaries[0].failed_examples == 1
    assert second.summaries[0].failed_examples == 0
    assert second.summaries[0].deterministic_means["recall_at_k"] == 1.0
    assert len(paths["records_path"].read_text().strip().splitlines()) == 1


def test_failures_count_in_success_but_not_quality_means():
    records = [
        make_record(make_example("q1"), answered_output("q1")),
        make_record(make_example("q2"), failed_output("q2")),
    ]

    report = build_evaluation_report(
        records=records,
        examples=[make_example("q1"), make_example("q2")],
        pipelines=["baseline"],
        dataset_path="benchmark.jsonl",
        ragas_enabled=False,
    )
    summary = report.summaries[0]

    assert summary.deterministic_means["recall_at_k"] == 1.0
    assert summary.deterministic_counts["recall_at_k"] == 1
    assert summary.deterministic_means["success"] == 0.5
    assert summary.deterministic_counts["success"] == 2


def test_judge_outage_records_are_rescored():
    example = make_example()

    assert record_needs_ragas(make_record(example, answered_output()))
    assert record_needs_ragas(
        make_record(
            example,
            answered_output(),
            RagasScores(errors={"faithfulness": "Connection refused"}),
        )
    )
    assert not record_needs_ragas(
        make_record(example, answered_output(), RagasScores(faithfulness=0.9))
    )
    assert not record_needs_ragas(make_record(example, failed_output()))


def test_corrective_retrieval_accepts_evaluation_pipeline_arguments():
    # pipelines.py passes these keywords; a missing one crashed every CRAG run.
    inspect.signature(run_corrective_retrieval).bind(
        question="q",
        retriever=None,
        llm=None,
        top_k=10,
        maximum_rewrite_attempts=2,
        maximum_grade_characters_per_document=1600,
        maximum_context_characters_per_document=2000,
        maximum_rewrite_characters_per_document=1200,
        maximum_accumulated_documents=12,
    )
