import json

from agentic_rag.evaluation.models import EvaluationExample, PipelineOutput
from agentic_rag.evaluation.runner import EvaluationRunner


class FakePipelineRunner:
    def __init__(self):
        self.calls = 0

    def run(self, pipeline, example):
        self.calls += 1
        return PipelineOutput(
            pipeline=pipeline,
            question_id=example.question_id,
            answer="A grounded answer.",
            contexts=["Evidence."],
            retrieved_source_ids=["source-1"],
            retrieved_paper_ids=["W1"],
            cited_source_ids=["source-1"],
            cited_paper_ids=["W1"],
            final_status="accepted",
            retrieval_used=True,
            latency_seconds=0.2,
        )


def make_example():
    return EvaluationExample(
        question_id="q1",
        question="What is RAG?",
        reference_answer="RAG retrieves evidence.",
        category="rag",
        difficulty="easy",
        expected_paper_ids=["W1"],
    )


def test_evaluation_runner_writes_report_and_resumes(tmp_path):
    pipeline_runner = FakePipelineRunner()
    runner = EvaluationRunner(pipeline_runner=pipeline_runner)
    records_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"

    first_report = runner.run(
        examples=[make_example()],
        pipelines=["baseline"],
        records_path=records_path,
        report_path=report_path,
        dataset_path="benchmark.jsonl",
    )
    second_report = runner.run(
        examples=[make_example()],
        pipelines=["baseline"],
        records_path=records_path,
        report_path=report_path,
        dataset_path="benchmark.jsonl",
    )

    assert pipeline_runner.calls == 1
    assert first_report.status == "complete"
    assert second_report.records_completed == 1
    assert len(records_path.read_text(encoding="utf-8").splitlines()) == 1
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["summaries"][0]["deterministic_means"]["recall_at_k"] == 1.0
