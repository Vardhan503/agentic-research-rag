from agentic_rag.evaluation.metrics import score_deterministic_metrics
from agentic_rag.evaluation.models import (
    EvaluationExample,
    EvaluationRecord,
    PipelineOutput,
    RagasScores,
)
from agentic_rag.evaluation.ragas_runner import score_saved_records, write_records


class FakeRagasEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, example, output):
        self.calls += 1
        return RagasScores(faithfulness=0.8, answer_relevancy=0.9)


def make_record():
    example = EvaluationExample(
        question_id="q1",
        question="What is RAG?",
        reference_answer="RAG retrieves evidence.",
        category="rag",
        difficulty="easy",
    )
    output = PipelineOutput(
        pipeline="baseline",
        question_id="q1",
        answer="RAG retrieves evidence.",
        contexts=["RAG retrieves evidence."],
        retrieval_used=True,
        final_status="accepted",
    )
    return EvaluationRecord(
        example=example,
        output=output,
        deterministic=score_deterministic_metrics(example, output),
    )


def test_score_saved_records_checkpoints_without_rerunning_pipeline(tmp_path):
    records_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    write_records(records_path, [make_record()])
    evaluator = FakeRagasEvaluator()

    report = score_saved_records(
        records_path=records_path,
        report_path=report_path,
        dataset_path="benchmark.jsonl",
        evaluator=evaluator,
    )

    assert evaluator.calls == 1
    assert report.status == "complete"
    assert report.ragas_records_completed == 1
    assert '"faithfulness": 0.8' in records_path.read_text(encoding="utf-8")
