from agentic_rag.evaluation.metrics import score_deterministic_metrics
from agentic_rag.evaluation.models import (
    EvaluationExample,
    EvaluationRecord,
    PipelineOutput,
    RagasScores,
)
from agentic_rag.evaluation.ragas_runner import score_saved_records, write_records


class FakeRagasEvaluator:
    metric_names = ["faithfulness", "answer_relevancy"]

    def __init__(self):
        self.calls = 0
        self.requested_metrics: list[list[str]] = []

    async def evaluate(self, example, output, metric_names=None):
        self.calls += 1
        self.requested_metrics.append(list(metric_names or self.metric_names))
        scores = RagasScores()
        for name in metric_names or self.metric_names:
            setattr(scores, name, {"faithfulness": 0.8, "answer_relevancy": 0.9}[name])
        return scores


def make_record(ragas: RagasScores | None = None):
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
        ragas=ragas,
    )


def test_only_metrics_lost_to_judge_errors_are_rescored(tmp_path):
    records_path = tmp_path / "records.jsonl"
    partially_scored = make_record(
        RagasScores(
            answer_relevancy=0.5,
            errors={"faithfulness": "The output is incomplete due to a max_tokens length limit."},
        )
    )
    write_records(records_path, [partially_scored])
    evaluator = FakeRagasEvaluator()

    report = score_saved_records(
        records_path=records_path,
        report_path=tmp_path / "report.json",
        dataset_path="benchmark.jsonl",
        evaluator=evaluator,
    )

    assert evaluator.requested_metrics == [["faithfulness"]]
    assert report.status == "complete"
    saved = records_path.read_text(encoding="utf-8")
    assert '"faithfulness": 0.8' in saved
    # The score that was already there is kept, and the fixed error is cleared.
    assert '"answer_relevancy": 0.5' in saved
    assert '"errors": {}' in saved


def test_fully_scored_records_are_left_alone(tmp_path):
    records_path = tmp_path / "records.jsonl"
    write_records(
        records_path,
        [make_record(RagasScores(faithfulness=0.7, answer_relevancy=0.6))],
    )
    evaluator = FakeRagasEvaluator()

    score_saved_records(
        records_path=records_path,
        report_path=tmp_path / "report.json",
        dataset_path="benchmark.jsonl",
        evaluator=evaluator,
    )

    assert evaluator.calls == 0


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
