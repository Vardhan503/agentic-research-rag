import asyncio

from agentic_rag.evaluation.models import EvaluationExample, PipelineOutput
import pytest

from agentic_rag.evaluation.ragas_evaluator import (
    DEFAULT_JUDGE_MAX_TOKENS,
    RagasEvaluator,
    judge_model_args,
)


class FakeMetricResult:
    def __init__(self, value):
        self.value = value


class FakeMetric:
    def __init__(self, value):
        self.value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMetricResult(self.value)


def make_example(reference_mode="static"):
    return EvaluationExample(
        question_id="q1",
        question="What is RAG?",
        reference_answer="RAG retrieves evidence.",
        category="rag",
        difficulty="easy",
        reference_mode=reference_mode,
    )


def make_output():
    return PipelineOutput(
        pipeline="baseline",
        question_id="q1",
        answer="RAG retrieves evidence.",
        contexts=["RAG retrieves evidence."],
        retrieval_used=True,
        final_status="accepted",
    )


def test_judge_model_args_raise_the_default_token_budget():
    assert judge_model_args({}) == {
        "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
        "temperature": 0.0,
    }
    assert judge_model_args({"judge_max_tokens": 8192})["max_tokens"] == 8192

    with pytest.raises(ValueError, match="judge_max_tokens"):
        judge_model_args({"judge_max_tokens": 0})


def test_ragas_evaluator_scores_all_static_metrics_without_network():
    evaluator = RagasEvaluator(
        {
            "metrics": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "factual_correctness",
            ]
        }
    )
    evaluator._metrics = {
        "faithfulness": FakeMetric(0.91),
        "answer_relevancy": FakeMetric(0.92),
        "context_precision": FakeMetric(0.93),
        "context_recall": FakeMetric(0.94),
        "factual_correctness": FakeMetric(0.95),
    }

    scores = asyncio.run(evaluator.evaluate(make_example(), make_output()))

    assert scores.faithfulness == 0.91
    assert scores.answer_relevancy == 0.92
    assert scores.context_precision == 0.93
    assert scores.context_recall == 0.94
    assert scores.factual_correctness == 0.95
    assert scores.errors == {}


def test_dynamic_reference_skips_reference_dependent_metrics():
    evaluator = RagasEvaluator(
        {
            "metrics": [
                "context_precision",
                "context_recall",
                "factual_correctness",
            ]
        }
    )
    evaluator._metrics = {
        "context_precision": FakeMetric(0.1),
        "context_recall": FakeMetric(0.1),
        "factual_correctness": FakeMetric(0.1),
    }

    scores = asyncio.run(
        evaluator.evaluate(make_example(reference_mode="dynamic"), make_output())
    )

    assert scores.context_precision is None
    assert scores.context_recall is None
    assert scores.factual_correctness is None
