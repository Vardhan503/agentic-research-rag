import pytest

from agentic_rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_deterministic_metrics,
)
from agentic_rag.evaluation.models import EvaluationExample, PipelineOutput


def test_retrieval_metrics_use_ranked_relevance():
    retrieved = ["noise", "W2", "W1"]
    relevant = ["https://openalex.org/W1", "W2"]

    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.693426, rel=1e-5)


def test_retrieval_metrics_are_none_without_gold_ids():
    assert precision_at_k(["W1"], [], 10) is None
    assert recall_at_k(["W1"], [], 10) is None
    assert reciprocal_rank(["W1"], []) is None
    assert ndcg_at_k(["W1"], [], 10) is None


def test_deterministic_scores_cover_citations_keywords_routes_and_success():
    example = EvaluationExample(
        question_id="q1",
        question="Explain hybrid retrieval.",
        reference_answer="It combines dense and sparse retrieval.",
        category="retrieval",
        difficulty="easy",
        expected_paper_ids=["W1"],
        must_contain=["dense", "sparse"],
    )
    output = PipelineOutput(
        pipeline="agentic",
        question_id="q1",
        answer="Dense and sparse signals are combined [chunk-1].",
        retrieved_source_ids=["chunk-1"],
        retrieved_paper_ids=["W1"],
        cited_source_ids=["chunk-1"],
        cited_paper_ids=["W1"],
        final_status="accepted",
        retrieval_used=True,
    )

    scores = score_deterministic_metrics(example, output)

    assert scores.recall_at_k == 1.0
    assert scores.citation_validity == 1.0
    assert scores.citation_recall == 1.0
    assert scores.keyword_coverage == 1.0
    assert scores.route_accuracy == 1.0
    assert scores.success == 1.0


def test_unanswerable_question_counts_abstention_as_success():
    example = EvaluationExample(
        question_id="q-fake",
        question="What did the paper 'Quantum Banana Retrieval' conclude?",
        reference_answer="No such paper exists; the system should abstain.",
        category="unanswerable",
        difficulty="hard",
        expects_abstention=True,
    )

    def output(final_status: str, answer: str) -> PipelineOutput:
        return PipelineOutput(
            pipeline="agentic",
            question_id="q-fake",
            answer=answer,
            final_status=final_status,
            retrieval_used=True,
        )

    abstained = score_deterministic_metrics(
        example, output("insufficient_evidence", "There is not enough supported evidence.")
    )
    invented = score_deterministic_metrics(
        example, output("accepted", "The paper concluded ZB-17 is state of the art.")
    )

    assert abstained.success == 1.0
    assert invented.success == 0.0


def test_time_bound_question_accepts_an_answer_or_an_abstention():
    example = EvaluationExample(
        question_id="time-001",
        question="What agentic RAG research was published in the last seven days?",
        reference_answer="Name a source inside the window, or abstain.",
        category="web_fallback",
        difficulty="hard",
        accepts_abstention=True,
    )

    def output(final_status: str) -> PipelineOutput:
        return PipelineOutput(
            pipeline="agentic",
            question_id="time-001",
            answer="There is not enough supported evidence to answer the question reliably.",
            final_status=final_status,
            retrieval_used=True,
            web_search_used=True,
        )

    assert score_deterministic_metrics(example, output("insufficient_evidence")).success == 1.0
    assert score_deterministic_metrics(example, output("accepted")).success == 1.0
    assert score_deterministic_metrics(example, output("needs_more_context")).success == 0.0


def test_route_accuracy_detects_unnecessary_web_search():
    example = EvaluationExample(
        question_id="q1",
        question="Explain RAG.",
        reference_answer="RAG retrieves evidence.",
        category="rag",
        difficulty="easy",
    )
    output = PipelineOutput(
        pipeline="agentic",
        question_id="q1",
        retrieval_used=True,
        web_search_used=True,
    )

    scores = score_deterministic_metrics(example, output)

    assert scores.route_accuracy == 0.5


def test_forbidden_phrase_zeros_keyword_coverage():
    example = EvaluationExample(
        question_id="time-001",
        question="What agentic RAG research was published in the last seven days?",
        reference_answer="Only sources inside the last seven days count.",
        category="web_fallback",
        difficulty="hard",
        must_contain=["seven days"],
        must_not_contain=["2024-08-18", "Time Series Analysis"],
    )
    stale = PipelineOutput(
        pipeline="baseline",
        question_id="time-001",
        answer=(
            "Recent work published in the last seven days includes "
            "Agentic Retrieval-Augmented Generation for Time Series Analysis (2024-08-18)."
        ),
        final_status="accepted",
    )
    honest = PipelineOutput(
        pipeline="agentic",
        question_id="time-001",
        answer="No agentic RAG papers from the last seven days were confirmed.",
        final_status="accepted",
    )

    assert score_deterministic_metrics(example, stale).keyword_coverage == pytest.approx(1 / 3)
    assert score_deterministic_metrics(example, honest).keyword_coverage == 1.0
