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
