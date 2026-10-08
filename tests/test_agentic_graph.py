from datetime import date
from typing import Any

import pytest

from agentic_rag.graph.crag import CRAGAssessment
from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.query_rewriting import RewrittenQuery
from agentic_rag.graph.schemas import (
    AnswerCritique,
    GeneratedAnswer,
    HallucinationResult,
    RetrievalDecision,
)
from agentic_rag.graph.state import create_initial_state
from agentic_rag.graph.web_search import WebSearchResult
from agentic_rag.graph.workflow import build_agentic_rag_graph, mentions_time_window


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses

    def invoke(self, **_kwargs: Any) -> Any:
        if not self.responses:
            raise AssertionError("FakeLLM has no response remaining.")

        return self.responses.pop(0)


class FakeRetriever:
    pass


class FakeWebSearch:
    def __init__(self, result: WebSearchResult) -> None:
        self.result = result
        self.queries: list[str] = []
        self.date_filters: list[dict[str, str]] = []

    def search(self, query: str, **date_filter: str) -> WebSearchResult:
        self.queries.append(query)
        self.date_filters.append(date_filter)
        return self.result


class FakeRuntime:
    def __init__(
        self,
        llm: FakeLLM,
        web_search: FakeWebSearch | None = None,
    ) -> None:
        self.agent_config = {
            "retrieval_top_k": 10,
            "maximum_rewrite_attempts": 2,
            "maximum_generation_attempts": 2,
            "maximum_accumulated_documents": 12,
            "maximum_context_characters_per_document": 2400,
            "maximum_grader_characters_per_document": 1600,
            "maximum_rewrite_characters_per_document": 1200,
            "maximum_verifier_characters_per_document": 1800,
        }
        self.web_config = {
            "enabled": True,
            "maximum_combined_documents": 12,
        }
        self.llm = llm
        self.retriever = FakeRetriever()
        self.web_search = web_search

    def get_llm(self) -> FakeLLM:
        return self.llm

    def get_retriever(self) -> FakeRetriever:
        return self.retriever

    def get_web_search(self) -> FakeWebSearch:
        if self.web_search is None:
            raise AssertionError("Web search was not expected.")

        return self.web_search


def create_document(
    source_id: str = "source-1",
    source: str = "corpus",
) -> EvidenceDocument:
    return EvidenceDocument(
        source_id=source_id,
        paper_id="W1001",
        title="Corrective Retrieval-Augmented Generation",
        section_heading="Methods",
        text=(
            "The method detects retrieval failure and rewrites the "
            "query before retrieving again."
        ),
        source=source,
    )


def correct_assessment(
    document: EvidenceDocument,
) -> CRAGAssessment:
    graded_document = document.model_copy(
        update={
            "grade": "correct",
            "grade_reason": "The evidence answers the question.",
        }
    )

    return CRAGAssessment(
        route="correct",
        graded_documents=[graded_document],
        selected_documents=[graded_document],
        context_status="sufficient",
        context_reason="The evidence is sufficient.",
        missing_information="",
    )


def patch_successful_self_rag(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.generate_answer",
        lambda **_kwargs: (
            GeneratedAnswer(
                answer="CRAG rewrites failed queries [source-1].",
                source_ids=["source-1"],
            ),
            ["source-1"],
        ),
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.check_hallucination",
        lambda **_kwargs: HallucinationResult(
            grounded=True,
            reason="Every claim is supported.",
            unsupported_claims=[],
        ),
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.critique_answer",
        lambda **_kwargs: AnswerCritique(
            useful=True,
            needs_more_context=False,
            reason="The answer is complete.",
            improvement_feedback="",
        ),
    )


def test_initial_state_rejects_an_empty_question() -> None:
    try:
        create_initial_state("   ")
    except ValueError as error:
        assert "Question cannot be empty" in str(error)
    else:
        raise AssertionError("An empty question should fail.")


def test_graph_accepts_a_grounded_useful_answer(
    monkeypatch: Any,
) -> None:
    document = create_document()
    runtime = FakeRuntime(
        FakeLLM(
            [
                RetrievalDecision(
                    retrieve=True,
                    reason="Research evidence is required.",
                )
            ]
        )
    )

    monkeypatch.setattr(
        "agentic_rag.graph.workflow.retrieve_evidence",
        lambda **_kwargs: [document],
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.run_crag_assessment",
        lambda **_kwargs: correct_assessment(document),
    )
    patch_successful_self_rag(monkeypatch)

    graph = build_agentic_rag_graph(runtime)
    result = graph.invoke(create_initial_state("How does CRAG recover?"))

    assert result["final_status"] == "accepted"
    assert result["grounded"] is True
    assert result["useful"] is True
    assert result["source_ids"] == ["source-1"]
    assert len(result["citations"]) == 1


def test_graph_accepts_question_only_input_like_langgraph_studio(
    monkeypatch: Any,
) -> None:
    document = create_document()
    runtime = FakeRuntime(
        FakeLLM(
            [
                RetrievalDecision(
                    retrieve=True,
                    reason="Research evidence is required.",
                )
            ]
        )
    )
    retrieval_queries: list[str] = []

    def fake_retrieve(**kwargs: Any) -> list[EvidenceDocument]:
        retrieval_queries.append(kwargs["question"])
        return [document]

    monkeypatch.setattr(
        "agentic_rag.graph.workflow.retrieve_evidence",
        fake_retrieve,
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.run_crag_assessment",
        lambda **_kwargs: correct_assessment(document),
    )
    patch_successful_self_rag(monkeypatch)

    runtime.llm.responses[0].recent_days = 30

    graph = build_agentic_rag_graph(runtime)
    # Studio supplies only the question, plus leftovers from an earlier run
    # on the same thread.
    result = graph.invoke(
        {
            "question": "How does CRAG recover?",
            "retrieval_query": "a stale query from the previous question",
            "rewrite_count": 2,
        }
    )

    assert retrieval_queries == ["How does CRAG recover?"]
    assert result["final_status"] == "accepted"
    assert result["rewrite_count"] == 0
    # The router guessed a window, but the question has no time wording.
    assert result["recent_days"] is None


def test_incomplete_context_rewrites_and_retrieves_again(
    monkeypatch: Any,
) -> None:
    document = create_document()
    runtime = FakeRuntime(
        FakeLLM(
            [
                RetrievalDecision(
                    retrieve=True,
                    reason="Research evidence is required.",
                )
            ]
        )
    )
    assessment_calls = 0

    def fake_assessment(**_kwargs: Any) -> CRAGAssessment:
        nonlocal assessment_calls
        assessment_calls += 1

        if assessment_calls == 1:
            return CRAGAssessment(
                route="ambiguous",
                graded_documents=[document],
                selected_documents=[document],
                context_status="incomplete",
                context_reason="Recovery evidence is missing.",
                missing_information="How the query is corrected.",
            )

        return correct_assessment(document)

    monkeypatch.setattr(
        "agentic_rag.graph.workflow.retrieve_evidence",
        lambda **_kwargs: [document],
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.run_crag_assessment",
        fake_assessment,
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.rewrite_query",
        lambda **_kwargs: RewrittenQuery(
            rewritten_query="CRAG query rewriting recovery",
            reason="Search directly for the missing recovery step.",
        ),
    )
    patch_successful_self_rag(monkeypatch)

    graph = build_agentic_rag_graph(runtime)
    result = graph.invoke(
        create_initial_state("How does CRAG detect and correct retrieval failure?")
    )

    assert result["final_status"] == "accepted"
    assert result["rewrite_count"] == 1
    assert result["query_history"][-1] == ("CRAG query rewriting recovery")
    assert assessment_calls == 2


def test_ambiguous_evidence_is_answered_after_corrections_are_exhausted(
    monkeypatch: Any,
) -> None:
    document = create_document()
    web_search = FakeWebSearch(
        WebSearchResult(
            status="complete",
            query="CRAG recovery",
            documents=[create_document(source_id="web-source-1", source="web")],
        )
    )
    runtime = FakeRuntime(
        FakeLLM(
            [
                RetrievalDecision(
                    retrieve=True,
                    reason="Research evidence is required.",
                )
            ]
        ),
        web_search=web_search,
    )
    assessment_calls = 0

    def fake_assessment(**_kwargs: Any) -> CRAGAssessment:
        nonlocal assessment_calls
        assessment_calls += 1

        return CRAGAssessment(
            route="ambiguous",
            graded_documents=[document],
            selected_documents=[document],
            context_status="incomplete",
            context_reason="Part of the recovery step is missing.",
            missing_information="How the query is corrected.",
        )

    monkeypatch.setattr(
        "agentic_rag.graph.workflow.retrieve_evidence",
        lambda **_kwargs: [document],
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.run_crag_assessment",
        fake_assessment,
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.rewrite_query",
        lambda **_kwargs: RewrittenQuery(
            rewritten_query="CRAG query rewriting recovery",
            reason="Search directly for the missing recovery step.",
        ),
    )
    patch_successful_self_rag(monkeypatch)

    graph = build_agentic_rag_graph(runtime)
    result = graph.invoke(create_initial_state("How does CRAG recover?"))

    assert result["rewrite_count"] == 2
    assert result["web_search_used"] is True
    assert assessment_calls == 4
    assert result["final_status"] == "accepted"
    assert result["source_ids"] == ["source-1"]


def test_incorrect_context_uses_web_fallback(
    monkeypatch: Any,
) -> None:
    corpus_document = create_document()
    web_document = create_document(
        source_id="web-source-1",
        source="web",
    )
    web_search = FakeWebSearch(
        WebSearchResult(
            status="complete",
            query="CRAG recovery",
            documents=[web_document],
        )
    )
    runtime = FakeRuntime(
        FakeLLM(
            [
                RetrievalDecision(
                    retrieve=True,
                    reason="Research evidence is required.",
                )
            ]
        ),
        web_search=web_search,
    )
    assessment_calls = 0

    def fake_assessment(
        documents: list[EvidenceDocument],
        **_kwargs: Any,
    ) -> CRAGAssessment:
        nonlocal assessment_calls
        assessment_calls += 1

        if assessment_calls == 1:
            return CRAGAssessment(
                route="incorrect",
                graded_documents=documents,
                selected_documents=[],
                context_status="irrelevant",
                context_reason="Local evidence is irrelevant.",
                missing_information="Relevant CRAG recovery evidence.",
            )

        graded_web_document = web_document.model_copy(
            update={
                "source_id": "source-1",
                "grade": "correct",
                "grade_reason": "The web evidence is relevant.",
            }
        )

        return correct_assessment(graded_web_document)

    monkeypatch.setattr(
        "agentic_rag.graph.workflow.retrieve_evidence",
        lambda **_kwargs: [corpus_document],
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.run_crag_assessment",
        fake_assessment,
    )
    patch_successful_self_rag(monkeypatch)

    graph = build_agentic_rag_graph(runtime)
    result = graph.invoke(create_initial_state("How does CRAG recover?"))

    assert result["final_status"] == "accepted"
    assert result["web_search_used"] is True
    assert result["web_search_status"] == "complete"
    assert len(web_search.queries) == 1
    assert web_search.date_filters == [{}]
    assert assessment_calls == 2


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What agentic RAG research came out in the last seven days?", True),
        ("What is the latest corrective RAG benchmark?", True),
        ("Which RAG papers appeared in the past 3 months?", True),
        ("Compare how CRAG, Self-RAG, and adaptive RAG decide when to retrieve.", False),
        ("What context window does Llama 4 Scout support?", False),
    ],
)
def test_mentions_time_window(question: str, expected: bool) -> None:
    assert mentions_time_window(question) is expected


def test_time_bound_question_restricts_web_search_dates(
    monkeypatch: Any,
) -> None:
    web_document = create_document(source_id="web-source-1", source="web")
    web_search = FakeWebSearch(
        WebSearchResult(
            status="complete",
            query="agentic RAG",
            documents=[web_document],
        )
    )
    runtime = FakeRuntime(
        FakeLLM(
            [
                RetrievalDecision(
                    retrieve=True,
                    reason="Recent research is requested.",
                    recent_days=7,
                )
            ]
        ),
        web_search=web_search,
    )

    def fake_assessment(
        documents: list[EvidenceDocument],
        **_kwargs: Any,
    ) -> CRAGAssessment:
        if not web_search.queries:
            return CRAGAssessment(
                route="incorrect",
                graded_documents=documents,
                selected_documents=[],
                context_status="irrelevant",
                context_reason="No source is from the last seven days.",
                missing_information="Agentic RAG research from the last seven days.",
            )

        return correct_assessment(
            web_document.model_copy(update={"source_id": "source-1", "grade": "correct"})
        )

    monkeypatch.setattr(
        "agentic_rag.graph.workflow.retrieve_evidence",
        lambda **_kwargs: [create_document()],
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.run_crag_assessment",
        fake_assessment,
    )
    monkeypatch.setattr(
        "agentic_rag.graph.workflow.date",
        type("FixedDate", (), {"today": staticmethod(lambda: date(2026, 10, 8))}),
    )
    patch_successful_self_rag(monkeypatch)

    graph = build_agentic_rag_graph(runtime)
    result = graph.invoke({"question": "What agentic RAG research came out this week?"})

    assert result["recent_days"] == 7
    assert result["final_status"] == "accepted"
    assert web_search.date_filters == [{"start_date": "2026-10-01", "end_date": "2026-10-08"}]
