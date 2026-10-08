from __future__ import annotations

from typing import Any, TypedDict

from agentic_rag.graph.documents import EvidenceDocument


class AgenticRAGState(TypedDict, total=False):
    """Shared information passed between all LangGraph nodes."""

    question: str

    retrieval_needed: bool
    router_reason: str
    recent_days: int | None
    retrieval_query: str

    documents: list[dict[str, Any]]
    graded_documents: list[dict[str, Any]]

    crag_route: str
    context_status: str
    context_reason: str
    missing_information: str

    query_history: list[str]
    rewrite_reasons: list[str]
    rewrite_count: int

    web_search_used: bool
    web_search_status: str
    web_search_error: str

    answer: str
    source_ids: list[str]
    citations: list[dict[str, Any]]

    generation_count: int
    grounded: bool
    useful: bool
    needs_more_context: bool
    hallucination_reason: str
    unsupported_claims: list[str]
    critic_reason: str
    improvement_feedback: str

    final_status: str


def create_initial_state(question: str) -> AgenticRAGState:
    """Create a clean state for one user question."""

    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    return AgenticRAGState(
        question=clean_question,
        retrieval_needed=True,
        router_reason="",
        recent_days=None,
        retrieval_query=clean_question,
        documents=[],
        graded_documents=[],
        crag_route="",
        context_status="",
        context_reason="",
        missing_information="",
        query_history=[clean_question],
        rewrite_reasons=[],
        rewrite_count=0,
        web_search_used=False,
        web_search_status="not_used",
        web_search_error="",
        answer="",
        source_ids=[],
        citations=[],
        generation_count=0,
        grounded=False,
        useful=False,
        needs_more_context=False,
        hallucination_reason="",
        unsupported_claims=[],
        critic_reason="",
        improvement_feedback="",
        final_status="running",
    )


def documents_to_state(
    documents: list[EvidenceDocument],
) -> list[dict[str, Any]]:
    """Convert evidence objects into JSON-serializable dictionaries."""

    serialized_documents: list[dict[str, Any]] = []

    for document in documents:
        serialized_documents.append(document.model_dump(mode="json"))

    return serialized_documents


def documents_from_state(
    records: list[dict[str, Any]],
) -> list[EvidenceDocument]:
    """Recreate evidence objects from graph-state dictionaries."""

    documents: list[EvidenceDocument] = []

    for record in records:
        documents.append(EvidenceDocument.model_validate(record))

    return documents
