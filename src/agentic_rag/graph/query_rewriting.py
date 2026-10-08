from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.graph.baseline import retrieve_evidence
from agentic_rag.graph.crag import (
    CRAGAssessment,
    run_crag_assessment,
)
from agentic_rag.graph.documents import (
    EvidenceDocument,
    build_context,
    merge_evidence,
)
from agentic_rag.graph.schemas import RewrittenQuery
from agentic_rag.llm.ollama_client import OllamaStructuredClient


QUERY_REWRITER_SYSTEM_PROMPT = """
You rewrite search queries for a scientific Corrective RAG system.

The previous retrieval found some useful evidence, but it could not completely
answer the original question. Write one focused query for the missing evidence.

Rules:
- Preserve the meaning of the original question.
- Focus on the missing fact, comparison, mechanism, or reasoning step.
- Use useful entities and technical terms found in the current evidence.
- Make the rewritten query standalone.
- Do not answer the question.
- Do not invent facts or entities.
- Do not return multiple queries.
""".strip()


class CorrectiveRetrievalResult(BaseModel):
    """Evidence and diagnostics produced by corrective retrieval."""

    original_question: str
    final_query: str
    query_history: list[str] = Field(default_factory=list)
    rewrite_reasons: list[str] = Field(default_factory=list)
    rewrite_count: int = 0
    documents: list[EvidenceDocument] = Field(default_factory=list)
    assessment: CRAGAssessment


def select_corrective_evidence(
    existing: list[EvidenceDocument],
    incoming: list[EvidenceDocument],
    maximum_documents: int = 12,
) -> list[EvidenceDocument]:
    """Combine graded evidence with a new retrieval round.

    Order is: existing chunks graded correct, then newly retrieved chunks,
    then remaining existing chunks. New evidence ranks above ambiguous
    evidence so the cap cannot crowd out what the rewritten query found.
    """

    if maximum_documents <= 0:
        raise ValueError("maximum_documents must be positive.")

    useful_existing = [
        document for document in existing if document.grade != "incorrect"
    ]
    confirmed = [
        document for document in useful_existing if document.grade == "correct"
    ]
    unconfirmed = [
        document for document in useful_existing if document.grade != "correct"
    ]

    combined = merge_evidence(existing=confirmed, incoming=incoming)
    combined = merge_evidence(existing=combined, incoming=unconfirmed)

    return combined[:maximum_documents]


def normalize_query(query: str) -> str:
    """Normalize whitespace and case for query comparison."""

    return " ".join(query.lower().split())


def build_rewrite_prompt(
    question: str,
    current_query: str,
    documents: list[EvidenceDocument],
    missing_information: str,
    maximum_characters_per_document: int = 1200,
) -> str:
    """Build a query-rewriting prompt from the current evidence gap."""

    if documents:
        evidence_clues = build_context(
            documents=documents,
            maximum_characters_per_document=(maximum_characters_per_document),
        )
    else:
        evidence_clues = "No useful evidence clues are available."

    missing_text = missing_information.strip()

    if not missing_text:
        missing_text = "More specific scientific evidence for the question."

    return (
        "Original question:\n"
        + question.strip()
        + "\n\nPrevious retrieval query:\n"
        + current_query.strip()
        + "\n\nMissing information:\n"
        + missing_text
        + "\n\nUseful evidence clues:\n"
        + evidence_clues
    )


def create_fallback_query(
    current_query: str,
    missing_information: str,
) -> str:
    """Create a deterministic query if the model repeats the old query."""

    missing_text = missing_information.strip()

    if not missing_text:
        missing_text = "additional supporting evidence"

    return current_query.strip() + " " + missing_text


def rewrite_query(
    question: str,
    current_query: str,
    documents: list[EvidenceDocument],
    missing_information: str,
    llm: OllamaStructuredClient,
    maximum_characters_per_document: int = 1200,
) -> RewrittenQuery:
    """Ask Ollama for one focused query targeting missing evidence."""

    prompt = build_rewrite_prompt(
        question=question,
        current_query=current_query,
        documents=documents,
        missing_information=missing_information,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    result = llm.invoke(
        system_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=RewrittenQuery,
    )

    if normalize_query(result.rewritten_query) == normalize_query(current_query):
        fallback_query = create_fallback_query(
            current_query=current_query,
            missing_information=missing_information,
        )

        return RewrittenQuery(
            rewritten_query=fallback_query,
            reason=(
                "The model repeated the previous query, so the "
                "missing-information description was appended."
            ),
        )

    return result


def run_corrective_retrieval(
    question: str,
    retriever: Any,
    llm: OllamaStructuredClient,
    top_k: int = 10,
    maximum_rewrite_attempts: int = 2,
    maximum_grade_characters_per_document: int = 1600,
    maximum_context_characters_per_document: int = 2000,
    maximum_rewrite_characters_per_document: int = 1200,
    maximum_accumulated_documents: int = 12,
) -> CorrectiveRetrievalResult:
    """Retrieve, grade, rewrite incomplete queries, and retrieve again."""

    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    if maximum_rewrite_attempts < 0:
        raise ValueError("maximum_rewrite_attempts cannot be negative.")

    current_query = clean_question
    query_history = [current_query]
    rewrite_reasons: list[str] = []

    documents = retrieve_evidence(
        question=current_query,
        retriever=retriever,
        top_k=top_k,
    )

    assessment = run_crag_assessment(
        question=clean_question,
        documents=documents,
        llm=llm,
        maximum_grade_characters_per_document=(maximum_grade_characters_per_document),
        maximum_context_characters_per_document=(maximum_context_characters_per_document),
    )

    rewrite_count = 0

    while assessment.route == "ambiguous" and rewrite_count < maximum_rewrite_attempts:
        rewritten = rewrite_query(
            question=clean_question,
            current_query=current_query,
            documents=assessment.selected_documents,
            missing_information=assessment.missing_information,
            llm=llm,
            maximum_characters_per_document=(maximum_rewrite_characters_per_document),
        )

        current_query = rewritten.rewritten_query.strip()
        query_history.append(current_query)
        rewrite_reasons.append(rewritten.reason)
        rewrite_count += 1

        new_documents = retrieve_evidence(
            question=current_query,
            retriever=retriever,
            top_k=top_k,
        )

        combined_documents = select_corrective_evidence(
            existing=assessment.selected_documents,
            incoming=new_documents,
            maximum_documents=maximum_accumulated_documents,
        )

        assessment = run_crag_assessment(
            question=clean_question,
            documents=combined_documents,
            llm=llm,
            maximum_grade_characters_per_document=(maximum_grade_characters_per_document),
            maximum_context_characters_per_document=(maximum_context_characters_per_document),
        )

    return CorrectiveRetrievalResult(
        original_question=clean_question,
        final_query=current_query,
        query_history=query_history,
        rewrite_reasons=rewrite_reasons,
        rewrite_count=rewrite_count,
        documents=assessment.selected_documents,
        assessment=assessment,
    )
