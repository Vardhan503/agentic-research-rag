from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic_rag.graph.crag import (
    CRAGAssessment,
    run_crag_assessment,
)
from agentic_rag.graph.documents import (
    EvidenceDocument,
    merge_evidence,
)
from agentic_rag.graph.query_rewriting import (
    CorrectiveRetrievalResult,
)
from agentic_rag.graph.web_search import (
    TavilyWebSearch,
    WebSearchResult,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient


class WebFallbackResult(BaseModel):
    """Result of optionally supplementing corpus evidence from the web."""

    used: bool
    status: Literal[
        "not_needed",
        "complete",
        "empty",
        "unavailable",
        "failed",
    ]
    query: str
    documents: list[EvidenceDocument] = Field(default_factory=list)
    web_documents: list[EvidenceDocument] = Field(default_factory=list)
    assessment: CRAGAssessment
    error: str = ""


def build_web_query(
    question: str,
    corrective_result: CorrectiveRetrievalResult,
) -> str:
    """Build a focused external-search query from the remaining gap."""

    query = corrective_result.final_query.strip()
    missing_information = corrective_result.assessment.missing_information.strip()

    if not query:
        query = question.strip()

    if missing_information:
        normalized_query = " ".join(query.lower().split())
        normalized_missing = " ".join(missing_information.lower().split())

        if normalized_missing not in normalized_query:
            query = query + " " + missing_information

    return query.strip()


def combine_web_and_corpus_evidence(
    corpus_documents: list[EvidenceDocument],
    web_documents: list[EvidenceDocument],
    maximum_documents: int = 12,
) -> list[EvidenceDocument]:
    """Keep a balanced corpus/web evidence set within the LLM budget."""

    if maximum_documents <= 0:
        raise ValueError("maximum_documents must be positive.")

    corpus_budget = maximum_documents // 2
    web_budget = maximum_documents - corpus_budget

    selected_corpus = corpus_documents[:corpus_budget]
    selected_web = web_documents[:web_budget]

    combined = merge_evidence(
        existing=selected_corpus,
        incoming=selected_web,
    )

    if len(combined) >= maximum_documents:
        return combined[:maximum_documents]

    all_documents = merge_evidence(
        existing=corpus_documents,
        incoming=web_documents,
    )

    return merge_evidence(
        existing=combined,
        incoming=all_documents,
    )[:maximum_documents]


def unchanged_assessment(
    corrective_result: CorrectiveRetrievalResult,
) -> CRAGAssessment:
    """Return the existing assessment without mutating it."""

    return corrective_result.assessment.model_copy(deep=True)


def run_web_fallback(
    question: str,
    corrective_result: CorrectiveRetrievalResult,
    web_search: TavilyWebSearch,
    llm: OllamaStructuredClient,
    enabled: bool = True,
    maximum_documents: int = 12,
    maximum_grade_characters_per_document: int = 1600,
    maximum_context_characters_per_document: int = 2000,
) -> WebFallbackResult:
    """Use and regrade web evidence when local evidence is insufficient."""

    existing_assessment = unchanged_assessment(corrective_result)

    if corrective_result.assessment.route == "correct":
        return WebFallbackResult(
            used=False,
            status="not_needed",
            query=corrective_result.final_query,
            documents=corrective_result.documents,
            assessment=existing_assessment,
        )

    web_query = build_web_query(
        question=question,
        corrective_result=corrective_result,
    )

    if not enabled:
        return WebFallbackResult(
            used=False,
            status="unavailable",
            query=web_query,
            documents=corrective_result.documents,
            assessment=existing_assessment,
            error="Web fallback is disabled in configuration.",
        )

    search_result: WebSearchResult = web_search.search(web_query)

    if search_result.status != "complete":
        return WebFallbackResult(
            used=True,
            status=search_result.status,
            query=web_query,
            documents=corrective_result.documents,
            assessment=existing_assessment,
            error=search_result.error,
        )

    combined_documents = combine_web_and_corpus_evidence(
        corpus_documents=corrective_result.documents,
        web_documents=search_result.documents,
        maximum_documents=maximum_documents,
    )

    assessment = run_crag_assessment(
        question=question,
        documents=combined_documents,
        llm=llm,
        maximum_grade_characters_per_document=(maximum_grade_characters_per_document),
        maximum_context_characters_per_document=(maximum_context_characters_per_document),
    )

    return WebFallbackResult(
        used=True,
        status="complete",
        query=web_query,
        documents=assessment.selected_documents,
        web_documents=search_result.documents,
        assessment=assessment,
    )
