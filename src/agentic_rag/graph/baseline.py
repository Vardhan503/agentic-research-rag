from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.graph.documents import (
    EvidenceDocument,
    build_context,
    convert_retrieval_results,
)
from agentic_rag.graph.schemas import GeneratedAnswer
from agentic_rag.llm.ollama_client import OllamaStructuredClient


GENERATOR_SYSTEM_PROMPT = """
You are a scientific research assistant using retrieval-augmented generation.

Answer the question using only the supplied research-paper evidence.

Rules:
- Do not use outside knowledge.
- Do not invent facts, methods, results, or citations.
- Cite supported claims using the exact Source ID in square brackets.
- A citation must look like [SOURCE_ID].
- Return only Source IDs that appear in the supplied evidence.
- Prefer evidence from several papers when the question asks for a comparison.
- Clearly state when the evidence is insufficient.
- Keep the answer focused, readable, and scientifically cautious.
""".strip()


class Citation(BaseModel):
    """Metadata for one source cited by the generated answer."""

    source_id: str
    paper_id: str
    title: str
    section_heading: str = ""
    doi: str | None = None
    url: str | None = None


class BaselineRAGResult(BaseModel):
    """Complete output of one baseline RAG request."""

    question: str
    answer: str
    retrieval_query: str
    source_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    documents: list[EvidenceDocument] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


def build_generation_prompt(
    question: str,
    documents: list[EvidenceDocument],
    maximum_characters_per_document: int,
) -> str:
    """Create the question and labelled evidence prompt."""

    if not documents:
        raise ValueError(
            "At least one evidence document is required."
        )

    context = build_context(
        documents=documents,
        maximum_characters_per_document=(
            maximum_characters_per_document
        ),
    )

    return (
        "Question:\n"
        + question.strip()
        + "\n\nResearch-paper evidence:\n"
        + context
    )


def validate_generated_answer(
    generated: GeneratedAnswer,
    documents: list[EvidenceDocument],
) -> list[str]:
    """Verify that every returned citation belongs to retrieved evidence."""

    available_source_ids: set[str] = set()

    for document in documents:
        available_source_ids.add(document.source_id)

    valid_source_ids: list[str] = []
    seen_source_ids: set[str] = set()

    for source_id in generated.source_ids:
        if source_id not in available_source_ids:
            raise ValueError(
                "The model returned an unknown source ID: "
                + source_id
            )

        if source_id in seen_source_ids:
            continue

        citation_text = "[" + source_id + "]"

        if citation_text not in generated.answer:
            raise ValueError(
                "The answer did not contain its declared citation: "
                + citation_text
            )

        seen_source_ids.add(source_id)
        valid_source_ids.append(source_id)

    if not valid_source_ids:
        raise ValueError(
            "The grounded answer must cite at least one retrieved source."
        )

    return valid_source_ids


def build_citations(
    source_ids: list[str],
    documents: list[EvidenceDocument],
) -> list[Citation]:
    """Create user-facing citation metadata in answer citation order."""

    document_index: dict[str, EvidenceDocument] = {}

    for document in documents:
        document_index[document.source_id] = document

    citations: list[Citation] = []

    for source_id in source_ids:
        document = document_index[source_id]

        citation = Citation(
            source_id=document.source_id,
            paper_id=document.paper_id,
            title=document.title,
            section_heading=document.section_heading,
            doi=document.doi,
            url=document.url,
        )

        citations.append(citation)

    return citations


def retrieve_evidence(
    question: str,
    retriever: Any,
    top_k: int,
) -> list[EvidenceDocument]:
    """Run hybrid retrieval and convert its results into evidence."""

    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    candidates = retriever.search(
        query=clean_question,
        top_k=top_k,
        use_reranker=True,
    )

    return convert_retrieval_results(candidates)


def generate_grounded_answer(
    question: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    maximum_characters_per_document: int = 2400,
) -> tuple[GeneratedAnswer, list[str]]:
    """Generate one answer and validate its exact chunk citations."""

    prompt = build_generation_prompt(
        question=question,
        documents=documents,
        maximum_characters_per_document=(
            maximum_characters_per_document
        ),
    )

    generated = llm.invoke(
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=GeneratedAnswer,
    )

    source_ids = validate_generated_answer(
        generated=generated,
        documents=documents,
    )

    return generated, source_ids


def run_baseline_rag(
    question: str,
    retriever: Any,
    llm: OllamaStructuredClient,
    top_k: int = 10,
    maximum_characters_per_document: int = 2400,
) -> BaselineRAGResult:
    """Retrieve evidence and return one grounded, cited answer."""

    started_at = time.perf_counter()
    clean_question = question.strip()

    documents = retrieve_evidence(
        question=clean_question,
        retriever=retriever,
        top_k=top_k,
    )

    if not documents:
        elapsed_seconds = time.perf_counter() - started_at

        return BaselineRAGResult(
            question=clean_question,
            answer=(
                "No relevant evidence was found in the local "
                "research corpus."
            ),
            retrieval_query=clean_question,
            elapsed_seconds=elapsed_seconds,
        )

    generated, source_ids = generate_grounded_answer(
        question=clean_question,
        documents=documents,
        llm=llm,
        maximum_characters_per_document=(
            maximum_characters_per_document
        ),
    )

    citations = build_citations(
        source_ids=source_ids,
        documents=documents,
    )

    elapsed_seconds = time.perf_counter() - started_at

    return BaselineRAGResult(
        question=clean_question,
        answer=generated.answer,
        retrieval_query=clean_question,
        source_ids=source_ids,
        citations=citations,
        documents=documents,
        elapsed_seconds=elapsed_seconds,
    )
