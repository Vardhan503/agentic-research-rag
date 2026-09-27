from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.graph.baseline import (
    build_citations,
    build_generation_prompt,
    run_baseline_rag,
    validate_generated_answer,
)
from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.schemas import GeneratedAnswer
from agentic_rag.llm.ollama_client import OllamaStructuredClient
from agentic_rag.processing.models import DocumentChunk
from agentic_rag.retrieval.models import RetrievalCandidate


class FakeRetriever:
    def __init__(self, results: list[RetrievalCandidate]) -> None:
        self.results = results
        self.received_query = ""

    def search(
        self,
        query: str,
        top_k: int,
        use_reranker: bool,
    ) -> list[RetrievalCandidate]:
        self.received_query = query
        return self.results[:top_k]


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, **_kwargs: Any):
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )


def create_candidate() -> RetrievalCandidate:
    text = (
        "Corrective RAG evaluates retrieved evidence and "
        "rewrites unsuccessful queries."
    )

    chunk = DocumentChunk(
        chunk_id="W1001-section-2-chunk-0",
        paper_id="W1001",
        title="Corrective Retrieval-Augmented Generation",
        publication_year=2024,
        section_id="W1001-section-2",
        section_heading="Corrective retrieval",
        section_type="methods",
        section_order=2,
        chunk_index=0,
        text=text,
        token_count=11,
        character_count=len(text),
        source_format="grobid_tei",
        source_path="W1001.tei.xml",
    )

    return RetrievalCandidate(
        chunk=chunk,
        dense_score=0.85,
        rerank_score=4.5,
        sources=["dense"],
    )


def test_generation_prompt_contains_exact_source_id() -> None:
    document = EvidenceDocument.from_retrieval_candidate(
        create_candidate()
    )

    prompt = build_generation_prompt(
        question="How does corrective RAG recover?",
        documents=[document],
        maximum_characters_per_document=2400,
    )

    assert "W1001-section-2-chunk-0" in prompt
    assert "How does corrective RAG recover?" in prompt


def test_unknown_source_id_is_rejected() -> None:
    document = EvidenceDocument.from_retrieval_candidate(
        create_candidate()
    )
    generated = GeneratedAnswer(
        answer="The system rewrites queries [unknown].",
        source_ids=["unknown"],
    )

    with pytest.raises(ValueError, match="unknown source ID"):
        validate_generated_answer(generated, [document])


def test_declared_source_must_appear_in_answer() -> None:
    document = EvidenceDocument.from_retrieval_candidate(
        create_candidate()
    )
    generated = GeneratedAnswer(
        answer="The system rewrites queries.",
        source_ids=[document.source_id],
    )

    with pytest.raises(ValueError, match="declared citation"):
        validate_generated_answer(generated, [document])


def test_citation_metadata_is_built() -> None:
    document = EvidenceDocument.from_retrieval_candidate(
        create_candidate()
    )

    citations = build_citations(
        source_ids=[document.source_id],
        documents=[document],
    )

    assert len(citations) == 1
    assert citations[0].paper_id == "W1001"
    assert citations[0].section_heading == "Corrective retrieval"


def test_baseline_rag_retrieves_and_generates() -> None:
    candidate = create_candidate()
    source_id = candidate.chunk.chunk_id
    response = (
        '{"answer": "It rewrites failed queries ['
        + source_id
        + '].", "source_ids": ["'
        + source_id
        + '"]}'
    )

    retriever = FakeRetriever([candidate])
    fake_ollama = FakeOllamaClient(response)
    llm = OllamaStructuredClient(
        model="qwen3:14b",
        max_retries=1,
        client=fake_ollama,
    )

    result = run_baseline_rag(
        question="How does corrective RAG recover?",
        retriever=retriever,
        llm=llm,
        top_k=5,
    )

    assert result.source_ids == [source_id]
    assert len(result.citations) == 1
    assert retriever.received_query == (
        "How does corrective RAG recover?"
    )


def test_baseline_rag_handles_empty_retrieval() -> None:
    retriever = FakeRetriever([])
    fake_ollama = FakeOllamaClient("{}")
    llm = OllamaStructuredClient(
        model="qwen3:14b",
        max_retries=1,
        client=fake_ollama,
    )

    result = run_baseline_rag(
        question="An unanswered question",
        retriever=retriever,
        llm=llm,
    )

    assert result.source_ids == []
    assert "No relevant evidence" in result.answer
