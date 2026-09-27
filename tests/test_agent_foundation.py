from types import SimpleNamespace

from agentic_rag.graph.documents import (
    EvidenceDocument,
    build_context,
    convert_retrieval_results,
    merge_evidence,
)
from agentic_rag.graph.schemas import (
    ContextAssessment,
    RetrievalDecision,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient
from agentic_rag.processing.models import DocumentChunk
from agentic_rag.retrieval.models import RetrievalCandidate


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )


def create_candidate() -> RetrievalCandidate:
    chunk = DocumentChunk(
        chunk_id="W1001-section-1-chunk-0",
        paper_id="W1001",
        title="Corrective Retrieval-Augmented Generation",
        doi="https://doi.org/10.1000/example",
        publication_year=2024,
        section_id="W1001-section-1",
        section_heading="Methods",
        section_type="methods",
        section_order=1,
        chunk_index=0,
        text="The system rewrites queries after retrieval failure.",
        token_count=9,
        character_count=52,
        source_format="grobid_tei",
        source_path="W1001.tei.xml",
    )

    return RetrievalCandidate(
        chunk=chunk,
        dense_score=0.81,
        sparse_score=12.5,
        rrf_score=0.03,
        rerank_score=4.2,
        sources=["dense", "sparse"],
    )


def test_structured_client_validates_response() -> None:
    fake_client = FakeOllamaClient(
        '{"retrieve": true, "reason": "Evidence is required."}'
    )

    client = OllamaStructuredClient(
        model="qwen3:14b",
        max_retries=1,
        client=fake_client,
    )

    result = client.invoke(
        system_prompt="Route the question.",
        user_prompt="What improves adaptive RAG?",
        response_model=RetrievalDecision,
    )

    assert result.retrieve is True
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["think"] is False


def test_candidate_is_converted_to_scientific_evidence() -> None:
    documents = convert_retrieval_results([create_candidate()])

    assert len(documents) == 1
    assert documents[0].paper_id == "W1001"
    assert documents[0].source_id == "W1001-section-1-chunk-0"
    assert documents[0].url == "https://openalex.org/W1001"
    assert documents[0].retrieval_sources == ["dense", "sparse"]


def test_merge_evidence_removes_duplicate_chunks() -> None:
    document = EvidenceDocument.from_retrieval_candidate(
        create_candidate()
    )

    merged = merge_evidence([document], [document])

    assert len(merged) == 1


def test_context_contains_citation_metadata() -> None:
    document = EvidenceDocument.from_retrieval_candidate(
        create_candidate()
    )

    context = build_context([document])

    assert "Source ID: W1001-section-1-chunk-0" in context
    assert "Paper ID: W1001" in context
    assert "Section: Methods" in context


def test_sufficient_context_clears_missing_information() -> None:
    assessment = ContextAssessment(
        status="sufficient",
        reason="The evidence answers the question.",
        missing_information="This should be cleared.",
    )

    assert assessment.missing_information == ""
