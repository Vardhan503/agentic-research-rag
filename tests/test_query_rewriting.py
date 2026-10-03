from types import SimpleNamespace
from typing import Any

from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.query_rewriting import (
    build_rewrite_prompt,
    rewrite_query,
    run_corrective_retrieval,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient
from agentic_rag.processing.models import DocumentChunk
from agentic_rag.retrieval.models import RetrievalCandidate


class SequenceOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def chat(self, **_kwargs: Any):
        response = self.responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(
            message=SimpleNamespace(content=response)
        )


class SequenceRetriever:
    def __init__(
        self,
        result_batches: list[list[RetrievalCandidate]],
    ) -> None:
        self.result_batches = result_batches
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        top_k: int,
        use_reranker: bool,
    ) -> list[RetrievalCandidate]:
        self.queries.append(query)
        batch_index = len(self.queries) - 1
        return self.result_batches[batch_index][:top_k]


def create_llm(responses: list[str]):
    client = SequenceOllamaClient(responses)
    llm = OllamaStructuredClient(
        model="qwen3:14b",
        max_retries=1,
        client=client,
    )
    return llm, client


def create_candidate(
    source_id: str,
    text: str,
) -> RetrievalCandidate:
    chunk = DocumentChunk(
        chunk_id=source_id,
        paper_id="W1001",
        title="Adaptive Corrective RAG",
        publication_year=2025,
        section_id=source_id + "-section",
        section_heading="Methods",
        section_type="methods",
        section_order=1,
        chunk_index=0,
        text=text,
        token_count=8,
        character_count=len(text),
        source_format="grobid_tei",
        source_path="W1001.tei.xml",
    )

    return RetrievalCandidate(
        chunk=chunk,
        rerank_score=4.0,
        sources=["dense", "sparse"],
    )


def grading_response(
    source_ids: list[str],
) -> str:
    items: list[str] = []

    for source_id in source_ids:
        item = (
            '{"source_id":"'
            + source_id
            + '","grade":"correct",'
            + '"reason":"Useful evidence."}'
        )
        items.append(item)

    return '{"grades":[' + ",".join(items) + "]}"


def test_rewrite_prompt_contains_missing_information() -> None:
    document = EvidenceDocument(
        source_id="source-1",
        paper_id="W1001",
        title="Adaptive RAG",
        text="A confidence score detects failure.",
    )

    prompt = build_rewrite_prompt(
        question="How is failure detected and corrected?",
        current_query="adaptive RAG failure",
        documents=[document],
        missing_information="The correction method.",
    )

    assert "The correction method." in prompt
    assert "A confidence score detects failure." in prompt


def test_repeated_model_query_uses_fallback() -> None:
    response = """
    {
      "rewritten_query": "adaptive RAG failure",
      "reason": "Repeated query."
    }
    """
    llm, _client = create_llm([response])

    result = rewrite_query(
        question="How is failure corrected?",
        current_query="adaptive RAG failure",
        documents=[],
        missing_information="query rewriting recovery method",
        llm=llm,
    )

    assert result.rewritten_query == (
        "adaptive RAG failure query rewriting recovery method"
    )


def test_ambiguous_context_is_rewritten_and_retrieved() -> None:
    first_candidate = create_candidate(
        "source-1",
        "A confidence score detects retrieval failure.",
    )
    second_candidate = create_candidate(
        "source-2",
        "Query rewriting recovers from retrieval failure.",
    )
    retriever = SequenceRetriever(
        [[first_candidate], [second_candidate]]
    )

    responses = [
        grading_response(["source-1"]),
        """
        {
          "status": "incomplete",
          "reason": "Recovery is missing.",
          "missing_information": "The recovery mechanism."
        }
        """,
        """
        {
          "rewritten_query": "adaptive RAG query rewriting recovery",
          "reason": "Targets the missing recovery mechanism."
        }
        """,
        grading_response(["source-1", "source-2"]),
        """
        {
          "status": "sufficient",
          "reason": "Detection and recovery are both supported.",
          "missing_information": ""
        }
        """,
    ]
    llm, client = create_llm(responses)

    result = run_corrective_retrieval(
        question="How is retrieval failure detected and corrected?",
        retriever=retriever,
        llm=llm,
        maximum_rewrite_attempts=2,
    )

    assert result.assessment.route == "correct"
    assert result.rewrite_count == 1
    assert len(result.documents) == 2
    assert retriever.queries == [
        "How is retrieval failure detected and corrected?",
        "adaptive RAG query rewriting recovery",
    ]
    assert client.call_count == 5


def test_correct_context_does_not_rewrite() -> None:
    candidate = create_candidate(
        "source-1",
        "Confidence detects failure and query rewriting recovers.",
    )
    retriever = SequenceRetriever([[candidate]])
    responses = [
        grading_response(["source-1"]),
        """
        {
          "status": "sufficient",
          "reason": "Complete evidence is available.",
          "missing_information": ""
        }
        """,
    ]
    llm, _client = create_llm(responses)

    result = run_corrective_retrieval(
        question="How is failure detected and corrected?",
        retriever=retriever,
        llm=llm,
    )

    assert result.rewrite_count == 0
    assert len(retriever.queries) == 1
