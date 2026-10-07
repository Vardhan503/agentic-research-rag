from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.graph.crag import (
    grade_documents,
    run_crag_assessment,
    validate_grade_batch,
)
from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.schemas import DocumentGradeBatch
from agentic_rag.llm.ollama_client import OllamaStructuredClient


class SequenceOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def chat(self, **_kwargs: Any):
        response = self.responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(message=SimpleNamespace(content=response))


def create_document(
    source_id: str,
    text: str,
) -> EvidenceDocument:
    return EvidenceDocument(
        source_id=source_id,
        paper_id="W1001",
        title="Corrective RAG",
        section_heading="Methods",
        text=text,
    )


def create_llm(responses: list[str]):
    client = SequenceOllamaClient(responses)
    llm = OllamaStructuredClient(
        model="qwen3:14b",
        max_retries=1,
        client=client,
    )
    return llm, client


def test_documents_are_graded_in_one_request() -> None:
    documents = [
        create_document("source-1", "Query rewriting."),
        create_document("source-2", "Image classification."),
    ]
    response = """
    {
      "grades": [
        {
          "source_id": "source-1",
          "grade": "correct",
          "reason": "It describes query rewriting."
        },
        {
          "source_id": "source-2",
          "grade": "incorrect",
          "reason": "It concerns computer vision."
        }
      ]
    }
    """
    llm, client = create_llm([response])

    graded = grade_documents(
        question="How does CRAG rewrite queries?",
        documents=documents,
        llm=llm,
    )

    assert client.call_count == 1
    assert graded[0].grade == "correct"
    assert graded[1].grade == "incorrect"


def test_missing_source_grade_is_rejected() -> None:
    documents = [
        create_document("source-1", "Evidence one."),
        create_document("source-2", "Evidence two."),
    ]
    grade_batch = DocumentGradeBatch.model_validate(
        {
            "grades": [
                {
                    "source_id": "source-1",
                    "grade": "correct",
                    "reason": "Useful evidence.",
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="omitted source IDs"):
        validate_grade_batch(grade_batch, documents)


def test_all_incorrect_routes_without_context_call() -> None:
    document = create_document(
        "source-1",
        "Unrelated image classification evidence.",
    )
    grading_response = """
    {
      "grades": [
        {
          "source_id": "source-1",
          "grade": "incorrect",
          "reason": "Unrelated to textual retrieval."
        }
      ]
    }
    """
    llm, client = create_llm([grading_response])

    result = run_crag_assessment(
        question="How does CRAG improve retrieval?",
        documents=[document],
        llm=llm,
    )

    assert result.route == "incorrect"
    assert result.selected_documents == []
    assert client.call_count == 1


def test_sufficient_context_routes_correct() -> None:
    document = create_document(
        "source-1",
        "CRAG rewrites a query after retrieval failure.",
    )
    grading_response = """
    {
      "grades": [
        {
          "source_id": "source-1",
          "grade": "correct",
          "reason": "It explains recovery through query rewriting."
        }
      ]
    }
    """
    context_response = """
    {
      "status": "sufficient",
      "reason": "The evidence answers the question.",
      "missing_information": ""
    }
    """
    llm, client = create_llm([grading_response, context_response])

    result = run_crag_assessment(
        question="How does CRAG recover from failure?",
        documents=[document],
        llm=llm,
    )

    assert result.route == "correct"
    assert result.context_status == "sufficient"
    assert client.call_count == 2


def test_incomplete_context_routes_ambiguous() -> None:
    document = create_document(
        "source-1",
        "A confidence score detects retrieval failure.",
    )
    grading_response = """
    {
      "grades": [
        {
          "source_id": "source-1",
          "grade": "correct",
          "reason": "It explains failure detection."
        }
      ]
    }
    """
    context_response = """
    {
      "status": "incomplete",
      "reason": "Recovery is not described.",
      "missing_information": "How the failed retrieval is corrected."
    }
    """
    llm, _client = create_llm([grading_response, context_response])

    result = run_crag_assessment(
        question="How is failure detected and corrected?",
        documents=[document],
        llm=llm,
    )

    assert result.route == "ambiguous"
    assert "corrected" in result.missing_information
