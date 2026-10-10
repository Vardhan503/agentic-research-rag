from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.graph.crag import (
    UNGRADED_REASON,
    any_document_in_time_window,
    assess_combined_context,
    grade_documents,
    resolve_source_id,
    run_crag_assessment,
    validate_grade_batch,
)
from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.schemas import ContextAssessment, DocumentGradeBatch
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


def test_context_grader_receives_current_date_and_publication_year() -> None:
    prompts: list[str] = []

    class RecordingLLM:
        def invoke(self, **kwargs: Any) -> ContextAssessment:
            prompts.append(kwargs["user_prompt"])
            return ContextAssessment(
                status="incomplete",
                reason="No source is dated inside the requested window.",
                missing_information=("Agentic RAG research published between 2026-10-01 and 2026-10-08"),
            )

    document = create_document("chunk-1", "Agentic RAG for time series.")
    document.publication_year = 2024

    assessment = assess_combined_context(
        question="What agentic RAG research was published in the last seven days?",
        documents=[document],
        llm=RecordingLLM(),  # type: ignore[arg-type]
        today=date(2026, 10, 8),
    )

    assert assessment.status == "incomplete"
    assert prompts[0].startswith("Current date: 2026-10-08")
    assert "Year: 2024" in prompts[0]


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


def test_ungraded_source_is_kept_as_ambiguous() -> None:
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

    grade_index = validate_grade_batch(grade_batch, documents)

    assert grade_index["source-1"] == ("correct", "Useful evidence.")
    assert grade_index["source-2"] == ("ambiguous", UNGRADED_REASON)


def test_truncated_and_hallucinated_source_ids_are_repaired() -> None:
    documents = [
        create_document("W4414588074-chunk-947e000f2c1a", "Evidence one."),
        create_document("W4416076356-chunk-24ee52b5d860", "Evidence two."),
    ]
    grade_batch = DocumentGradeBatch.model_validate(
        {
            "grades": [
                {
                    # Last two characters dropped by the model.
                    "source_id": "W4414588074-chunk-947e000f2c",
                    "grade": "correct",
                    "reason": "Useful evidence.",
                },
                {
                    # Extra character added by the model.
                    "source_id": "W4416076356-chunk-24ee52b5d860a",
                    "grade": "incorrect",
                    "reason": "Unrelated.",
                },
                {
                    "source_id": "W9999999999-chunk-000000000000",
                    "grade": "correct",
                    "reason": "Does not exist.",
                },
            ]
        }
    )

    grade_index = validate_grade_batch(grade_batch, documents)

    assert grade_index == {
        "W4414588074-chunk-947e000f2c1a": ("correct", "Useful evidence."),
        "W4416076356-chunk-24ee52b5d860": ("incorrect", "Unrelated."),
    }


def test_prefix_shared_by_two_chunks_is_not_guessed() -> None:
    expected = {"W1-chunk-aaaa1111", "W1-chunk-aaaa2222"}

    assert resolve_source_id("W1-chunk-aaaa", expected) is None
    assert resolve_source_id("W1-chunk-aaaa1111", expected) == "W1-chunk-aaaa1111"


def test_batch_matching_no_source_is_rejected() -> None:
    documents = [create_document("source-1", "Evidence one.")]
    grade_batch = DocumentGradeBatch.model_validate(
        {
            "grades": [
                {
                    "source_id": "something-else-entirely",
                    "grade": "correct",
                    "reason": "Wrong chunk.",
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="no grade for any supplied source ID"):
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


def test_time_window_overrides_llm_and_skips_context_call() -> None:
    document = create_document("source-1", "Agentic RAG for time series analysis.")
    document.published_date = "2024-08-18"
    grading_response = """
    {
      "grades": [
        {
          "source_id": "source-1",
          "grade": "correct",
          "reason": "It discusses agentic RAG."
        }
      ]
    }
    """
    llm, client = create_llm([grading_response])

    result = run_crag_assessment(
        question="What agentic RAG research was published in the last seven days?",
        documents=[document],
        llm=llm,
        recent_days=7,
        today=date(2026, 10, 10),
    )

    assert result.route == "incorrect"
    assert result.context_status == "irrelevant"
    assert "2026-10-03" in result.missing_information
    assert client.call_count == 1


def test_date_filtered_web_result_counts_as_inside_window() -> None:
    document = create_document("web-1", "A new agentic RAG method.")
    document.source = "web"
    document.section_heading = "Web search result filtered to pages published 2026-10-03 to 2026-10-10"

    assert any_document_in_time_window([document], 7, date(2026, 10, 10)) is True
    old = create_document("old", "Old paper.")
    old.published_date = "2024-08-18"
    assert any_document_in_time_window([old], 7, date(2026, 10, 10)) is False
