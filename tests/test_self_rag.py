from types import SimpleNamespace
from typing import Any

from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.self_rag import (
    build_verification_prompt,
    run_self_rag,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient


class SequenceOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.prompts: list[str] = []

    def chat(self, **kwargs: Any):
        messages = kwargs["messages"]
        self.prompts.append(str(messages[1]["content"]))
        response = self.responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(
            message=SimpleNamespace(content=response)
        )


def create_document() -> EvidenceDocument:
    return EvidenceDocument(
        source_id="source-1",
        paper_id="W1001",
        title="Corrective Retrieval-Augmented Generation",
        section_heading="Methods",
        text=(
            "The method detects retrieval failure and rewrites the "
            "query before retrieving again."
        ),
    )


def create_llm(responses: list[str]):
    client = SequenceOllamaClient(responses)
    llm = OllamaStructuredClient(
        model="qwen3:14b",
        max_retries=1,
        client=client,
    )
    return llm, client


def generated_answer(answer: str) -> str:
    return (
        '{"answer":"'
        + answer
        + '","source_ids":["source-1"]}'
    )


def test_verification_prompt_contains_answer_and_evidence() -> None:
    prompt = build_verification_prompt(
        question="How does CRAG recover?",
        answer="It rewrites the query [source-1].",
        documents=[create_document()],
        maximum_characters_per_document=1800,
    )

    assert "It rewrites the query [source-1]." in prompt
    assert "detects retrieval failure" in prompt


def test_hallucination_causes_regeneration() -> None:
    responses = [
        generated_answer(
            "It retrains the generator [source-1]."
        ),
        """
        {
          "grounded": false,
          "reason": "Retraining is not supported.",
          "unsupported_claims": ["It retrains the generator."]
        }
        """,
        generated_answer(
            "It rewrites the failed query [source-1]."
        ),
        """
        {
          "grounded": true,
          "reason": "The claim is supported.",
          "unsupported_claims": []
        }
        """,
        """
        {
          "useful": true,
          "needs_more_context": false,
          "reason": "The answer is complete.",
          "improvement_feedback": ""
        }
        """,
    ]
    llm, client = create_llm(responses)

    result = run_self_rag(
        question="How does CRAG recover?",
        documents=[create_document()],
        llm=llm,
        maximum_generation_attempts=2,
    )

    assert result.status == "accepted"
    assert result.grounded is True
    assert result.useful is True
    assert result.generation_count == 2
    assert client.call_count == 5
    assert "Required correction" in client.prompts[2]
    assert "Retraining is not supported" in client.prompts[2]


def test_critic_can_request_more_context() -> None:
    responses = [
        generated_answer(
            "Failure is detected with confidence [source-1]."
        ),
        """
        {
          "grounded": true,
          "reason": "The claim is supported.",
          "unsupported_claims": []
        }
        """,
        """
        {
          "useful": false,
          "needs_more_context": true,
          "reason": "The recovery comparison is missing.",
          "improvement_feedback": "Retrieve recovery evidence."
        }
        """,
    ]
    llm, client = create_llm(responses)

    result = run_self_rag(
        question="Compare failure detection and recovery.",
        documents=[create_document()],
        llm=llm,
    )

    assert result.status == "needs_more_context"
    assert result.grounded is True
    assert result.useful is False
    assert result.needs_more_context is True
    assert client.call_count == 3


def test_critic_feedback_causes_answer_rewrite() -> None:
    responses = [
        generated_answer(
            "It rewrites queries [source-1]."
        ),
        """
        {
          "grounded": true,
          "reason": "The claim is supported.",
          "unsupported_claims": []
        }
        """,
        """
        {
          "useful": false,
          "needs_more_context": false,
          "reason": "Failure detection was omitted.",
          "improvement_feedback": "Explain detection before recovery."
        }
        """,
        generated_answer(
            "It detects failure and rewrites queries [source-1]."
        ),
        """
        {
          "grounded": true,
          "reason": "Both claims are supported.",
          "unsupported_claims": []
        }
        """,
        """
        {
          "useful": true,
          "needs_more_context": false,
          "reason": "The complete process is explained.",
          "improvement_feedback": ""
        }
        """,
    ]
    llm, client = create_llm(responses)

    result = run_self_rag(
        question="How does CRAG detect and recover?",
        documents=[create_document()],
        llm=llm,
        maximum_generation_attempts=2,
    )

    assert result.status == "accepted"
    assert result.generation_count == 2
    assert client.call_count == 6


def test_empty_evidence_returns_controlled_response() -> None:
    llm, client = create_llm([])

    result = run_self_rag(
        question="How does CRAG recover?",
        documents=[],
        llm=llm,
    )

    assert result.status == "insufficient_evidence"
    assert result.generation_count == 0
    assert client.call_count == 0
