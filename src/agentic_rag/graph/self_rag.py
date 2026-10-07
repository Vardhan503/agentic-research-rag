from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic_rag.graph.baseline import (
    Citation,
    build_citations,
    build_generation_prompt,
    invoke_grounded_generation,
)
from agentic_rag.graph.documents import (
    EvidenceDocument,
    build_context,
)
from agentic_rag.graph.schemas import (
    AnswerCritique,
    GeneratedAnswer,
    HallucinationResult,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient


HALLUCINATION_CHECKER_SYSTEM_PROMPT = """
You are the grounding verifier in a scientific Self-RAG system.

Determine whether every factual claim in the generated answer is supported by
the supplied evidence.

Rules:
- Evaluate only claims made in the generated answer.
- Do not use outside knowledge.
- Evidence may be combined across several sources.
- Equivalent wording does not need to be an exact quotation.
- A citation ID by itself is not evidence; verify the cited text.
- Verify every relationship in a multi-step claim.
- An honest statement that evidence is insufficient is not a hallucination.
- unsupported_claims must contain only claims from the generated answer.
- If no claim is unsupported, grounded must be true.
""".strip()


ANSWER_CRITIC_SYSTEM_PROMPT = """
You are the answer-quality critic in a scientific Self-RAG system.

The answer has already passed citation validation and grounding verification.
Decide whether it directly and completely answers the user's question.

Check that the answer:
- Addresses every part of the question.
- Uses important information available in the evidence.
- Is clear, specific, and scientifically cautious.
- Uses citations after the claims they support.
- Avoids irrelevant detail.

Set needs_more_context to true only when evidence required for a complete answer
is genuinely absent. Set it to false when the existing evidence is sufficient
but the answer merely needs rewriting.

When useful is true, improvement_feedback must be empty.
""".strip()


class SelfRAGResult(BaseModel):
    """Final answer and Self-RAG verification diagnostics."""

    status: Literal[
        "accepted",
        "needs_more_context",
        "retry_exhausted",
        "insufficient_evidence",
    ]
    answer: str
    source_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    useful: bool = False
    needs_more_context: bool = False
    hallucination_reason: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)
    critic_reason: str = ""
    improvement_feedback: str = ""
    generation_count: int = 0


def build_verification_prompt(
    question: str,
    answer: str,
    documents: list[EvidenceDocument],
    maximum_characters_per_document: int,
) -> str:
    """Build shared evidence input for grounding and usefulness checks."""

    if not documents:
        raise ValueError("At least one evidence document is required for verification.")

    context = build_context(
        documents=documents,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    return (
        "Question:\n"
        + question.strip()
        + "\n\nGenerated answer:\n"
        + answer.strip()
        + "\n\nAvailable evidence:\n"
        + context
    )


def build_regeneration_prompt(
    question: str,
    documents: list[EvidenceDocument],
    feedback: str,
    previous_answer: str,
    maximum_characters_per_document: int,
) -> str:
    """Add verifier feedback to the normal grounded-generation prompt."""

    prompt = build_generation_prompt(
        question=question,
        documents=documents,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    if not feedback.strip():
        return prompt

    return (
        prompt
        + "\n\nPrevious answer:\n"
        + previous_answer.strip()
        + "\n\nRequired correction:\n"
        + feedback.strip()
        + "\n\nGenerate a corrected answer using only the evidence."
    )


def generate_answer(
    question: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    feedback: str = "",
    previous_answer: str = "",
    maximum_characters_per_document: int = 2400,
) -> tuple[GeneratedAnswer, list[str]]:
    """Generate an answer, optionally using feedback from a prior attempt."""

    prompt = build_regeneration_prompt(
        question=question,
        documents=documents,
        feedback=feedback,
        previous_answer=previous_answer,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    return invoke_grounded_generation(
        prompt=prompt,
        documents=documents,
        llm=llm,
    )


def check_hallucination(
    question: str,
    answer: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    maximum_characters_per_document: int = 1800,
) -> HallucinationResult:
    """Check whether every factual answer claim is evidence-grounded."""

    prompt = build_verification_prompt(
        question=question,
        answer=answer,
        documents=documents,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    return llm.invoke(
        system_prompt=HALLUCINATION_CHECKER_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=HallucinationResult,
    )


def critique_answer(
    question: str,
    answer: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    maximum_characters_per_document: int = 1800,
) -> AnswerCritique:
    """Check answer completeness after it passes grounding verification."""

    prompt = build_verification_prompt(
        question=question,
        answer=answer,
        documents=documents,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    return llm.invoke(
        system_prompt=ANSWER_CRITIC_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=AnswerCritique,
    )


def hallucination_feedback(
    result: HallucinationResult,
) -> str:
    """Convert unsupported claims into regeneration instructions."""

    feedback = "Remove or correct every unsupported factual claim."

    if result.unsupported_claims:
        feedback = feedback + " Unsupported claims: "
        feedback = feedback + "; ".join(result.unsupported_claims)

    if result.reason:
        feedback = feedback + " Verifier reason: " + result.reason

    return feedback


def run_self_rag(
    question: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    maximum_generation_attempts: int = 2,
    maximum_generation_characters_per_document: int = 2400,
    maximum_verifier_characters_per_document: int = 1800,
) -> SelfRAGResult:
    """Generate, verify, critique, and conditionally regenerate an answer."""

    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    if maximum_generation_attempts <= 0:
        raise ValueError("maximum_generation_attempts must be positive.")

    if not documents:
        return SelfRAGResult(
            status="insufficient_evidence",
            answer=("There is not enough supported evidence to answer the question."),
            critic_reason="No evidence documents were available.",
        )

    feedback = ""
    previous_answer = ""
    last_generated: GeneratedAnswer | None = None
    last_source_ids: list[str] = []
    last_hallucination: HallucinationResult | None = None
    last_critique: AnswerCritique | None = None

    for attempt_number in range(1, maximum_generation_attempts + 1):
        generated, source_ids = generate_answer(
            question=clean_question,
            documents=documents,
            llm=llm,
            feedback=feedback,
            previous_answer=previous_answer,
            maximum_characters_per_document=(maximum_generation_characters_per_document),
        )

        last_generated = generated
        last_source_ids = source_ids

        hallucination = check_hallucination(
            question=clean_question,
            answer=generated.answer,
            documents=documents,
            llm=llm,
            maximum_characters_per_document=(maximum_verifier_characters_per_document),
        )
        last_hallucination = hallucination

        if not hallucination.grounded:
            feedback = hallucination_feedback(hallucination)
            previous_answer = generated.answer
            continue

        critique = critique_answer(
            question=clean_question,
            answer=generated.answer,
            documents=documents,
            llm=llm,
            maximum_characters_per_document=(maximum_verifier_characters_per_document),
        )
        last_critique = critique

        citations = build_citations(
            source_ids=source_ids,
            documents=documents,
        )

        if critique.useful:
            return SelfRAGResult(
                status="accepted",
                answer=generated.answer,
                source_ids=source_ids,
                citations=citations,
                grounded=True,
                useful=True,
                hallucination_reason=hallucination.reason,
                unsupported_claims=(hallucination.unsupported_claims),
                critic_reason=critique.reason,
                generation_count=attempt_number,
            )

        if critique.needs_more_context:
            return SelfRAGResult(
                status="needs_more_context",
                answer=generated.answer,
                source_ids=source_ids,
                citations=citations,
                grounded=True,
                useful=False,
                needs_more_context=True,
                hallucination_reason=hallucination.reason,
                critic_reason=critique.reason,
                improvement_feedback=(critique.improvement_feedback),
                generation_count=attempt_number,
            )

        feedback = critique.improvement_feedback
        previous_answer = generated.answer

    answer = ""

    if last_generated is not None:
        answer = last_generated.answer

    citations = build_citations(
        source_ids=last_source_ids,
        documents=documents,
    )

    result = SelfRAGResult(
        status="retry_exhausted",
        answer=answer,
        source_ids=last_source_ids,
        citations=citations,
        generation_count=maximum_generation_attempts,
    )

    if last_hallucination is not None:
        result.grounded = last_hallucination.grounded
        result.hallucination_reason = last_hallucination.reason
        result.unsupported_claims = last_hallucination.unsupported_claims

    if last_critique is not None:
        result.useful = last_critique.useful
        result.needs_more_context = last_critique.needs_more_context
        result.critic_reason = last_critique.reason
        result.improvement_feedback = last_critique.improvement_feedback

    return result
