from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from agentic_rag.graph.documents import (
    EvidenceDocument,
    build_context,
)
from agentic_rag.graph.schemas import (
    ContextAssessment,
    DocumentGradeBatch,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient


DOCUMENT_GRADER_SYSTEM_PROMPT = """
You are the document grader in a Corrective RAG system over scientific papers.

Grade every supplied evidence chunk for the user's question.

Grades:
- correct: The chunk explicitly supports a fact, method, result, comparison, or
  relationship needed to answer at least one part of the question.
- ambiguous: The chunk is directly related and may help, but its evidence is
  indirect, incomplete, or requires another reasoning step.
- incorrect: The chunk does not provide useful evidence for this question.

Rules:
- Grade the evidence, not the paper title alone.
- A correct chunk does not need to contain the complete final answer.
- Preserve every Source ID exactly as supplied.
- Return exactly one grade for every supplied Source ID.
- Do not create Source IDs.
- Keep each reason to one short sentence.
- Do not answer the question.
""".strip()


CONTEXT_GRADER_SYSTEM_PROMPT = """
You are the context-sufficiency grader in a Corrective RAG system.

Judge all supplied scientific evidence together.

Statuses:
- sufficient: The evidence supports a complete answer to the question.
- incomplete: Some useful evidence exists, but a fact, comparison, or reasoning
  step is still missing.
- irrelevant: The evidence cannot support an answer to the question.

Rules:
- Combine evidence across papers and sections.
- Do not require one chunk to contain the entire answer.
- Do not use outside knowledge.
- Do not answer the question.
- For incomplete evidence, state exactly what information is missing.
- For sufficient evidence, missing_information must be empty.

Time-bound questions:
- Use the supplied current date to resolve phrases such as "last seven days",
  "this week", "recent", "latest", or "this year" into a concrete date window.
- Evidence counts toward a time-bound question only when its Year, its
  Published date, or a Section saying it was filtered to pages published in a
  date range shows it falls inside that window. Never assume a source is recent.
- If no source's date confirms it is inside the window, the status is
  incomplete, and missing_information must name the date window, for example
  "Agentic RAG research published between 2026-10-01 and 2026-10-08".
""".strip()


class CRAGAssessment(BaseModel):
    """Document grades and the resulting Corrective RAG route."""

    route: Literal["correct", "ambiguous", "incorrect"]
    graded_documents: list[EvidenceDocument] = Field(default_factory=list)
    selected_documents: list[EvidenceDocument] = Field(default_factory=list)
    context_status: Literal[
        "sufficient",
        "incomplete",
        "irrelevant",
    ]
    context_reason: str
    missing_information: str = ""


def build_document_grading_prompt(
    question: str,
    documents: list[EvidenceDocument],
    maximum_characters_per_document: int,
) -> str:
    """Build one prompt containing all chunks to grade."""

    if not documents:
        raise ValueError("At least one document is required for grading.")

    context = build_context(
        documents=documents,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    return "Question:\n" + question.strip() + "\n\nEvidence chunks to grade:\n" + context


TOKENS_PER_GRADE_ITEM = 160
GRADE_BATCH_BASE_TOKENS = 64


def grading_output_tokens(document_count: int) -> int:
    """Return an output budget large enough for one grade per chunk.

    Each grade item carries a chunk ID, a grade, and a reason of up to
    400 characters, so a fixed small num_predict truncates the JSON
    once several chunks are graded together.
    """

    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    return GRADE_BATCH_BASE_TOKENS + TOKENS_PER_GRADE_ITEM * document_count


def validate_grade_batch(
    grade_batch: DocumentGradeBatch,
    documents: list[EvidenceDocument],
) -> dict[str, tuple[str, str]]:
    """Require exactly one grade for every retrieved source ID."""

    expected_source_ids: set[str] = set()

    for document in documents:
        expected_source_ids.add(document.source_id)

    grade_index: dict[str, tuple[str, str]] = {}

    for grade_item in grade_batch.grades:
        source_id = grade_item.source_id

        if source_id not in expected_source_ids:
            raise ValueError("Document grader returned an unknown source ID: " + source_id)

        if source_id in grade_index:
            raise ValueError("Document grader returned a duplicate source ID: " + source_id)

        grade_index[source_id] = (
            grade_item.grade,
            grade_item.reason,
        )

    missing_source_ids = expected_source_ids.difference(grade_index.keys())

    if missing_source_ids:
        missing_text = ", ".join(sorted(missing_source_ids))

        raise ValueError("Document grader omitted source IDs: " + missing_text)

    return grade_index


def grade_documents(
    question: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    maximum_characters_per_document: int = 1600,
) -> list[EvidenceDocument]:
    """Grade all retrieved chunks in one structured Ollama request."""

    if not documents:
        return []

    prompt = build_document_grading_prompt(
        question=question,
        documents=documents,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    grade_batch = llm.invoke(
        system_prompt=DOCUMENT_GRADER_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=DocumentGradeBatch,
        num_predict=grading_output_tokens(len(documents)),
    )

    grade_index = validate_grade_batch(
        grade_batch=grade_batch,
        documents=documents,
    )

    graded_documents: list[EvidenceDocument] = []

    for document in documents:
        grade, reason = grade_index[document.source_id]

        graded_document = document.model_copy(
            deep=True,
            update={
                "grade": grade,
                "grade_reason": reason,
            },
        )

        graded_documents.append(graded_document)

    return graded_documents


def select_useful_documents(
    documents: list[EvidenceDocument],
) -> list[EvidenceDocument]:
    """Discard only chunks that the grader marked incorrect."""

    selected_documents: list[EvidenceDocument] = []

    for document in documents:
        if document.grade == "incorrect":
            continue

        selected_documents.append(document)

    return selected_documents


def assess_combined_context(
    question: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    maximum_characters_per_document: int = 2000,
    today: date | None = None,
) -> ContextAssessment:
    """Decide whether all selected chunks can answer the question."""

    if not documents:
        return ContextAssessment(
            status="irrelevant",
            reason="No useful retrieved evidence remains.",
            missing_information=("All evidence needed to answer the question."),
        )

    context = build_context(
        documents=documents,
        maximum_characters_per_document=(maximum_characters_per_document),
    )

    current_date = today or date.today()

    prompt = (
        "Current date: "
        + current_date.isoformat()
        + "\n\nQuestion:\n"
        + question.strip()
        + "\n\nSelected scientific evidence:\n"
        + context
    )

    return llm.invoke(
        system_prompt=CONTEXT_GRADER_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=ContextAssessment,
    )


def route_context_status(
    status: str,
) -> Literal["correct", "ambiguous", "incorrect"]:
    """Convert context sufficiency into the CRAG branch name."""

    if status == "sufficient":
        return "correct"

    if status == "incomplete":
        return "ambiguous"

    return "incorrect"


def run_crag_assessment(
    question: str,
    documents: list[EvidenceDocument],
    llm: OllamaStructuredClient,
    maximum_grade_characters_per_document: int = 1600,
    maximum_context_characters_per_document: int = 2000,
) -> CRAGAssessment:
    """Grade retrieved chunks and select the next CRAG action."""

    if not documents:
        return CRAGAssessment(
            route="incorrect",
            context_status="irrelevant",
            context_reason="Hybrid retrieval returned no evidence.",
            missing_information=("All evidence needed to answer the question."),
        )

    graded_documents = grade_documents(
        question=question,
        documents=documents,
        llm=llm,
        maximum_characters_per_document=(maximum_grade_characters_per_document),
    )

    selected_documents = select_useful_documents(graded_documents)

    if not selected_documents:
        return CRAGAssessment(
            route="incorrect",
            graded_documents=graded_documents,
            selected_documents=[],
            context_status="irrelevant",
            context_reason=("Every retrieved chunk was graded incorrect."),
            missing_information=("Relevant scientific evidence for the question."),
        )

    context_assessment = assess_combined_context(
        question=question,
        documents=selected_documents,
        llm=llm,
        maximum_characters_per_document=(maximum_context_characters_per_document),
    )

    route = route_context_status(context_assessment.status)

    return CRAGAssessment(
        route=route,
        graded_documents=graded_documents,
        selected_documents=selected_documents,
        context_status=context_assessment.status,
        context_reason=context_assessment.reason,
        missing_information=(context_assessment.missing_information),
    )
