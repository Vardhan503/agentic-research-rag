from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
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


UNGRADED_REASON = "The grader returned no grade for this chunk; kept for the context check."


def resolve_source_id(
    returned_id: str,
    expected_source_ids: set[str],
) -> str | None:
    """Map a grader-returned ID onto the real chunk ID it refers to.

    Models sometimes drop the last characters of a long chunk ID or add a
    stray suffix. Accept the ID when exactly one real ID is a prefix match
    in either direction; return None when it matches nothing or is ambiguous.
    """

    clean_id = returned_id.strip()

    if clean_id in expected_source_ids:
        return clean_id

    if len(clean_id) < 8:
        return None

    candidates = [
        source_id
        for source_id in expected_source_ids
        if source_id.startswith(clean_id) or clean_id.startswith(source_id)
    ]

    if len(candidates) == 1:
        return candidates[0]

    return None


def validate_grade_batch(
    grade_batch: DocumentGradeBatch,
    documents: list[EvidenceDocument],
) -> dict[str, tuple[str, str]]:
    """Return one grade per retrieved source ID, repairing grader slips.

    Truncated IDs are matched to the real chunk, hallucinated IDs are
    ignored, and an ungraded chunk is kept as ambiguous. Only a batch that
    matches none of the supplied chunks is rejected, because that means the
    grader did not grade this evidence at all.
    """

    expected_source_ids: set[str] = set()

    for document in documents:
        expected_source_ids.add(document.source_id)

    grade_index: dict[str, tuple[str, str]] = {}

    for grade_item in grade_batch.grades:
        source_id = resolve_source_id(grade_item.source_id, expected_source_ids)

        if source_id is None or source_id in grade_index:
            continue

        grade_index[source_id] = (
            grade_item.grade,
            grade_item.reason,
        )

    if not grade_index:
        raise ValueError("Document grader returned no grade for any supplied source ID.")

    for source_id in expected_source_ids.difference(grade_index.keys()):
        grade_index[source_id] = ("ambiguous", UNGRADED_REASON)

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


ISO_DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
DATE_FILTERED_WEB_HEADING = "filtered to pages published"


def parse_iso_date(value: str | None) -> date | None:
    """Parse YYYY-MM-DD, ignoring a trailing time if one is present."""

    if not value:
        return None

    match = ISO_DATE_PATTERN.match(value.strip())

    if match is None:
        return None

    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def document_falls_in_window(
    document: EvidenceDocument,
    start: date,
    end: date,
    recent_days: int,
) -> bool:
    """Return True when this chunk is dated inside the requested window.

    A year alone cannot prove a paper is from the last week, so year-only
    metadata counts only for windows of a year or longer. Web results that
    Tavily already date-filtered are treated as inside the window.
    """

    published = parse_iso_date(document.published_date)

    if published is not None:
        return start <= published <= end

    if document.source == "web" and DATE_FILTERED_WEB_HEADING in document.section_heading.lower():
        return True

    if document.publication_year is None or recent_days < 365:
        return False

    return start.year <= document.publication_year <= end.year


def any_document_in_time_window(
    documents: list[EvidenceDocument],
    recent_days: int,
    today: date,
) -> bool:
    """Return True when at least one chunk falls inside today minus recent_days."""

    start = today - timedelta(days=int(recent_days))

    for document in documents:
        if document_falls_in_window(document, start, today, recent_days):
            return True

    return False


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

    current_date = today or datetime.now(UTC).date()

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
    recent_days: int | None = None,
    today: date | None = None,
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

    current_date = today or datetime.now(UTC).date()

    if recent_days and not any_document_in_time_window(
        selected_documents,
        recent_days,
        current_date,
    ):
        start = current_date - timedelta(days=int(recent_days))
        window = start.isoformat() + " to " + current_date.isoformat()
        return CRAGAssessment(
            route="incorrect",
            graded_documents=graded_documents,
            selected_documents=selected_documents,
            context_status="irrelevant",
            context_reason=("No selected source is dated inside " + window + "."),
            missing_information=("Sources published between " + window + "."),
        )

    context_assessment = assess_combined_context(
        question=question,
        documents=selected_documents,
        llm=llm,
        maximum_characters_per_document=(maximum_context_characters_per_document),
        today=current_date,
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
