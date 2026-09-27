from __future__ import annotations

from typing import Iterable, Literal

from pydantic import BaseModel, Field

from agentic_rag.retrieval.models import RetrievalCandidate


class EvidenceDocument(BaseModel):
    """One evidence item shared by retrieval, grading, and generation."""

    source_id: str
    paper_id: str
    title: str
    text: str
    section_heading: str = ""
    doi: str | None = None
    publication_year: int | None = None
    url: str | None = None
    source: Literal["corpus", "web"] = "corpus"

    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)

    grade: Literal["correct", "ambiguous", "incorrect"] | None = None
    grade_reason: str = ""

    @classmethod
    def from_retrieval_candidate(
        cls,
        candidate: RetrievalCandidate,
    ) -> "EvidenceDocument":
        """Convert one hybrid-search result into graph evidence."""

        chunk = candidate.chunk
        url = None

        if chunk.paper_id.startswith("W"):
            url = "https://openalex.org/" + chunk.paper_id

        return cls(
            source_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            title=chunk.title,
            text=chunk.text,
            section_heading=chunk.section_heading,
            doi=chunk.doi,
            publication_year=chunk.publication_year,
            url=url,
            dense_score=candidate.dense_score,
            sparse_score=candidate.sparse_score,
            rrf_score=candidate.rrf_score,
            rerank_score=candidate.rerank_score,
            retrieval_sources=candidate.sources.copy(),
        )

    def prompt_text(self, maximum_characters: int = 2400) -> str:
        """Format this evidence item for an LLM prompt."""

        text = self.text.strip()

        if len(text) > maximum_characters:
            text = text[:maximum_characters].rstrip() + "..."

        lines = [
            "Source ID: " + self.source_id,
            "Paper ID: " + self.paper_id,
            "Title: " + self.title,
        ]

        if self.section_heading:
            lines.append("Section: " + self.section_heading)

        if self.publication_year is not None:
            lines.append("Year: " + str(self.publication_year))

        if self.doi:
            lines.append("DOI: " + self.doi)

        lines.append("Text: " + text)

        return "\n".join(lines)


def convert_retrieval_results(
    candidates: Iterable[RetrievalCandidate],
) -> list[EvidenceDocument]:
    """Convert hybrid results and remove repeated chunk IDs."""

    documents: list[EvidenceDocument] = []
    seen_source_ids: set[str] = set()

    for candidate in candidates:
        document = EvidenceDocument.from_retrieval_candidate(candidate)

        if document.source_id in seen_source_ids:
            continue

        seen_source_ids.add(document.source_id)
        documents.append(document)

    return documents


def merge_evidence(
    existing: Iterable[EvidenceDocument],
    incoming: Iterable[EvidenceDocument],
) -> list[EvidenceDocument]:
    """Combine evidence from repeated retrieval without duplicate chunks."""

    merged: list[EvidenceDocument] = []
    seen_source_ids: set[str] = set()

    for collection in (existing, incoming):
        for document in collection:
            if document.source_id in seen_source_ids:
                continue

            seen_source_ids.add(document.source_id)
            merged.append(document)

    return merged


def build_context(
    documents: Iterable[EvidenceDocument],
    maximum_characters_per_document: int = 2400,
) -> str:
    """Build the labelled evidence block sent to the LLM."""

    blocks: list[str] = []

    for document in documents:
        blocks.append(
            document.prompt_text(
                maximum_characters=maximum_characters_per_document
            )
        )

    return "\n\n---\n\n".join(blocks)
