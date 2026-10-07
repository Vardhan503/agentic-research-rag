from agentic_rag.processing.models import (
    PaperSection,
    ParsedPaper,
)
import pytest
from pydantic import ValidationError
from agentic_rag.processing.models import (
    DocumentChunk,
    PaperSection,
    ParsedPaper,
)


def test_parsed_paper_combines_all_text() -> None:
    """Title, abstract, headings, and text should be combined."""

    section = PaperSection(
        section_id="W1001-section-1",
        heading="Introduction",
        text="This section explains retrieval.",
        order=1,
    )

    paper = ParsedPaper(
        paper_id="W1001",
        title="Retrieval Paper",
        abstract="This is the paper abstract.",
        source_format="jats_xml",
        source_path="W1001.xml",
        sections=[section],
    )

    combined_text = paper.combined_text()

    assert "Retrieval Paper" in combined_text
    assert "This is the paper abstract." in combined_text
    assert "Introduction" in combined_text
    assert "This section explains retrieval." in combined_text


def test_parsed_papers_have_independent_lists() -> None:
    """Adding a section to one paper must not change another."""

    first_paper = ParsedPaper(
        paper_id="W1001",
        title="First paper",
        source_format="jats_xml",
        source_path="W1001.xml",
    )

    second_paper = ParsedPaper(
        paper_id="W1002",
        title="Second paper",
        source_format="jats_xml",
        source_path="W1002.xml",
    )

    section = PaperSection(
        section_id="W1001-section-1",
        heading="Methods",
        text="Method text",
        order=1,
    )

    first_paper.sections.append(section)

    assert len(first_paper.sections) == 1
    assert len(second_paper.sections) == 0


def test_document_chunk_builds_embedding_text() -> None:
    """Embedding text should include paper and section context."""

    text = "The retriever combines dense and sparse results."

    chunk = DocumentChunk(
        chunk_id="W1001-section-1-chunk-0",
        paper_id="W1001",
        title="Hybrid Retrieval Study",
        doi="10.1000/example",
        publication_year=2024,
        section_id="W1001-section-1",
        section_heading="Methods",
        section_type="methods",
        section_order=1,
        chunk_index=0,
        text=text,
        token_count=10,
        character_count=len(text),
        source_format="grobid_tei",
        source_path="data/raw/content/tei/W1001.tei.xml",
    )

    embedding_text = chunk.embedding_text()

    assert "Paper: Hybrid Retrieval Study" in embedding_text
    assert "Section: Methods" in embedding_text
    assert text in embedding_text


def test_document_chunk_rejects_empty_text() -> None:
    """A searchable chunk must contain usable text."""

    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="W1001-section-1-chunk-0",
            paper_id="W1001",
            title="Empty Chunk Test",
            section_id="W1001-section-1",
            section_heading="Methods",
            section_type="methods",
            section_order=1,
            chunk_index=0,
            text="",
            token_count=0,
            character_count=0,
            source_format="grobid_tei",
            source_path="W1001.tei.xml",
        )
