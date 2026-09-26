from agentic_rag.processing.models import (
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