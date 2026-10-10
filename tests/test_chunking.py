from agentic_rag.processing.chunking import (
    build_chunk_id,
    chunk_paper,
    count_tokens,
    split_text,
)
from agentic_rag.processing.models import (
    PaperSection,
    ParsedPaper,
)


def create_test_paper(
    section_text: str,
) -> ParsedPaper:
    """Create one paper for chunking tests."""

    section = PaperSection(
        section_id="W1001-section-1",
        heading="Methods",
        text=section_text,
        section_type="methods",
        order=1,
    )

    return ParsedPaper(
        paper_id="W1001",
        title="Hybrid Retrieval Study",
        abstract="This is the research paper abstract.",
        doi="10.1000/example",
        publication_year=2024,
        source_format="grobid_tei",
        source_path="W1001.tei.xml",
        sections=[section],
    )


def test_short_text_stays_in_one_chunk() -> None:
    """Short text should not be split unnecessarily."""

    chunks = split_text(
        text="This is one short scientific sentence.",
        max_tokens=20,
        overlap_tokens=5,
    )

    assert len(chunks) == 1
    assert chunks[0] == ("This is one short scientific sentence.")


def test_long_section_creates_multiple_chunks() -> None:
    """Long sections should respect the token limit."""

    sentences: list[str] = []

    for number in range(1, 21):
        sentence = f"Sentence {number} explains hybrid retrieval and document reranking."
        sentences.append(sentence)

    section_text = " ".join(sentences)
    paper = create_test_paper(section_text)

    chunks = chunk_paper(
        paper=paper,
        max_tokens=30,
        overlap_tokens=5,
        include_abstract=False,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.token_count <= 30
        assert chunk.section_id == "W1001-section-1"


def test_chunking_does_not_mix_sections() -> None:
    """Each chunk must belong to exactly one section."""

    first_section = PaperSection(
        section_id="W1001-section-1",
        heading="Introduction",
        text="INTRODUCTION_MARKER retrieval background.",
        section_type="introduction",
        order=1,
    )

    second_section = PaperSection(
        section_id="W1001-section-2",
        heading="Results",
        text="RESULTS_MARKER evaluation results.",
        section_type="results",
        order=2,
    )

    paper = ParsedPaper(
        paper_id="W1001",
        title="Section Boundary Test",
        source_format="jats_xml",
        source_path="W1001.xml",
        sections=[
            first_section,
            second_section,
        ],
    )

    chunks = chunk_paper(
        paper=paper,
        max_tokens=20,
        overlap_tokens=5,
        include_abstract=False,
    )

    assert len(chunks) == 2

    assert "INTRODUCTION_MARKER" in chunks[0].text
    assert "RESULTS_MARKER" not in chunks[0].text

    assert "RESULTS_MARKER" in chunks[1].text
    assert "INTRODUCTION_MARKER" not in chunks[1].text


def test_abstract_becomes_searchable_chunk() -> None:
    """A paper abstract should become its own chunk."""

    paper = create_test_paper("This is the methods section.")

    chunks = chunk_paper(
        paper=paper,
        max_tokens=50,
        overlap_tokens=10,
        include_abstract=True,
    )

    assert chunks[0].section_type == "abstract"
    assert chunks[0].section_order == 0
    assert chunks[0].section_heading == "Abstract"


def test_chunk_ids_are_deterministic() -> None:
    """The same content should always produce the same ID."""

    first_id = build_chunk_id(
        paper_id="W1001",
        section_id="W1001-section-1",
        chunk_index=0,
        text="Stable research text.",
    )

    second_id = build_chunk_id(
        paper_id="W1001",
        section_id="W1001-section-1",
        chunk_index=0,
        text="Stable research text.",
    )

    assert first_id == second_id


def test_token_counter_includes_punctuation() -> None:
    """Words and punctuation should be counted."""

    token_count = count_tokens("Dense retrieval, sparse retrieval.")

    assert token_count == 6
