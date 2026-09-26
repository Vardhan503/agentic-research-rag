from pathlib import Path

import pytest

from agentic_rag.processing.jats_parser import (
    JATSParsingError,
    parse_jats_file,
)


def create_complete_jats_file(path: Path) -> None:
    """Create a small structured JATS document."""

    xml = """
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="doi">
            10.1000/example
          </article-id>

          <title-group>
            <article-title>
              Retrieval-Augmented Generation Study
            </article-title>
          </title-group>

          <pub-date>
            <year>2024</year>
          </pub-date>

          <abstract>
            <p>
              This paper studies retrieval-augmented generation.
            </p>
          </abstract>
        </article-meta>
      </front>

      <body>
        <sec>
          <title>Introduction</title>
          <p>
            Retrieval connects language models with external
            documents.
          </p>
        </sec>

        <sec>
          <title>Methods</title>
          <p>
            We use dense and sparse retrieval.
          </p>

          <sec>
            <title>Retriever</title>
            <p>
              Results are combined using reciprocal rank fusion.
            </p>
          </sec>
        </sec>
      </body>
    </article>
    """

    path.write_text(xml, encoding="utf-8")


def test_parse_complete_jats_file(
    tmp_path: Path,
) -> None:
    """A valid JATS file should become a structured paper."""

    xml_path = tmp_path / "W1001.xml"
    create_complete_jats_file(xml_path)

    paper = parse_jats_file(
        xml_path=xml_path,
        paper_id="W1001",
    )

    assert paper.paper_id == "W1001"
    assert paper.title == (
        "Retrieval-Augmented Generation Study"
    )
    assert paper.doi == "10.1000/example"
    assert paper.publication_year == 2024
    assert paper.source_format == "jats_xml"

    assert paper.abstract is not None
    assert "retrieval-augmented" in paper.abstract

    assert len(paper.sections) == 3

    assert paper.sections[0].heading == "Introduction"
    assert paper.sections[0].section_type == "introduction"
    assert paper.sections[0].level == 1

    assert paper.sections[1].heading == "Methods"
    assert paper.sections[1].section_type == "methods"

    assert paper.sections[2].heading == "Retriever"
    assert paper.sections[2].level == 2


def test_parser_uses_metadata_when_xml_metadata_is_missing(
    tmp_path: Path,
) -> None:
    """Corpus metadata should fill missing XML metadata."""

    xml_path = tmp_path / "W1002.xml"

    xml_path.write_text(
        (
            "<article>"
            "<body>"
            "<sec>"
            "<title>Introduction</title>"
            "<p>Paper body text.</p>"
            "</sec>"
            "</body>"
            "</article>"
        ),
        encoding="utf-8",
    )

    metadata = {
        "title": "Metadata title",
        "abstract": "Metadata abstract",
        "doi": "10.1000/metadata",
        "publication_year": 2023,
    }

    paper = parse_jats_file(
        xml_path=xml_path,
        paper_id="W1002",
        metadata=metadata,
    )

    assert paper.title == "Metadata title"
    assert paper.abstract == "Metadata abstract"
    assert paper.doi == "10.1000/metadata"
    assert paper.publication_year == 2023


def test_parser_rejects_invalid_xml(
    tmp_path: Path,
) -> None:
    """Malformed XML should produce a clear parser error."""

    xml_path = tmp_path / "W1003.xml"
    xml_path.write_text(
        "<article><body>",
        encoding="utf-8",
    )

    with pytest.raises(
        JATSParsingError,
        match="Invalid JATS XML file",
    ):
        parse_jats_file(
            xml_path=xml_path,
            paper_id="W1003",
        )