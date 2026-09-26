from pathlib import Path

import pytest

from agentic_rag.processing.tei_parser import (
    TEIParsingError,
    parse_tei_file,
)


def create_complete_tei(path: Path) -> None:
    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title>Agentic Retrieval Study</title>
          </titleStmt>

          <publicationStmt>
            <date when="2024-05-10" />
          </publicationStmt>

          <sourceDesc>
            <biblStruct>
              <idno type="DOI">10.1000/agentic</idno>
            </biblStruct>
          </sourceDesc>
        </fileDesc>

        <profileDesc>
          <abstract>
            <p>
              This paper studies agentic retrieval systems.
            </p>
          </abstract>
        </profileDesc>
      </teiHeader>

      <text>
        <body>
          <div>
            <head>Introduction</head>
            <p>
              Retrieval provides external information.
            </p>
          </div>

          <div>
            <head>Methods</head>
            <p>
              We use hybrid retrieval.
            </p>

            <div>
              <head>Reranking</head>
              <p>
                A cross encoder reranks the documents.
              </p>
            </div>
          </div>
        </body>
      </text>
    </TEI>
    """

    path.write_text(xml, encoding="utf-8")


def test_parse_complete_tei_file(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "W1001.tei.xml"
    create_complete_tei(xml_path)

    paper = parse_tei_file(
        xml_path=xml_path,
        paper_id="W1001",
    )

    assert paper.title == "Agentic Retrieval Study"
    assert paper.abstract is not None
    assert paper.doi == "10.1000/agentic"
    assert paper.publication_year == 2024
    assert paper.source_format == "grobid_tei"

    assert len(paper.sections) == 3
    assert paper.sections[0].heading == "Introduction"
    assert paper.sections[0].section_type == "introduction"
    assert paper.sections[1].heading == "Methods"
    assert paper.sections[1].section_type == "methods"
    assert paper.sections[2].heading == "Reranking"
    assert paper.sections[2].level == 2


def test_tei_parser_uses_metadata_fallback(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "W1002.tei.xml"

    xml_path.write_text(
        (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text>"
            "<body>"
            "<div>"
            "<head>Body</head>"
            "<p>Extracted paper content.</p>"
            "</div>"
            "</body>"
            "</text>"
            "</TEI>"
        ),
        encoding="utf-8",
    )

    metadata = {
        "title": "Metadata title",
        "abstract": "Metadata abstract",
        "doi": "10.1000/metadata",
        "publication_year": 2022,
    }

    paper = parse_tei_file(
        xml_path=xml_path,
        paper_id="W1002",
        metadata=metadata,
    )

    assert paper.title == "Metadata title"
    assert paper.abstract == "Metadata abstract"
    assert paper.doi == "10.1000/metadata"
    assert paper.publication_year == 2022


def test_tei_parser_rejects_invalid_xml(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "W1003.tei.xml"
    xml_path.write_text(
        "<TEI><text>",
        encoding="utf-8",
    )

    with pytest.raises(
        TEIParsingError,
        match="Invalid TEI XML file",
    ):
        parse_tei_file(
            xml_path=xml_path,
            paper_id="W1003",
        )