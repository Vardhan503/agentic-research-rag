from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from agentic_rag.processing.jats_parser import (
    classify_section_type,
    normalize_whitespace,
    optional_string,
    optional_year,
)
from agentic_rag.processing.models import (
    PaperSection,
    ParsedPaper,
)


class TEIParsingError(Exception):
    """Raised when GROBID TEI cannot be parsed."""


def local_name(tag: str) -> str:
    """Remove the namespace from an XML element name."""

    if "}" in tag:
        return tag.split("}", maxsplit=1)[1].lower()

    return tag.lower()


def element_text(element: ET.Element) -> str:
    """Extract readable text from an XML element."""

    text_parts: list[str] = []

    for text in element.itertext():
        clean_text = normalize_whitespace(text)

        if clean_text:
            text_parts.append(clean_text)

    return normalize_whitespace(" ".join(text_parts))


def find_first_element(
    root: ET.Element,
    element_name: str,
) -> ET.Element | None:
    """Find the first element with the requested local name."""

    expected_name = element_name.lower()

    for element in root.iter():
        if local_name(element.tag) == expected_name:
            return element

    return None


def find_header(root: ET.Element) -> ET.Element | None:
    """Find the TEI metadata header."""

    return find_first_element(root, "teiheader")


def extract_title(root: ET.Element) -> str | None:
    """Extract the article title from the TEI header."""

    header = find_header(root)

    if header is None:
        return None

    title_element = find_first_element(header, "title")

    if title_element is None:
        return None

    title = element_text(title_element)

    if not title:
        return None

    return title


def extract_abstract(root: ET.Element) -> str | None:
    """Extract the article abstract from the TEI header."""

    header = find_header(root)

    if header is None:
        return None

    abstract_element = find_first_element(
        header,
        "abstract",
    )

    if abstract_element is None:
        return None

    abstract = element_text(abstract_element)

    if not abstract:
        return None

    return abstract


def extract_doi(root: ET.Element) -> str | None:
    """Extract a DOI from TEI identifier elements."""

    header = find_header(root)

    if header is None:
        return None

    for element in header.iter():
        if local_name(element.tag) != "idno":
            continue

        identifier_type = element.attrib.get("type", "")

        if identifier_type.lower() != "doi":
            continue

        doi = element_text(element)

        if doi:
            return doi

    return None


def extract_publication_year(
    root: ET.Element,
) -> int | None:
    """Extract a publication year from TEI dates."""

    header = find_header(root)

    if header is None:
        return None

    for element in header.iter():
        if local_name(element.tag) != "date":
            continue

        date_value = element.attrib.get("when", "")

        if not date_value:
            date_value = element_text(element)

        match = re.search(
            r"\b(19|20)\d{2}\b",
            date_value,
        )

        if match is not None:
            return int(match.group(0))

    return None


def direct_division_heading(
    division: ET.Element,
) -> str:
    """Extract the heading directly inside one TEI division."""

    for child in division:
        if local_name(child.tag) != "head":
            continue

        return element_text(child)

    return ""


def extract_non_division_text(
    element: ET.Element,
) -> str:
    """Extract section text without duplicating nested divisions."""

    text_parts: list[str] = []

    text_block_names = (
        "p",
        "quote",
        "item",
        "formula",
        "note",
        "ab",
    )

    for child in element:
        child_name = local_name(child.tag)

        if child_name in ("head", "div"):
            continue

        if child_name in text_block_names:
            text = element_text(child)

            if text:
                text_parts.append(text)

            continue

        nested_text = extract_non_division_text(child)

        if nested_text:
            text_parts.append(nested_text)

    return normalize_whitespace(" ".join(text_parts))


def parse_division_tree(
    division: ET.Element,
    paper_id: str,
    level: int,
    starting_order: int,
) -> tuple[list[PaperSection], int]:
    """Parse one TEI division and its nested divisions."""

    sections: list[PaperSection] = []
    current_order = starting_order

    heading = direct_division_heading(division)
    text = extract_non_division_text(division)

    if text:
        section = PaperSection(
            section_id=(
                f"{paper_id}-section-{current_order}"
            ),
            heading=heading,
            text=text,
            section_type=classify_section_type(heading),
            order=current_order,
            level=level,
        )

        sections.append(section)
        current_order += 1

    for child in division:
        if local_name(child.tag) != "div":
            continue

        child_sections, current_order = (
            parse_division_tree(
                division=child,
                paper_id=paper_id,
                level=level + 1,
                starting_order=current_order,
            )
        )

        sections.extend(child_sections)

    return sections, current_order


def extract_sections(
    root: ET.Element,
    paper_id: str,
) -> list[PaperSection]:
    """Extract ordered paper sections from the TEI body."""

    sections: list[PaperSection] = []
    body = find_first_element(root, "body")

    if body is None:
        return sections

    current_order = 1

    opening_text = extract_non_division_text(body)

    if opening_text:
        section = PaperSection(
            section_id=f"{paper_id}-section-1",
            heading="Body",
            text=opening_text,
            section_type="body",
            order=1,
            level=1,
        )

        sections.append(section)
        current_order += 1

    for child in body:
        if local_name(child.tag) != "div":
            continue

        child_sections, current_order = (
            parse_division_tree(
                division=child,
                paper_id=paper_id,
                level=1,
                starting_order=current_order,
            )
        )

        sections.extend(child_sections)

    if sections:
        return sections

    body_text = element_text(body)

    if body_text:
        section = PaperSection(
            section_id=f"{paper_id}-section-1",
            heading="Body",
            text=body_text,
            section_type="body",
            order=1,
            level=1,
        )

        sections.append(section)

    return sections


def parse_tei_file(
    xml_path: Path,
    paper_id: str,
    metadata: dict[str, Any] | None = None,
) -> ParsedPaper:
    """Convert one GROBID TEI file into a ParsedPaper."""

    if metadata is None:
        metadata = {}

    if not xml_path.exists():
        raise TEIParsingError(
            f"TEI XML file does not exist: {xml_path}"
        )

    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError) as error:
        raise TEIParsingError(
            f"Invalid TEI XML file: {xml_path}"
        ) from error

    root = tree.getroot()

    if local_name(root.tag) != "tei":
        raise TEIParsingError(
            f"Unexpected TEI root element: {root.tag}"
        )

    warnings: list[str] = []

    title = extract_title(root)

    if title is None:
        title = optional_string(metadata.get("title"))

    if title is None:
        title = "Untitled paper"
        warnings.append("missing_title")

    abstract = extract_abstract(root)

    if abstract is None:
        abstract = optional_string(
            metadata.get("abstract")
        )

    if abstract is None:
        warnings.append("missing_abstract")

    doi = optional_string(metadata.get("doi"))

    if doi is None:
        doi = extract_doi(root)

    publication_year = optional_year(
        metadata.get("publication_year")
    )

    if publication_year is None:
        publication_year = optional_year(
            metadata.get("year")
        )

    if publication_year is None:
        publication_year = extract_publication_year(root)

    sections = extract_sections(root, paper_id)

    if not sections:
        warnings.append("missing_body_sections")

    return ParsedPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        doi=doi,
        publication_year=publication_year,
        source_format="grobid_tei",
        source_path=str(xml_path),
        sections=sections,
        extraction_warnings=warnings,
    )