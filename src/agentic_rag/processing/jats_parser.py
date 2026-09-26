from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from agentic_rag.processing.models import (
    PaperSection,
    ParsedPaper,
)


class JATSParsingError(Exception):
    """Raised when a JATS file cannot be parsed."""


def local_name(tag: str) -> str:
    """Remove an XML namespace from an element name."""

    if "}" in tag:
        return tag.split("}", maxsplit=1)[1].lower()

    return tag.lower()


def normalize_whitespace(text: str) -> str:
    """Replace repeated spaces and line breaks with single spaces."""

    return " ".join(text.split())


def element_text(element: ET.Element) -> str:
    """Extract all readable text contained inside an XML element."""

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
    """Find the first XML element with the requested local name."""

    expected_name = element_name.lower()

    for element in root.iter():
        if local_name(element.tag) == expected_name:
            return element

    return None


def get_attribute(
    element: ET.Element,
    attribute_name: str,
) -> str | None:
    """Read an attribute even when it contains an XML namespace."""

    expected_name = attribute_name.lower()

    for name, value in element.attrib.items():
        if local_name(name) == expected_name:
            return value

    return None


def extract_title(root: ET.Element) -> str | None:
    """Extract the article title from JATS metadata."""

    title_element = find_first_element(root, "article-title")

    if title_element is None:
        return None

    title = element_text(title_element)

    if not title:
        return None

    return title


def extract_abstract(root: ET.Element) -> str | None:
    """Extract abstract paragraphs from the JATS document."""

    abstract_element = find_first_element(root, "abstract")

    if abstract_element is None:
        return None

    paragraph_parts: list[str] = []

    for element in abstract_element.iter():
        if local_name(element.tag) != "p":
            continue

        paragraph_text = element_text(element)

        if paragraph_text:
            paragraph_parts.append(paragraph_text)

    if paragraph_parts:
        return normalize_whitespace(" ".join(paragraph_parts))

    abstract_text = element_text(abstract_element)

    if not abstract_text:
        return None

    return abstract_text


def extract_doi(root: ET.Element) -> str | None:
    """Extract the DOI from article identifiers."""

    for element in root.iter():
        if local_name(element.tag) != "article-id":
            continue

        identifier_type = get_attribute(
            element,
            "pub-id-type",
        )

        if identifier_type is None:
            continue

        if identifier_type.lower() != "doi":
            continue

        doi = element_text(element)

        if doi:
            return doi

    return None


def extract_publication_year(
    root: ET.Element,
) -> int | None:
    """Extract a four-digit publication year."""

    for element in root.iter():
        if local_name(element.tag) != "year":
            continue

        year_text = element_text(element)
        match = re.search(r"\b(19|20)\d{2}\b", year_text)

        if match is not None:
            return int(match.group(0))

    return None


def direct_child_title(
    section_element: ET.Element,
) -> str:
    """Read the title directly belonging to one section."""

    for child in section_element:
        if local_name(child.tag) != "title":
            continue

        return element_text(child)

    return ""


def extract_non_section_text(
    element: ET.Element,
) -> str:
    """Extract text without including nested JATS sections."""

    text_parts: list[str] = []

    text_block_names = (
        "p",
        "disp-quote",
        "statement",
        "list-item",
        "verse-group",
        "speech",
    )

    for child in element:
        child_name = local_name(child.tag)

        if child_name in ("title", "label", "sec"):
            continue

        if child_name in text_block_names:
            child_text = element_text(child)

            if child_text:
                text_parts.append(child_text)

            continue

        nested_text = extract_non_section_text(child)

        if nested_text:
            text_parts.append(nested_text)

    return normalize_whitespace(" ".join(text_parts))


def classify_section_type(heading: str) -> str:
    """Classify a section using its heading."""

    lower_heading = heading.lower()

    if "abstract" in lower_heading:
        return "abstract"

    if "introduction" in lower_heading:
        return "introduction"

    if "background" in lower_heading:
        return "background"

    if "related work" in lower_heading:
        return "related_work"

    if "literature review" in lower_heading:
        return "related_work"

    if "method" in lower_heading:
        return "methods"

    if "experimental setup" in lower_heading:
        return "methods"

    if "result" in lower_heading:
        return "results"

    if "discussion" in lower_heading:
        return "discussion"

    if "conclusion" in lower_heading:
        return "conclusion"

    if "limitation" in lower_heading:
        return "limitations"

    return "body"


def parse_section_tree(
    section_element: ET.Element,
    paper_id: str,
    level: int,
    starting_order: int,
) -> tuple[list[PaperSection], int]:
    """Parse one section and any sections nested inside it."""

    parsed_sections: list[PaperSection] = []
    current_order = starting_order

    heading = direct_child_title(section_element)
    text = extract_non_section_text(section_element)

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

        parsed_sections.append(section)
        current_order += 1

    for child in section_element:
        if local_name(child.tag) != "sec":
            continue

        child_sections, current_order = parse_section_tree(
            section_element=child,
            paper_id=paper_id,
            level=level + 1,
            starting_order=current_order,
        )

        parsed_sections.extend(child_sections)

    return parsed_sections, current_order


def extract_sections(
    root: ET.Element,
    paper_id: str,
) -> list[PaperSection]:
    """Extract ordered body sections from a JATS article."""

    sections: list[PaperSection] = []
    body_element = find_first_element(root, "body")

    if body_element is None:
        return sections

    current_order = 1

    opening_text = extract_non_section_text(body_element)

    if opening_text:
        opening_section = PaperSection(
            section_id=f"{paper_id}-section-{current_order}",
            heading="Body",
            text=opening_text,
            section_type="body",
            order=current_order,
            level=1,
        )

        sections.append(opening_section)
        current_order += 1

    for child in body_element:
        if local_name(child.tag) != "sec":
            continue

        child_sections, current_order = parse_section_tree(
            section_element=child,
            paper_id=paper_id,
            level=1,
            starting_order=current_order,
        )

        sections.extend(child_sections)

    if sections:
        return sections

    body_text = element_text(body_element)

    if body_text:
        fallback_section = PaperSection(
            section_id=f"{paper_id}-section-1",
            heading="Body",
            text=body_text,
            section_type="body",
            order=1,
            level=1,
        )

        sections.append(fallback_section)

    return sections


def optional_string(value: Any) -> str | None:
    """Convert a metadata value into a clean optional string."""

    if value is None:
        return None

    if not isinstance(value, str):
        return None

    clean_value = value.strip()

    if not clean_value:
        return None

    return clean_value


def optional_year(value: Any) -> int | None:
    """Convert a metadata year into an integer."""

    if value is None:
        return None

    try:
        year = int(value)
    except (TypeError, ValueError):
        return None

    if year < 1900 or year > 2100:
        return None

    return year


def parse_jats_file(
    xml_path: Path,
    paper_id: str,
    metadata: dict[str, Any] | None = None,
) -> ParsedPaper:
    """Convert one JATS XML file into a ParsedPaper."""

    if metadata is None:
        metadata = {}

    if not xml_path.exists():
        raise JATSParsingError(
            f"JATS XML file does not exist: {xml_path}"
        )

    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError) as error:
        raise JATSParsingError(
            f"Invalid JATS XML file: {xml_path}"
        ) from error

    root = tree.getroot()

    if local_name(root.tag) != "article":
        article_element = find_first_element(root, "article")

        if article_element is None:
            raise JATSParsingError(
                f"No JATS article element found: {xml_path}"
            )

        root = article_element

    warnings: list[str] = []

    xml_title = extract_title(root)
    metadata_title = optional_string(metadata.get("title"))

    title = xml_title or metadata_title

    if title is None:
        title = "Untitled paper"
        warnings.append("missing_title")

    xml_abstract = extract_abstract(root)
    metadata_abstract = optional_string(
        metadata.get("abstract")
    )

    abstract = xml_abstract or metadata_abstract

    if abstract is None:
        warnings.append("missing_abstract")

    metadata_doi = optional_string(metadata.get("doi"))
    doi = metadata_doi or extract_doi(root)

    metadata_year = optional_year(
        metadata.get("publication_year")
    )

    if metadata_year is None:
        metadata_year = optional_year(metadata.get("year"))

    publication_year = (
        metadata_year or extract_publication_year(root)
    )

    sections = extract_sections(root, paper_id)

    if not sections:
        warnings.append("missing_body_sections")

    parsed_paper = ParsedPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        doi=doi,
        publication_year=publication_year,
        source_format="jats_xml",
        source_path=str(xml_path),
        sections=sections,
        extraction_warnings=warnings,
    )

    return parsed_paper