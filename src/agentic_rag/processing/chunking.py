from __future__ import annotations

import hashlib
import re

from agentic_rag.processing.models import (
    DocumentChunk,
    ParsedPaper,
    PaperSection,
)


TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    re.UNICODE,
)

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """Replace repeated whitespace with single spaces."""

    return " ".join(text.split())


def count_tokens(text: str) -> int:
    """Estimate token count using words and punctuation."""

    matches = TOKEN_PATTERN.findall(text)

    return len(matches)


def trailing_token_text(
    text: str,
    token_limit: int,
) -> str:
    """Return the last requested tokens from some text."""

    if token_limit <= 0:
        return ""

    matches = list(TOKEN_PATTERN.finditer(text))

    if not matches:
        return ""

    if len(matches) <= token_limit:
        return text.strip()

    start_match = matches[-token_limit]
    start_position = start_match.start()

    return text[start_position:].strip()


def split_oversized_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text that cannot fit inside one chunk."""

    matches = list(TOKEN_PATTERN.finditer(text))
    chunks: list[str] = []

    if not matches:
        return chunks

    step_size = max_tokens - overlap_tokens

    start_index = 0

    while start_index < len(matches):
        end_index = min(
            start_index + max_tokens,
            len(matches),
        )

        start_position = matches[start_index].start()
        end_position = matches[end_index - 1].end()

        chunk_text = text[start_position:end_position].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end_index == len(matches):
            break

        start_index += step_size

    return chunks


def split_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text at sentence boundaries with overlap."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    clean_text = normalize_text(text)

    if not clean_text:
        return []

    if count_tokens(clean_text) <= max_tokens:
        return [clean_text]

    sentences = SENTENCE_BOUNDARY_PATTERN.split(clean_text)

    chunks: list[str] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        clean_sentence = sentence.strip()

        if not clean_sentence:
            continue

        sentence_tokens = count_tokens(clean_sentence)

        if sentence_tokens > max_tokens:
            if current_sentences:
                current_text = " ".join(current_sentences)
                chunks.append(current_text)
                current_sentences = []

            oversized_chunks = split_oversized_text(
                text=clean_sentence,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )

            chunks.extend(oversized_chunks)
            continue

        candidate_sentences = current_sentences + [clean_sentence]
        candidate_text = " ".join(candidate_sentences)

        if count_tokens(candidate_text) <= max_tokens:
            current_sentences.append(clean_sentence)
            continue

        current_text = " ".join(current_sentences)
        chunks.append(current_text)

        overlap_text = trailing_token_text(
            text=current_text,
            token_limit=overlap_tokens,
        )

        current_sentences = []

        if overlap_text:
            overlap_candidate = f"{overlap_text} {clean_sentence}"

            if count_tokens(overlap_candidate) <= max_tokens:
                current_sentences.append(overlap_text)

        current_sentences.append(clean_sentence)

    if current_sentences:
        final_text = " ".join(current_sentences)
        chunks.append(final_text)

    return chunks


def build_chunk_id(
    paper_id: str,
    section_id: str,
    chunk_index: int,
    text: str,
) -> str:
    """Create a stable ID from chunk identity and content."""

    identity = f"{paper_id}|{section_id}|{chunk_index}|{text}"

    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]

    return f"{paper_id}-chunk-{digest}"


def create_section_chunks(
    paper: ParsedPaper,
    section: PaperSection,
    max_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    """Convert one paper section into document chunks."""

    chunk_texts = split_text(
        text=section.text,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunks: list[DocumentChunk] = []

    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk_id = build_chunk_id(
            paper_id=paper.paper_id,
            section_id=section.section_id,
            chunk_index=chunk_index,
            text=chunk_text,
        )

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            title=paper.title,
            doi=paper.doi,
            publication_year=paper.publication_year,
            section_id=section.section_id,
            section_heading=section.heading,
            section_type=section.section_type,
            section_order=section.order,
            chunk_index=chunk_index,
            text=chunk_text,
            token_count=count_tokens(chunk_text),
            character_count=len(chunk_text),
            source_format=paper.source_format,
            source_path=paper.source_path,
            page_start=section.page_start,
            page_end=section.page_end,
        )

        chunks.append(chunk)

    return chunks


def create_abstract_chunks(
    paper: ParsedPaper,
    max_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    """Convert a paper abstract into document chunks."""

    if paper.abstract is None:
        return []

    clean_abstract = normalize_text(paper.abstract)

    if not clean_abstract:
        return []

    abstract_section = PaperSection(
        section_id=f"{paper.paper_id}-abstract",
        heading="Abstract",
        text=clean_abstract,
        section_type="abstract",
        order=0,
        level=1,
    )

    return create_section_chunks(
        paper=paper,
        section=abstract_section,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def chunk_paper(
    paper: ParsedPaper,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
    include_abstract: bool = True,
) -> list[DocumentChunk]:
    """Create all searchable chunks for one paper."""

    chunks: list[DocumentChunk] = []

    if include_abstract:
        abstract_chunks = create_abstract_chunks(
            paper=paper,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        chunks.extend(abstract_chunks)

    for section in paper.sections:
        section_chunks = create_section_chunks(
            paper=paper,
            section=section,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        chunks.extend(section_chunks)

    return chunks
