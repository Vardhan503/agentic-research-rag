from pydantic import BaseModel, Field


class PaperSection(BaseModel):
    """One structured section extracted from a research paper."""

    section_id: str
    heading: str
    text: str
    section_type: str = "body"
    order: int
    level: int = 1
    page_start: int | None = None
    page_end: int | None = None


class ParsedPaper(BaseModel):
    """One research paper converted into our standard format."""

    paper_id: str
    title: str
    abstract: str | None = None
    doi: str | None = None
    publication_year: int | None = None

    source_format: str
    source_path: str

    sections: list[PaperSection] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)

    def combined_text(self) -> str:
        """Combine the title, abstract, and sections into one text."""

        text_parts: list[str] = []

        clean_title = self.title.strip()

        if clean_title:
            text_parts.append(clean_title)

        if self.abstract:
            clean_abstract = self.abstract.strip()

            if clean_abstract:
                text_parts.append(clean_abstract)

        for section in self.sections:
            clean_heading = section.heading.strip()
            clean_text = section.text.strip()

            if clean_heading:
                text_parts.append(clean_heading)

            if clean_text:
                text_parts.append(clean_text)

        return "\n\n".join(text_parts)

    def content_character_count(self) -> int:
        """Return the number of extracted text characters."""

        return len(self.combined_text())

    def has_usable_text(
        self,
        minimum_characters: int = 500,
    ) -> bool:
        """Check whether enough text was extracted for retrieval."""

        character_count = self.content_character_count()

        return character_count >= minimum_characters