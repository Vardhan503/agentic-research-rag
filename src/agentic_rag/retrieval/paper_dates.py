"""Paper-level publication dates for retrieved chunks.

The index stores only publication_year per chunk, but the corpus file has
full OpenAlex publication dates. Loading the 4k-entry map once is far cheaper
than re-embedding 200k chunks, and it lets the graders see whether a paper
falls inside a "last seven days" window.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalise_paper_id(value: str) -> str:
    """Return the bare OpenAlex work ID, with any URL prefix removed."""

    return value.strip().rsplit("/", 1)[-1]


def load_paper_dates(corpus_path: str | Path) -> dict[str, str]:
    """Map paper_id to ISO publication_date from the corpus JSONL."""

    path = Path(corpus_path)

    if not path.exists():
        return {}

    dates: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue

            record = json.loads(line)
            paper_id = normalise_paper_id(str(record.get("id") or ""))
            publication_date = str(record.get("publication_date") or "").strip()

            if paper_id and ISO_DATE_PATTERN.match(publication_date):
                dates[paper_id] = publication_date

    return dates
