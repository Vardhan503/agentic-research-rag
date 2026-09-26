from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.processing.corpus_audit import (
    get_record_work_id,
    read_jsonl,
    write_json,
    write_jsonl,
)
from agentic_rag.processing.models import ParsedPaper


def load_parsed_papers(
    path: Path,
) -> list[ParsedPaper]:
    """Load and validate parsed papers from JSONL."""

    papers: list[ParsedPaper] = []

    if not path.exists():
        return papers

    records = read_jsonl(path)

    for record in records:
        paper = ParsedPaper.model_validate(record)
        papers.append(paper)

    return papers


def assemble_parsed_corpus(
    corpus_path: Path,
    parsed_jats_path: Path,
    parsed_tei_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Combine JATS and TEI papers in final-corpus order."""

    corpus_records = read_jsonl(corpus_path)
    jats_papers = load_parsed_papers(parsed_jats_path)
    tei_papers = load_parsed_papers(parsed_tei_path)

    paper_index: dict[str, ParsedPaper] = {}
    duplicates_removed = 0

    for paper in jats_papers:
        if paper.paper_id in paper_index:
            duplicates_removed += 1
            continue

        paper_index[paper.paper_id] = paper

    for paper in tei_papers:
        if paper.paper_id in paper_index:
            duplicates_removed += 1
            continue

        paper_index[paper.paper_id] = paper

    output_records: list[dict[str, Any]] = []
    missing_paper_ids: list[str] = []

    jats_count = 0
    tei_count = 0

    for corpus_record in corpus_records:
        work_id = get_record_work_id(corpus_record)

        if work_id is None:
            continue

        paper = paper_index.get(work_id)

        if paper is None:
            missing_paper_ids.append(work_id)
            continue

        output_record = paper.model_dump(mode="json")
        output_record["character_count"] = (
            paper.content_character_count()
        )

        output_records.append(output_record)

        if paper.source_format == "jats_xml":
            jats_count += 1
        elif paper.source_format == "grobid_tei":
            tei_count += 1

    write_jsonl(output_path, output_records)

    expected_papers = len(corpus_records)
    parsed_papers = len(output_records)

    coverage_rate = 0.0

    if expected_papers > 0:
        coverage_rate = parsed_papers / expected_papers

    status = "partial"

    if parsed_papers == expected_papers:
        status = "complete"

    missing_sample: list[str] = []

    for paper_id in missing_paper_ids[:20]:
        missing_sample.append(paper_id)

    report = {
        "status": status,
        "assembled_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "expected_papers": expected_papers,
        "parsed_jats_papers": jats_count,
        "parsed_tei_papers": tei_count,
        "duplicates_removed": duplicates_removed,
        "final_parsed_papers": parsed_papers,
        "missing_parsed_papers": len(missing_paper_ids),
        "coverage_rate": coverage_rate,
        "missing_paper_sample": missing_sample,
        "output_path": str(output_path),
    }

    write_json(report_path, report)

    return report