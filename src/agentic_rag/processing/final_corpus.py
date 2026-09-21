import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from agentic_rag.processing.ollama_grader import paper_key

def read_jsonl(path):
    records = []
    input_path = Path(path)

    if not input_path.exists():
        return records

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            records.append(
                json.loads(stripped_line)
            )

    return records


def write_jsonl(records, path):
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            json.dump(
                record,
                output_file,
                ensure_ascii=False,
            )
            output_file.write("\n")


def write_json(data, path):
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            ensure_ascii=False,
        )


def resolve_project_path(
    path_value,
    project_root,
):
    path = Path(path_value)

    if not path.is_absolute():
        path = Path(project_root) / path

    return path


def normalize_openalex_id(paper):
    value = (
        paper.get("id")
        or paper.get("paper_id")
        or ""
    )

    value = str(value).strip().lower()

    if "openalex.org/" in value:
        value = value.rsplit("/", 1)[-1]

    return value


def normalize_doi(paper):
    value = paper.get("doi") or ""
    value = str(value).strip().lower()

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ]

    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break

    return value.strip()


def normalize_title(paper):
    title = paper.get("title") or ""

    title = unicodedata.normalize(
        "NFKC",
        str(title),
    )

    title = title.lower()

    title = re.sub(
        r"[^a-z0-9]+",
        " ",
        title,
    )

    return " ".join(title.split())


def has_downloadable_content(paper):
    has_content = paper.get(
        "has_content"
    ) or {}

    has_pdf = bool(
        has_content.get("pdf")
    )

    has_xml = bool(
        has_content.get("grobid_xml")
    )

    return has_pdf or has_xml


def find_quality_failures(
    paper,
    require_full_text,
):
    failures = []

    if paper.get("is_retracted"):
        failures.append("retracted")

    if not normalize_title(paper):
        failures.append("missing_title")

    if require_full_text:
        if not has_downloadable_content(paper):
            failures.append(
                "missing_downloadable_content"
            )

    return failures


def find_duplicate_reason(
    paper,
    seen_openalex_ids,
    seen_dois,
    seen_titles,
):
    openalex_id = normalize_openalex_id(
        paper
    )

    doi = normalize_doi(paper)
    title = normalize_title(paper)

    if openalex_id:
        if openalex_id in seen_openalex_ids:
            return "duplicate_openalex_id"

    if doi:
        if doi in seen_dois:
            return "duplicate_doi"

    if title:
        if title in seen_titles:
            return "duplicate_title"

    return None


def remember_paper(
    paper,
    seen_openalex_ids,
    seen_dois,
    seen_titles,
):
    openalex_id = normalize_openalex_id(
        paper
    )

    doi = normalize_doi(paper)
    title = normalize_title(paper)

    if openalex_id:
        seen_openalex_ids.add(
            openalex_id
        )

    if doi:
        seen_dois.add(doi)

    if title:
        seen_titles.add(title)


def count_content(final_records):
    with_pdf = 0
    with_xml = 0
    with_abstract = 0

    for paper in final_records:
        has_content = paper.get(
            "has_content"
        ) or {}

        if has_content.get("pdf"):
            with_pdf += 1

        if has_content.get("grobid_xml"):
            with_xml += 1

        abstract = paper.get("abstract")

        if not abstract:
            abstract = paper.get(
                "abstract_inverted_index"
            )

        if abstract:
            with_abstract += 1

    return {
        "papers_with_pdf": with_pdf,
        "papers_with_grobid_xml": with_xml,
        "papers_with_abstract": with_abstract,
    }

def measure_ambiguous_pipeline(
    ambiguous_records,
    graded_records,
    screening_records,
    shortlist_records,
):
    ambiguous_keys = set()
    graded_keys = set()
    screened_keys = set()
    shortlist_keys = set()

    for paper in ambiguous_records:
        ambiguous_keys.add(
            paper_key(paper)
        )

    for paper in graded_records:
        graded_keys.add(
            paper_key(paper)
        )

    for paper in screening_records:
        screened_keys.add(
            paper_key(paper)
        )

    for paper in shortlist_records:
        shortlist_keys.add(
            paper_key(paper)
        )

    resolved_keys = graded_keys.union(
        screened_keys
    )

    screening_complete = (
        ambiguous_keys.issubset(
            resolved_keys
        )
    )

    shortlist_grading_complete = (
        shortlist_keys.issubset(
            graded_keys
        )
    )

    pipeline_complete = (
        screening_complete
        and shortlist_grading_complete
    )

    return {
        "ambiguous_candidates": len(
            ambiguous_keys
        ),
        "ambiguous_graded": len(
            ambiguous_keys.intersection(
                graded_keys
            )
        ),
        "ambiguous_screened": len(
            ambiguous_keys.intersection(
                screened_keys
            )
        ),
        "screening_shortlist": len(
            shortlist_keys
        ),
        "shortlist_graded": len(
            shortlist_keys.intersection(
                graded_keys
            )
        ),
        "screening_complete": (
            screening_complete
        ),
        "shortlist_grading_complete": (
            shortlist_grading_complete
        ),
        "pipeline_complete": (
            pipeline_complete
        ),
    }

def assemble_final_corpus(
    config,
    project_root,
):
    corpus_config = config["corpus"]
    selection_config = config["selection"]
    grading_config = config[
        "ollama_grading"
    ]

    screening_config = config[
        "ollama_screening"
    ]

    final_config = config["final_corpus"]

    retained_path = resolve_project_path(
        selection_config["retained_output"],
        project_root,
    )

    ambiguous_path = resolve_project_path(
        selection_config["ambiguous_output"],
        project_root,
    )

    graded_path = resolve_project_path(
        grading_config["graded_output_path"],
        project_root,
    )

    screening_decisions_path = (
        resolve_project_path(
            screening_config[
                "decisions_output_path"
            ],
            project_root,
        )
    )

    screening_shortlist_path = (
        resolve_project_path(
            screening_config[
                "shortlist_output_path"
            ],
            project_root,
        )
    )

    accepted_path = resolve_project_path(
        grading_config["accepted_output_path"],
        project_root,
    )

    manual_path = resolve_project_path(
        final_config[
            "manual_approved_input"
        ],
        project_root,
    )

    output_path = resolve_project_path(
        final_config["output"],
        project_root,
    )

    report_path = resolve_project_path(
        final_config["report_output"],
        project_root,
    )

    retained_records = read_jsonl(
        retained_path
    )

    accepted_records = read_jsonl(
        accepted_path
    )

    manual_records = read_jsonl(
        manual_path
    )

    ambiguous_records = read_jsonl(
        ambiguous_path
    )

    graded_records = read_jsonl(
        graded_path
    )

    screening_records = read_jsonl(
        screening_decisions_path
    )

    shortlist_records = read_jsonl(
        screening_shortlist_path
    )

    sources = [
        (
            "deterministic_retained",
            retained_records,
        ),
        (
            "ollama_accepted",
            accepted_records,
        ),
        (
            "manual_approved",
            manual_records,
        ),
    ]

    final_records = []

    seen_openalex_ids = set()
    seen_dois = set()
    seen_titles = set()

    input_counts = {}
    accepted_by_source = Counter()
    duplicates_removed = Counter()
    quality_rejections = Counter()

    require_full_text = corpus_config.get(
        "require_full_text",
        False,
    )

    for source_name, records in sources:
        input_counts[source_name] = len(
            records
        )

        for paper in records:
            failures = find_quality_failures(
                paper,
                require_full_text,
            )

            if failures:
                for failure in failures:
                    quality_rejections[
                        failure
                    ] += 1

                continue

            duplicate_reason = (
                find_duplicate_reason(
                    paper,
                    seen_openalex_ids,
                    seen_dois,
                    seen_titles,
                )
            )

            if duplicate_reason:
                duplicates_removed[
                    duplicate_reason
                ] += 1
                continue

            final_paper = dict(paper)

            final_paper["final_corpus"] = {
                "source": source_name,
            }

            final_records.append(
                final_paper
            )

            accepted_by_source[
                source_name
            ] += 1

            remember_paper(
                paper,
                seen_openalex_ids,
                seen_dois,
                seen_titles,
            )

    write_jsonl(
        final_records,
        output_path,
    )

    pipeline_status = (
        measure_ambiguous_pipeline(
            ambiguous_records,
            graded_records,
            screening_records,
            shortlist_records,
        )
    )

    grading_complete = pipeline_status[
        "pipeline_complete"
    ]

    target_papers = int(
        corpus_config["target_papers"]
    )

    final_count = len(final_records)

    content_counts = count_content(
        final_records
    )

    status = "complete"

    if not grading_complete:
        status = "provisional"

    report = {
        "status": status,
        "assembled_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "ambiguous_grading_complete": (
            grading_complete
        ),
        "screening_complete": (
            pipeline_status[
                "screening_complete"
            ]
        ),
        "shortlist_grading_complete": (
            pipeline_status[
                "shortlist_grading_complete"
            ]
        ),
        "ambiguous_candidates": (
            pipeline_status[
                "ambiguous_candidates"
            ]
        ),
        "ambiguous_graded": (
            pipeline_status[
                "ambiguous_graded"
            ]
        ),
        "ambiguous_screened": (
            pipeline_status[
                "ambiguous_screened"
            ]
        ),
        "screening_shortlist": (
            pipeline_status[
                "screening_shortlist"
            ]
        ),
        "shortlist_graded": (
            pipeline_status[
                "shortlist_graded"
            ]
        ),
        "input_counts": input_counts,
        "accepted_by_source": dict(
            accepted_by_source
        ),
        "duplicates_removed": dict(
            duplicates_removed
        ),
        "quality_rejections": dict(
            quality_rejections
        ),
        "final_corpus_size": final_count,
        "target_papers": target_papers,
        "difference_from_target": (
            final_count - target_papers
        ),
        "remaining_gap": max(
            target_papers - final_count,
            0,
        ),
        "content_coverage": (
            content_counts
        ),
        "output_path": str(
            output_path
        ),
    }

    write_json(
        report,
        report_path,
    )

    return report