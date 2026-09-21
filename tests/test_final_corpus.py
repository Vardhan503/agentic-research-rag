import json

from agentic_rag.processing.final_corpus import (
    assemble_final_corpus,
    normalize_doi,
    normalize_title,
    read_jsonl,
    write_jsonl,
)


def make_paper(
    paper_id,
    title,
    doi=None,
    has_pdf=False,
    has_xml=False,
):
    return {
        "id": paper_id,
        "title": title,
        "doi": doi,
        "is_retracted": False,
        "has_content": {
            "pdf": has_pdf,
            "grobid_xml": has_xml,
        },
    }


def test_normalize_doi():
    paper = {
        "doi": "https://doi.org/10.1000/Test"
    }

    assert normalize_doi(paper) == (
        "10.1000/test"
    )


def test_normalize_title():
    paper = {
        "title": (
            "Retrieval-Augmented  Generation!"
        )
    }

    assert normalize_title(paper) == (
        "retrieval augmented generation"
    )


def test_assemble_final_corpus(tmp_path):
    retained_path = (
        tmp_path / "retained.jsonl"
    )

    ambiguous_path = (
        tmp_path / "ambiguous.jsonl"
    )

    graded_path = (
        tmp_path / "graded.jsonl"
    )

    accepted_path = (
        tmp_path / "accepted.jsonl"
    )

    manual_path = (
        tmp_path / "manual.jsonl"
    )

    output_path = (
        tmp_path / "final.jsonl"
    )

    report_path = (
        tmp_path / "report.json"
    )

    retained_records = [
        make_paper(
            "https://openalex.org/W1",
            "Retrieval-Augmented Generation",
            doi="https://doi.org/10.1/a",
            has_pdf=True,
        ),
        make_paper(
            "https://openalex.org/W2",
            "Dense Passage Retrieval",
            doi="https://doi.org/10.1/b",
            has_xml=True,
        ),
    ]

    accepted_records = [
        make_paper(
            "https://openalex.org/W3",
            "Retrieval Augmented Generation",
            doi="https://doi.org/10.1/c",
            has_pdf=True,
        ),
        make_paper(
            "https://openalex.org/W4",
            "Paper Without Full Text",
            doi="https://doi.org/10.1/d",
        ),
    ]

    manual_records = [
        make_paper(
            "https://openalex.org/W5",
            "Query Rewriting for Retrieval",
            doi="https://doi.org/10.1/e",
            has_pdf=True,
        ),
    ]

    ambiguous_records = [
        make_paper(
            "https://openalex.org/W6",
            "Ambiguous One",
            has_pdf=True,
        ),
        make_paper(
            "https://openalex.org/W7",
            "Ambiguous Two",
            has_pdf=True,
        ),
    ]

    graded_records = [
        ambiguous_records[0],
    ]

    write_jsonl(
        retained_records,
        retained_path,
    )

    write_jsonl(
        accepted_records,
        accepted_path,
    )

    write_jsonl(
        manual_records,
        manual_path,
    )

    write_jsonl(
        ambiguous_records,
        ambiguous_path,
    )

    write_jsonl(
        graded_records,
        graded_path,
    )

    config = {
        "corpus": {
            "target_papers": 3,
            "require_full_text": True,
        },
        "selection": {
            "retained_output": str(
                retained_path
            ),
            "ambiguous_output": str(
                ambiguous_path
            ),
        },
        "ollama_grading": {
            "graded_output_path": str(
                graded_path
            ),
            "accepted_output_path": str(
                accepted_path
            ),
        },
        "ollama_screening": {
            "decisions_output_path": str(
                tmp_path / "screening_decisions.jsonl"
            ),
            "shortlist_output_path": str(
                tmp_path / "screening_shortlist.jsonl"
            ),
        },
        "final_corpus": {
            "manual_approved_input": str(
                manual_path
            ),
            "output": str(
                output_path
            ),
            "report_output": str(
                report_path
            ),
        },
    }

    report = assemble_final_corpus(
        config,
        tmp_path,
    )

    final_records = read_jsonl(
        output_path
    )

    assert len(final_records) == 3
    assert report["status"] == "provisional"
    assert report["final_corpus_size"] == 3

    assert (
        report["duplicates_removed"][
            "duplicate_title"
        ]
        == 1
    )

    assert (
        report["quality_rejections"][
            "missing_downloadable_content"
        ]
        == 1
    )

    with open(
        report_path,
        "r",
        encoding="utf-8",
    ) as report_file:
        saved_report = json.load(
            report_file
        )

    assert (
        saved_report["final_corpus_size"]
        == 3
    )