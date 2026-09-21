from agentic_rag.processing.ollama_grader import (
    PaperGrade,
    build_paper_payload,
    reconstruct_abstract,
    route_grade,
    validate_grade_consistency,
)


def test_reconstruct_abstract():
    inverted_index = {
        "Dense": [0],
        "retrieval": [1],
        "works": [2],
    }

    assert reconstruct_abstract(inverted_index) == "Dense retrieval works"


def test_payload_excludes_leaky_fields():
    paper = {
        "title": "A paper",
        "abstract": "An abstract",
        "matched_queries": ["retrieval augmented generation"],
        "selection": {"relevance_score": 7},
        "topics": [{"display_name": "Information Retrieval"}],
        "keywords": [{"display_name": "RAG"}],
    }

    payload = build_paper_payload(paper, 1000)

    assert payload == {
        "title": "A paper",
        "abstract": "An abstract",
    }


def test_payload_truncates_long_abstract():
    paper = {
        "title": "A paper",
        "abstract": "1234567890",
    }

    payload = build_paper_payload(paper, 5)
    assert payload["abstract"] == "12345"


def test_relevant_grade_routes_to_accepted():
    grade = PaperGrade(
        label="relevant",
        relevance_score=0.91,
        reason="Studies dense passage retrieval.",
    )

    config = {
        "accepted_min_score": 0.80,
        "rejected_max_score": 0.39,
    }

    assert route_grade(grade, config) == "accepted"


def test_ambiguous_grade_routes_to_review():
    grade = PaperGrade(
        label="ambiguous",
        relevance_score=0.55,
        reason="Retrieval may be central but evidence is incomplete.",
    )

    config = {
        "accepted_min_score": 0.80,
        "rejected_max_score": 0.39,
    }

    assert route_grade(grade, config) == "needs_review"


def test_irrelevant_grade_routes_to_rejected():
    grade = PaperGrade(
        label="irrelevant",
        relevance_score=0.05,
        reason="Studies medical imaging rather than textual retrieval.",
    )

    config = {
        "accepted_min_score": 0.80,
        "rejected_max_score": 0.39,
    }

    assert route_grade(grade, config) == "rejected"


def test_grade_consistency_rejects_bad_score():
    grade = PaperGrade(
        label="relevant",
        relevance_score=0.50,
        reason="Inconsistent test grade.",
    )

    try:
        validate_grade_consistency(grade)
        assert False
    except ValueError:
        assert True

