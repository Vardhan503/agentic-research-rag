from agentic_rag.processing.final_corpus import (
    determine_final_status,
    measure_ambiguous_pipeline,
)

def test_pipeline_complete_after_screen_and_grade():
    ambiguous = [
        {"id": "A"},
        {"id": "B"},
        {"id": "C"},
    ]

    graded = [
        {"id": "A"},
        {"id": "B"},
    ]

    screening = [
        {
            "id": "C",
            "ollama_screening": {
                "decision": "clear_reject"
            },
        }
    ]

    shortlist = [
        {"id": "B"}
    ]

    status = measure_ambiguous_pipeline(
        ambiguous,
        graded,
        screening,
        shortlist,
    )

    assert status["screening_complete"]
    assert status[
        "shortlist_grading_complete"
    ]
    assert status["pipeline_complete"]


def test_pipeline_incomplete_when_shortlist_not_graded():
    ambiguous = [
        {"id": "A"},
        {"id": "B"},
    ]

    graded = [
        {"id": "A"}
    ]

    screening = [
        {
            "id": "B",
            "ollama_screening": {
                "decision": "keep"
            },
        }
    ]

    shortlist = [
        {"id": "B"}
    ]

    status = measure_ambiguous_pipeline(
        ambiguous,
        graded,
        screening,
        shortlist,
    )

    assert status["screening_complete"]
    assert not status[
        "shortlist_grading_complete"
    ]
    assert not status["pipeline_complete"]

def test_incomplete_shortlist_is_counted_as_deferred():
    ambiguous = [
        {"id": "A"},
        {"id": "B"},
    ]

    graded = [
        {"id": "A"}
    ]

    screening = [
        {
            "id": "B",
            "ollama_screening": {
                "decision": "keep"
            },
        }
    ]

    shortlist = [
        {"id": "B"}
    ]

    status = measure_ambiguous_pipeline(
        ambiguous,
        graded,
        screening,
        shortlist,
    )

    assert status["deferred_review"] == 1
    assert status["screening_complete"]
    assert not status["pipeline_complete"]


def test_deferred_review_can_finalize_corpus():
    pipeline_status = {
        "pipeline_complete": False,
        "screening_complete": True,
    }

    status = determine_final_status(
        pipeline_status,
        allow_deferred_review=True,
    )
    assert (
        status
        == "complete_with_deferred_review"
    )


def test_incomplete_screening_stays_provisional():
    pipeline_status = {
        "pipeline_complete": False,
        "screening_complete": False,
    }

    status = determine_final_status(
        pipeline_status,
        allow_deferred_review=True,
    )

    assert status == "provisional"