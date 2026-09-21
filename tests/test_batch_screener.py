import json

from agentic_rag.processing.batch_screener import (
    build_screening_items,
    build_screening_outputs,
    normalize_reject_indexes,
    screen_batch,
)
from agentic_rag.processing.ollama_grader import (
    read_jsonl,
    write_jsonl,
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.content)


def screening_config(tmp_path):
    return {
        "model": "qwen3:14b",
        "host": "http://localhost:11434",
        "batch_size": 20,
        "preview_characters": 5,
        "context_window": 8192,
        "num_predict": 120,
        "timeout_seconds": 300,
        "max_retries": 1,
        "retry_delay_seconds": 0,
        "keep_alive": "30m",
        "input_path": str(
            tmp_path / "candidates.jsonl"
        ),
        "existing_graded_path": str(
            tmp_path / "graded.jsonl"
        ),
        "decisions_output_path": str(
            tmp_path / "decisions.jsonl"
        ),
        "shortlist_output_path": str(
            tmp_path / "shortlist.jsonl"
        ),
        "rejected_output_path": str(
            tmp_path / "rejected.jsonl"
        ),
        "report_path": str(
            tmp_path / "report.json"
        ),
    }


def test_build_screening_items_truncates_abstract():
    papers = [
        {
            "title": "Dense retrieval",
            "abstract": "123456789",
            "selection": {
                "relevance_score": 7
            },
        }
    ]

    items = build_screening_items(
        papers,
        preview_characters=5,
    )

    assert items == [
        {
            "index": 0,
            "title": "Dense retrieval",
            "abstract_preview": "12345",
        }
    ]


def test_normalize_reject_indexes_is_safe():
    indexes = [
        -1,
        0,
        0,
        2,
        10,
        True,
    ]

    assert normalize_reject_indexes(
        indexes,
        batch_size=3,
    ) == [0, 2]


def test_screen_batch_reads_structured_result(
    tmp_path,
):
    config = screening_config(tmp_path)

    response = json.dumps(
        {
            "clear_reject_indexes": [
                1
            ]
        }
    )

    client = FakeClient(response)

    papers = [
        {
            "title": "Dense retrieval",
            "abstract": "Text search",
        },
        {
            "title": "Image classification",
            "abstract": "Computer vision",
        },
    ]

    rejected = screen_batch(
        papers,
        client,
        config,
    )

    assert rejected == [1]
    assert len(client.calls) == 1


def test_outputs_exclude_previously_graded(
    tmp_path,
):
    config = screening_config(tmp_path)

    candidates = [
        {"id": "A", "title": "Already graded"},
        {"id": "B", "title": "Keep"},
        {"id": "C", "title": "Reject"},
    ]

    graded = [
        {"id": "A", "title": "Already graded"}
    ]

    decisions = [
        {
            "id": "B",
            "title": "Keep",
            "ollama_screening": {
                "decision": "keep",
                "fallback_keep": False,
            },
        },
        {
            "id": "C",
            "title": "Reject",
            "ollama_screening": {
                "decision": "clear_reject",
                "fallback_keep": False,
            },
        },
    ]

    write_jsonl(
        candidates,
        config["input_path"],
    )

    write_jsonl(
        graded,
        config["existing_graded_path"],
    )

    write_jsonl(
        decisions,
        config["decisions_output_path"],
    )

    report = build_screening_outputs(
        config
    )

    shortlist = read_jsonl(
        config["shortlist_output_path"]
    )

    rejected = read_jsonl(
        config["rejected_output_path"]
    )

    assert report["status"] == "complete"
    assert report["previously_graded"] == 1
    assert report[
        "shortlisted_for_full_grading"
    ] == 1
    assert report["clear_rejects"] == 1
    assert shortlist[0]["id"] == "B"
    assert rejected[0]["id"] == "C"