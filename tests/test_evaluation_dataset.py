import json

import pytest

from agentic_rag.evaluation.dataset import load_evaluation_dataset


def make_record(question_id: str, category: str = "retrieval") -> dict:
    return {
        "question_id": question_id,
        "question": "What is hybrid retrieval?",
        "reference_answer": "It combines retrieval signals.",
        "category": category,
        "difficulty": "easy",
    }


def test_load_evaluation_dataset_filters_and_limits(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    records = [
        make_record("q1", "retrieval"),
        make_record("q2", "evaluation"),
        make_record("q3", "retrieval"),
    ]
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")

    examples = load_evaluation_dataset(
        path,
        category="retrieval",
        limit=1,
    )

    assert len(examples) == 1
    assert examples[0].question_id == "q1"


def test_load_evaluation_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    record = make_record("q1")
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate question_id"):
        load_evaluation_dataset(path)


def test_committed_benchmark_contains_fifty_unique_questions():
    examples = load_evaluation_dataset("data/benchmark/agentic_rag_evaluation.jsonl")
    identifiers = set()
    for example in examples:
        identifiers.add(example.question_id)

    assert len(examples) == 50
    assert len(identifiers) == 50
