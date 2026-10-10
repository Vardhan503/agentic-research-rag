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


def test_committed_benchmark_is_complete_and_labelled():
    examples = load_evaluation_dataset("data/benchmark/agentic_rag_evaluation.jsonl")
    identifiers = {example.question_id for example in examples}

    assert len(examples) == 71
    assert len(identifiers) == 71

    # Every static retrieval question carries retrieval gold, so recall, MRR
    # and nDCG are computed over the whole benchmark rather than four items.
    unlabelled = [
        example.question_id
        for example in examples
        if example.requires_retrieval
        and example.reference_mode == "static"
        and not example.requires_web
        and not example.expects_abstention
        and not example.expected_paper_ids
    ]
    assert unlabelled == []

    # The hard additions cover every graph branch.
    categories = {example.category for example in examples}
    assert {"multi_hop", "wrong_premise", "unanswerable", "web_fallback"} <= categories
    assert any(example.expects_abstention for example in examples)
