from __future__ import annotations

import uuid
from typing import Any

from agentic_rag.evaluation.models import EvaluationExample


def upload_dataset(
    examples: list[EvaluationExample],
    dataset_name: str,
    description: str,
) -> dict[str, Any]:
    """Create or safely upsert the benchmark into a LangSmith dataset."""

    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )

    records: list[dict[str, Any]] = []
    for example in examples:
        stable_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            dataset_name + ":" + example.question_id,
        )
        records.append(
            {
                "id": stable_id,
                "inputs": {"question": example.question},
                "outputs": {
                    "reference_answer": example.reference_answer,
                    "expected_paper_ids": example.expected_paper_ids,
                    "expected_source_ids": example.expected_source_ids,
                },
                "metadata": {
                    "question_id": example.question_id,
                    "category": example.category,
                    "difficulty": example.difficulty,
                    "split": example.split,
                    "requires_retrieval": example.requires_retrieval,
                    "requires_web": example.requires_web,
                },
            }
        )

    client.create_examples(dataset_id=dataset.id, examples=records)
    return {
        "dataset_name": dataset_name,
        "dataset_id": str(dataset.id),
        "examples_uploaded": len(records),
    }
