from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from agentic_rag.evaluation.models import EvaluationExample


def load_evaluation_dataset(
    path: str | Path,
    split: str | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[EvaluationExample]:
    """Read, validate, filter, and de-duplicate a JSONL benchmark."""

    dataset_path = Path(path)
    examples: list[EvaluationExample] = []
    seen_ids: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                example = EvaluationExample.model_validate(record)
            except Exception as error:
                raise ValueError(
                    "Invalid benchmark record on line " + str(line_number)
                ) from error

            if example.question_id in seen_ids:
                raise ValueError("Duplicate question_id: " + example.question_id)

            seen_ids.add(example.question_id)

            if split is not None and example.split != split:
                continue

            if category is not None and example.category != category:
                continue

            examples.append(example)

            if limit is not None and limit > 0 and len(examples) >= limit:
                break

    if not examples:
        raise ValueError("No evaluation examples matched the requested filters.")

    return examples


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """Write JSON records using one durable, reviewable record per line."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
