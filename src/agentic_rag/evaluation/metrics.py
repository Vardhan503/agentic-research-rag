from __future__ import annotations

import math
import re

from agentic_rag.evaluation.models import (
    DeterministicScores,
    EvaluationExample,
    PipelineOutput,
)


def normalize_identifier(value: str) -> str:
    """Normalize OpenAlex URLs and local IDs for reliable comparison."""

    normalized = value.strip().rstrip("/")
    if "/" in normalized:
        normalized = normalized.rsplit("/", maxsplit=1)[-1]
    return normalized.lower()


def _normalized_set(values: list[str]) -> set[str]:
    normalized_values: set[str] = set()
    for value in values:
        if value.strip():
            normalized_values.add(normalize_identifier(value))
    return normalized_values


def unique_ranked_identifiers(values: list[str]) -> list[str]:
    """Normalize IDs and drop repeats while keeping first-seen rank order.

    Several chunks from one paper produce the same paper ID; without this
    a single relevant paper is counted once per chunk and nDCG exceeds 1.
    """

    ranked: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value.strip():
            continue
        identifier = normalize_identifier(value)
        if identifier in seen:
            continue
        seen.add(identifier)
        ranked.append(identifier)
    return ranked


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Measure what fraction of the first k distinct retrieved items are relevant."""

    if not relevant:
        return None
    if k <= 0:
        raise ValueError("k must be positive.")

    relevant_ids = _normalized_set(relevant)
    selected = unique_ranked_identifiers(retrieved)[:k]
    if not selected:
        return 0.0

    hits = 0
    for identifier in selected:
        if identifier in relevant_ids:
            hits += 1
    return hits / len(selected)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Measure how much of the known relevant evidence appears in the first k."""

    if not relevant:
        return None

    relevant_ids = _normalized_set(relevant)
    retrieved_ids = set(unique_ranked_identifiers(retrieved)[:k])
    return len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float | None:
    """Reward a pipeline for placing its first relevant result near the top."""

    if not relevant:
        return None

    relevant_ids = _normalized_set(relevant)
    for rank, identifier in enumerate(unique_ranked_identifiers(retrieved), start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Measure ranking quality with binary relevance and position discounting."""

    if not relevant:
        return None

    relevant_ids = _normalized_set(relevant)
    dcg = 0.0
    for index, identifier in enumerate(unique_ranked_identifiers(retrieved)[:k]):
        if identifier in relevant_ids:
            dcg += 1.0 / math.log2(index + 2.0)

    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = 0.0
    for index in range(ideal_hits):
        ideal_dcg += 1.0 / math.log2(index + 2.0)

    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def citation_validity(output: PipelineOutput) -> float | None:
    """Measure whether every cited chunk was actually supplied to generation."""

    if not output.retrieval_used:
        return None
    if not output.cited_source_ids:
        return 0.0

    retrieved = _normalized_set(output.retrieved_source_ids)
    valid = 0
    for source_id in output.cited_source_ids:
        if normalize_identifier(source_id) in retrieved:
            valid += 1
    return valid / len(output.cited_source_ids)


def citation_recall(example: EvaluationExample, output: PipelineOutput) -> float | None:
    """Measure whether citations cover the benchmark's known relevant evidence."""

    if example.expected_source_ids:
        relevant = example.expected_source_ids
        cited = output.cited_source_ids
    elif example.expected_paper_ids:
        relevant = example.expected_paper_ids
        cited = output.cited_paper_ids
    else:
        return None

    relevant_ids = _normalized_set(relevant)
    cited_ids = _normalized_set(cited)
    return len(relevant_ids.intersection(cited_ids)) / len(relevant_ids)


def keyword_coverage(
    example: EvaluationExample, output: PipelineOutput
) -> float | None:
    """Check whether required concepts occur as complete, case-insensitive phrases."""

    if not example.must_contain:
        return None

    answer = " ".join(output.answer.lower().split())
    matches = 0
    for phrase in example.must_contain:
        clean_phrase = " ".join(phrase.lower().split())
        pattern = r"(?<!\w)" + re.escape(clean_phrase) + r"(?!\w)"
        if re.search(pattern, answer):
            matches += 1
    return matches / len(example.must_contain)


def route_accuracy(example: EvaluationExample, output: PipelineOutput) -> float:
    """Compare retrieval and web decisions with benchmark expectations."""

    retrieval_correct = output.retrieval_used == example.requires_retrieval
    web_correct = output.web_search_used == example.requires_web
    return (float(retrieval_correct) + float(web_correct)) / 2.0


def score_deterministic_metrics(
    example: EvaluationExample,
    output: PipelineOutput,
    k: int = 10,
) -> DeterministicScores:
    """Compute every model-free metric for one evaluation result."""

    if example.expected_source_ids:
        retrieved = output.retrieved_source_ids
        relevant = example.expected_source_ids
    else:
        retrieved = output.retrieved_paper_ids
        relevant = example.expected_paper_ids

    successful_statuses = {"accepted", "direct_answer", "complete"}
    if example.expects_abstention:
        # Inventing an answer to an unanswerable question is the failure here.
        successful_statuses = {"insufficient_evidence"}

    success = 0.0
    if (
        output.answer.strip()
        and not output.error
        and output.final_status in successful_statuses
    ):
        success = 1.0

    return DeterministicScores(
        precision_at_k=precision_at_k(retrieved, relevant, k),
        recall_at_k=recall_at_k(retrieved, relevant, k),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
        citation_validity=citation_validity(output),
        citation_recall=citation_recall(example, output),
        keyword_coverage=keyword_coverage(example, output),
        route_accuracy=route_accuracy(example, output),
        success=success,
    )
