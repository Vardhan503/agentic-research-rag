"""Build pooled, reference-anchored retrieval gold labels for the benchmark.

Retrieval metrics need to know which papers *should* have been retrieved.
Judging that by hand for 50 questions x 20 candidates is slow, so this module
follows the standard pooling approach: gather the top-k hybrid candidates for
each question, then judge every candidate against the benchmark's
human-written reference answer. A judge that sees the reference answer is
solving a different, easier task than the pipeline's grader, which only sees
the question, so the labels are not simply the grader agreeing with itself.

Labels are stored at paper level because chunk IDs change on every reindex.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.evaluation.models import EvaluationExample
from agentic_rag.graph.crag import resolve_source_id
from agentic_rag.retrieval.models import RetrievalCandidate

RELEVANCE_JUDGE_SYSTEM_PROMPT = """
You are building relevance judgments for a retrieval benchmark over scientific
papers.

You are given a question, the correct reference answer, and candidate text
chunks. For every candidate decide whether it contains information that
supports some part of the reference answer. The chunk does not need to cover
the whole answer, but it must contribute evidence for at least one claim in it.

Mark a chunk relevant when it states, explains, or demonstrates a fact that
appears in the reference answer.
Mark a chunk not relevant when it is only about the same broad topic, merely
mentions a term from the question, or contradicts the reference answer.

Rules:
- Return exactly one judgment for every supplied Source ID, copied exactly.
- Keep each reason to one short sentence.
- Do not answer the question.
""".strip()

JUDGMENT_TOKENS_PER_ITEM = 90
JUDGMENT_BASE_TOKENS = 64


class RelevanceJudgment(BaseModel):
    source_id: str = Field(min_length=1)
    relevant: bool
    reason: str = Field(min_length=3, max_length=300)


class RelevanceJudgmentBatch(BaseModel):
    judgments: list[RelevanceJudgment] = Field(min_length=1)


class CandidateJudgment(BaseModel):
    """One pooled candidate with its judgment, kept for review."""

    rank: int
    source_id: str
    paper_id: str
    title: str
    section_heading: str = ""
    relevant: bool | None = None
    reason: str = ""


class QuestionJudgments(BaseModel):
    question_id: str
    question: str
    reference_answer: str
    candidates: list[CandidateJudgment]

    @property
    def relevant_paper_ids(self) -> list[str]:
        """Papers with at least one relevant chunk, in first-seen rank order."""

        paper_ids: list[str] = []

        for candidate in self.candidates:
            if candidate.relevant and candidate.paper_id not in paper_ids:
                paper_ids.append(candidate.paper_id)

        return paper_ids


def build_judgment_prompt(
    example: EvaluationExample,
    candidates: list[RetrievalCandidate],
    maximum_characters_per_chunk: int = 900,
) -> str:
    """Show the question, reference answer, and every pooled chunk."""

    if not candidates:
        raise ValueError("At least one candidate is required.")

    blocks: list[str] = []

    for candidate in candidates:
        chunk = candidate.chunk
        text = chunk.text.strip()

        if len(text) > maximum_characters_per_chunk:
            text = text[:maximum_characters_per_chunk].rstrip() + "..."

        blocks.append(
            "Source ID: "
            + chunk.chunk_id
            + "\nTitle: "
            + chunk.title
            + "\nSection: "
            + (chunk.section_heading or "")
            + "\nText: "
            + text
        )

    return (
        "Question:\n"
        + example.question.strip()
        + "\n\nReference answer:\n"
        + example.reference_answer.strip()
        + "\n\nCandidate chunks:\n"
        + "\n\n---\n\n".join(blocks)
    )


def judge_candidates(
    example: EvaluationExample,
    candidates: list[RetrievalCandidate],
    llm: Any,
) -> QuestionJudgments:
    """Judge one question's pooled candidates in a single structured call."""

    judged = [
        CandidateJudgment(
            rank=rank,
            source_id=candidate.chunk.chunk_id,
            paper_id=candidate.chunk.paper_id,
            title=candidate.chunk.title,
            section_heading=candidate.chunk.section_heading,
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]
    result = QuestionJudgments(
        question_id=example.question_id,
        question=example.question,
        reference_answer=example.reference_answer,
        candidates=judged,
    )

    if not candidates:
        return result

    batch = llm.invoke(
        system_prompt=RELEVANCE_JUDGE_SYSTEM_PROMPT,
        user_prompt=build_judgment_prompt(example, candidates),
        response_model=RelevanceJudgmentBatch,
        num_predict=JUDGMENT_BASE_TOKENS + JUDGMENT_TOKENS_PER_ITEM * len(candidates),
    )

    by_source_id = {candidate.source_id: candidate for candidate in judged}
    expected_ids = set(by_source_id)

    for judgment in batch.judgments:
        source_id = resolve_source_id(judgment.source_id, expected_ids)

        if source_id is None:
            continue

        candidate = by_source_id[source_id]

        if candidate.relevant is None:
            candidate.relevant = judgment.relevant
            candidate.reason = judgment.reason

    # A chunk the judge skipped is treated as not relevant: gold labels must
    # only contain positive evidence that was actually confirmed.
    for candidate in judged:
        if candidate.relevant is None:
            candidate.relevant = False
            candidate.reason = "No judgment returned."

    return result


def write_judgments(path: str | Path, judgments: list[QuestionJudgments]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for item in judgments:
            output_file.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")


def load_judgments(path: str | Path) -> list[QuestionJudgments]:
    judgments: list[QuestionJudgments] = []

    with Path(path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                judgments.append(QuestionJudgments.model_validate_json(line))

    return judgments


def apply_gold_labels(
    benchmark_path: str | Path,
    judgments: list[QuestionJudgments],
    note: str,
) -> int:
    """Merge pooled relevant papers into the benchmark's expected_paper_ids.

    Existing hand-made labels are kept first and pooled papers are appended,
    so a judging run can only add gold, never erase it. Questions with no
    relevant pooled paper are left untouched.
    """

    relevant_by_question = {
        item.question_id: item.relevant_paper_ids for item in judgments
    }
    path = Path(benchmark_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    updated = 0

    for line in lines:
        if not line.strip():
            continue

        record = json.loads(line)
        paper_ids = relevant_by_question.get(record["question_id"])

        if paper_ids:
            merged = list(record.get("expected_paper_ids") or [])

            for paper_id in paper_ids:
                if paper_id not in merged:
                    merged.append(paper_id)

            record["expected_paper_ids"] = merged
            existing_note = str(record.get("notes") or "").strip()

            if note and note not in existing_note:
                record["notes"] = (existing_note + " " + note).strip()

            updated += 1

        updated_lines.append(json.dumps(record, ensure_ascii=False))

    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return updated
