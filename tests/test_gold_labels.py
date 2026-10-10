import json
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.gold_labels import (
    CandidateJudgment,
    QuestionJudgments,
    RelevanceJudgment,
    RelevanceJudgmentBatch,
    apply_gold_labels,
    judge_candidates,
)
from agentic_rag.evaluation.models import EvaluationExample
from agentic_rag.processing.models import DocumentChunk
from agentic_rag.retrieval.models import RetrievalCandidate


class FakeLLM:
    def __init__(self, batch: RelevanceJudgmentBatch) -> None:
        self.batch = batch
        self.prompts: list[str] = []

    def invoke(self, **kwargs: Any) -> RelevanceJudgmentBatch:
        self.prompts.append(kwargs["user_prompt"])
        return self.batch


def make_candidate(chunk_id: str, paper_id: str, text: str) -> RetrievalCandidate:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title="Paper " + paper_id,
        section_id=paper_id + "-s1",
        section_heading="Methods",
        section_type="methods",
        section_order=1,
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
        character_count=len(text),
        source_format="grobid_tei",
        source_path=paper_id + ".tei.xml",
    )
    return RetrievalCandidate(chunk=chunk, dense_score=0.5, sources=["dense"])


def make_example() -> EvaluationExample:
    return EvaluationExample(
        question_id="q1",
        question="How does CRAG grade evidence?",
        reference_answer="A T5-large evaluator scores documents against two thresholds.",
        category="corrective_rag",
        difficulty="hard",
    )


def test_judge_candidates_labels_papers_from_chunk_judgments() -> None:
    candidates = [
        make_candidate("W1-chunk-aaaa1111", "W1", "T5-large scores each document."),
        make_candidate("W1-chunk-bbbb2222", "W1", "Unrelated background."),
        make_candidate("W2-chunk-cccc3333", "W2", "Dense retrieval overview."),
    ]
    llm = FakeLLM(
        RelevanceJudgmentBatch(
            judgments=[
                # Truncated ID is repaired; the skipped chunk defaults to not relevant.
                RelevanceJudgment(source_id="W1-chunk-aaaa11", relevant=True, reason="States the evaluator."),
                RelevanceJudgment(source_id="W2-chunk-cccc3333", relevant=False, reason="Off topic."),
            ]
        )
    )

    judged = judge_candidates(make_example(), candidates, llm)

    assert "Reference answer:" in llm.prompts[0]
    assert judged.relevant_paper_ids == ["W1"]
    assert [c.relevant for c in judged.candidates] == [True, False, False]
    assert judged.candidates[1].reason == "No judgment returned."


def test_apply_gold_labels_merges_and_never_erases(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                json.dumps({"question_id": "q1", "expected_paper_ids": ["W-hand"]}),
                json.dumps({"question_id": "q2", "expected_paper_ids": ["W-keep"]}),
            ]
        )
        + "\n"
    )
    judgments = [
        QuestionJudgments(
            question_id="q1",
            question="q",
            reference_answer="a",
            candidates=[
                CandidateJudgment(rank=1, source_id="s1", paper_id="W-hand", title="t", relevant=True),
                CandidateJudgment(rank=2, source_id="s2", paper_id="W-new", title="t", relevant=True),
            ],
        ),
        QuestionJudgments(
            question_id="q2",
            question="q",
            reference_answer="a",
            candidates=[
                CandidateJudgment(rank=1, source_id="s3", paper_id="W-x", title="t", relevant=False),
            ],
        ),
    ]

    updated = apply_gold_labels(benchmark, judgments, note="pooled")

    rows = [json.loads(line) for line in benchmark.read_text().splitlines()]
    assert updated == 1
    assert rows[0]["expected_paper_ids"] == ["W-hand", "W-new"]
    assert rows[0]["notes"] == "pooled"
    # No relevant pooled paper: the hand label and the absence of a note are untouched.
    assert rows[1]["expected_paper_ids"] == ["W-keep"]
    assert "notes" not in rows[1]
