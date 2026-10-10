from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvaluationExample(BaseModel):
    """One human-authored benchmark question and its expected behaviour."""

    question_id: str
    question: str
    reference_answer: str
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["validation", "test"] = "test"
    reference_mode: Literal["static", "dynamic"] = "static"
    expected_paper_ids: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    # Phrases that must not appear. Used so a stale 2024 paper cannot score
    # full keyword coverage on a "last seven days" question.
    must_not_contain: list[str] = Field(default_factory=list)
    requires_retrieval: bool = True
    requires_web: bool = False
    # True for questions with no answer in any source: the correct outcome is
    # a clear "insufficient evidence" rather than a confident answer.
    expects_abstention: bool = False
    notes: str = ""

    @field_validator("question_id", "question", "reference_answer", "category")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("Evaluation text fields cannot be empty.")
        return clean_value


class PipelineOutput(BaseModel):
    """Normalized result returned by any evaluated RAG pipeline."""

    pipeline: str
    question_id: str
    answer: str = ""
    contexts: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    retrieved_paper_ids: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    cited_paper_ids: list[str] = Field(default_factory=list)
    final_status: str = "unknown"
    crag_route: str = "not_used"
    retrieval_used: bool = False
    web_search_used: bool = False
    rewrite_count: int = 0
    generation_count: int = 0
    grounded: bool | None = None
    useful: bool | None = None
    latency_seconds: float = 0.0
    error: str = ""


class DeterministicScores(BaseModel):
    """Scores computed directly from IDs, text, routes, and status."""

    precision_at_k: float | None = None
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    ndcg_at_k: float | None = None
    citation_validity: float | None = None
    citation_recall: float | None = None
    keyword_coverage: float | None = None
    route_accuracy: float | None = None
    success: float = 0.0


class RagasScores(BaseModel):
    """Semantic scores produced by RAGAS and the local evaluator model."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    factual_correctness: float | None = None
    errors: dict[str, str] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    """Complete stored result for one pipeline-question pair."""

    example: EvaluationExample
    output: PipelineOutput
    deterministic: DeterministicScores
    ragas: RagasScores | None = None


class PipelineSummary(BaseModel):
    """Aggregate metrics for one evaluated pipeline."""

    pipeline: str
    total_examples: int
    completed_examples: int
    ragas_completed_examples: int = 0
    failed_examples: int
    mean_latency_seconds: float | None = None
    status_distribution: dict[str, int] = Field(default_factory=dict)
    deterministic_means: dict[str, float | None] = Field(default_factory=dict)
    # Number of records each mean is based on; retrieval metrics only
    # exist for questions labelled with expected paper or source IDs.
    deterministic_counts: dict[str, int] = Field(default_factory=dict)
    ragas_means: dict[str, float | None] = Field(default_factory=dict)
    ragas_counts: dict[str, int] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Top-level comparison report written after every evaluation run."""

    status: Literal["partial", "complete"]
    evaluated_at: str
    dataset_path: str
    requested_pipelines: list[str]
    total_examples: int
    records_completed: int
    ragas_enabled: bool
    ragas_records_completed: int = 0
    summaries: list[PipelineSummary] = Field(default_factory=list)
