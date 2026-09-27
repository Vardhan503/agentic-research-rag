from pydantic import BaseModel, Field

from agentic_rag.processing.models import DocumentChunk


class RetrievalCandidate(BaseModel):
    """One chunk returned by one or more retrievers."""

    chunk: DocumentChunk

    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    sources: list[str] = Field(default_factory=list)