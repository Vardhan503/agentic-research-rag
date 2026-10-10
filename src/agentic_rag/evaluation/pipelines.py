from __future__ import annotations

import time
from typing import Any

from agentic_rag.evaluation.models import EvaluationExample, PipelineOutput
from agentic_rag.graph.baseline import generate_grounded_answer, run_baseline_rag
from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.query_rewriting import run_corrective_retrieval
from agentic_rag.graph.runtime import AgenticRAGRuntime
from agentic_rag.graph.state import create_initial_state
from agentic_rag.graph.web_fallback import run_web_fallback
from agentic_rag.graph.workflow import build_agentic_rag_graph

SUPPORTED_PIPELINES = ("baseline", "crag", "crag_web", "agentic")


class EvaluationPipelineRunner:
    """Run several RAG designs through one normalized evaluation interface."""

    def __init__(self, runtime: AgenticRAGRuntime | None = None) -> None:
        self.runtime = runtime or AgenticRAGRuntime()
        self._graph: Any | None = None

    def close(self) -> None:
        self.runtime.close()

    def run(self, pipeline: str, example: EvaluationExample) -> PipelineOutput:
        """Dispatch one benchmark example and convert failures into records."""

        started_at = time.perf_counter()
        try:
            if pipeline == "baseline":
                output = self._run_baseline(example)
            elif pipeline == "crag":
                output = self._run_crag(example, use_web=False)
            elif pipeline == "crag_web":
                output = self._run_crag(example, use_web=True)
            elif pipeline == "agentic":
                output = self._run_agentic(example)
            else:
                raise ValueError("Unsupported pipeline: " + pipeline)
        except Exception as error:
            output = PipelineOutput(
                pipeline=pipeline,
                question_id=example.question_id,
                final_status="failed",
                error=type(error).__name__ + ": " + str(error),
            )

        output.latency_seconds = time.perf_counter() - started_at
        return output

    def _run_baseline(self, example: EvaluationExample) -> PipelineOutput:
        config = self.runtime.agent_config
        result = run_baseline_rag(
            question=example.question,
            retriever=self.runtime.get_retriever(),
            llm=self.runtime.get_llm(),
            top_k=int(config.get("retrieval_top_k", 10)),
            maximum_characters_per_document=int(
                config.get("maximum_context_characters_per_document", 2400)
            ),
        )
        status = (
            "accepted"
            if result.documents and result.answer
            else "insufficient_evidence"
        )
        return self._from_documents(
            pipeline="baseline",
            example=example,
            answer=result.answer,
            documents=result.documents,
            cited_source_ids=result.source_ids,
            final_status=status,
            retrieval_used=True,
            generation_count=1 if result.answer else 0,
        )

    def _run_crag(
        self,
        example: EvaluationExample,
        use_web: bool,
    ) -> PipelineOutput:
        config = self.runtime.agent_config
        corrective = run_corrective_retrieval(
            question=example.question,
            retriever=self.runtime.get_retriever(),
            llm=self.runtime.get_llm(),
            top_k=int(config.get("retrieval_top_k", 10)),
            maximum_rewrite_attempts=int(config.get("maximum_rewrite_attempts", 2)),
            maximum_grade_characters_per_document=int(
                config.get("maximum_grader_characters_per_document", 1600)
            ),
            maximum_context_characters_per_document=int(
                config.get("maximum_context_characters_per_document", 2000)
            ),
            maximum_rewrite_characters_per_document=int(
                config.get("maximum_rewrite_characters_per_document", 1200)
            ),
            maximum_accumulated_documents=int(
                config.get("maximum_accumulated_documents", 12)
            ),
        )

        documents = corrective.documents
        route = corrective.assessment.route
        web_used = False

        if use_web:
            web_config = self.runtime.web_config
            web_result = run_web_fallback(
                question=example.question,
                corrective_result=corrective,
                web_search=self.runtime.get_web_search(),
                llm=self.runtime.get_llm(),
                enabled=bool(web_config.get("enabled", True)),
                maximum_documents=int(web_config.get("maximum_combined_documents", 12)),
                maximum_grade_characters_per_document=int(
                    config.get("maximum_grader_characters_per_document", 1600)
                ),
                maximum_context_characters_per_document=int(
                    config.get("maximum_context_characters_per_document", 2000)
                ),
            )
            documents = web_result.documents
            route = web_result.assessment.route
            web_used = web_result.used

        if not documents:
            return self._from_documents(
                pipeline="crag_web" if use_web else "crag",
                example=example,
                answer="There is not enough supported evidence to answer reliably.",
                documents=[],
                cited_source_ids=[],
                final_status="insufficient_evidence",
                retrieval_used=True,
                web_search_used=web_used,
                crag_route=route,
                rewrite_count=corrective.rewrite_count,
            )

        generated, source_ids = generate_grounded_answer(
            question=example.question,
            documents=documents,
            llm=self.runtime.get_llm(),
            maximum_characters_per_document=int(
                config.get("maximum_context_characters_per_document", 2400)
            ),
        )
        final_status = "accepted" if route == "correct" else "needs_more_context"
        return self._from_documents(
            pipeline="crag_web" if use_web else "crag",
            example=example,
            answer=generated.answer,
            documents=documents,
            cited_source_ids=source_ids,
            final_status=final_status,
            retrieval_used=True,
            web_search_used=web_used,
            crag_route=route,
            rewrite_count=corrective.rewrite_count,
            generation_count=1,
        )

    def _run_agentic(self, example: EvaluationExample) -> PipelineOutput:
        if self._graph is None:
            self._graph = build_agentic_rag_graph(self.runtime)

        recursion_limit = int(
            self.runtime.agent_config.get("graph_recursion_limit", 40)
        )
        state = self._graph.invoke(
            create_initial_state(example.question),
            config={"recursion_limit": recursion_limit},
        )
        documents: list[EvidenceDocument] = []
        for record in state.get("documents", []):
            documents.append(EvidenceDocument.model_validate(record))

        return self._from_documents(
            pipeline="agentic",
            example=example,
            answer=str(state.get("answer", "")),
            documents=documents,
            cited_source_ids=list(state.get("source_ids", [])),
            final_status=str(state.get("final_status", "unknown")),
            retrieval_used=bool(state.get("retrieval_needed", True)),
            web_search_used=bool(state.get("web_search_used", False)),
            crag_route=str(state.get("crag_route", "not_used")),
            rewrite_count=int(state.get("rewrite_count", 0)),
            generation_count=int(state.get("generation_count", 0)),
            grounded=bool(state.get("grounded", False)),
            useful=bool(state.get("useful", False)),
        )

    @staticmethod
    def _from_documents(
        pipeline: str,
        example: EvaluationExample,
        answer: str,
        documents: list[EvidenceDocument],
        cited_source_ids: list[str],
        final_status: str,
        retrieval_used: bool,
        web_search_used: bool = False,
        crag_route: str = "not_used",
        rewrite_count: int = 0,
        generation_count: int = 0,
        grounded: bool | None = None,
        useful: bool | None = None,
    ) -> PipelineOutput:
        paper_by_source: dict[str, str] = {}
        contexts: list[str] = []
        retrieved_source_ids: list[str] = []
        retrieved_paper_ids: list[str] = []

        for document in documents:
            paper_by_source[document.source_id] = document.paper_id
            contexts.append(document.text)
            retrieved_source_ids.append(document.source_id)
            retrieved_paper_ids.append(document.paper_id)

        cited_paper_ids: list[str] = []
        for source_id in cited_source_ids:
            paper_id = paper_by_source.get(source_id)
            if paper_id is not None:
                cited_paper_ids.append(paper_id)

        return PipelineOutput(
            pipeline=pipeline,
            question_id=example.question_id,
            answer=answer,
            contexts=contexts,
            retrieved_source_ids=retrieved_source_ids,
            retrieved_paper_ids=retrieved_paper_ids,
            cited_source_ids=cited_source_ids,
            cited_paper_ids=cited_paper_ids,
            final_status=final_status,
            crag_route=crag_route,
            retrieval_used=retrieval_used,
            web_search_used=web_search_used,
            rewrite_count=rewrite_count,
            generation_count=generation_count,
            grounded=grounded,
            useful=useful,
        )
