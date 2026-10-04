from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agentic_rag.graph.baseline import (
    build_citations,
    retrieve_evidence,
)
from agentic_rag.graph.crag import run_crag_assessment
from agentic_rag.graph.query_rewriting import (
    rewrite_query,
    select_corrective_evidence,
)
from agentic_rag.graph.runtime import AgenticRAGRuntime
from agentic_rag.graph.schemas import (
    DirectAnswer,
    RetrievalDecision,
)
from agentic_rag.graph.self_rag import (
    check_hallucination,
    critique_answer,
    generate_answer,
    hallucination_feedback,
)
from agentic_rag.graph.state import (
    AgenticRAGState,
    documents_from_state,
    documents_to_state,
)
from agentic_rag.graph.web_fallback import (
    combine_web_and_corpus_evidence,
)

RETRIEVAL_ROUTER_SYSTEM_PROMPT = """
You are the retrieval router for a scientific research assistant.

Decide whether the user question requires external research evidence.

Set retrieve to true for:
- Questions about scientific facts, papers, methods, results, or comparisons.
- Questions about RAG, information retrieval, LLMs, or related research.
- Questions whose answer should be supported with research citations.

Set retrieve to false only for:
- Greetings or thanks.
- Simple arithmetic.
- Requests that only transform text already supplied by the user.

When uncertain, set retrieve to true. Do not answer the question.
""".strip()


DIRECT_ANSWER_SYSTEM_PROMPT = """
You are a concise assistant.

The retrieval router determined that research evidence is unnecessary.
Answer the user's request directly. Do not invent research citations.
""".strip()


class AgenticRAGNodes:
    """Node functions and routing decisions for the RAG graph."""

    def __init__(self, runtime: AgenticRAGRuntime) -> None:
        self.runtime = runtime

    def route_question(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Decide whether the question needs research retrieval."""

        decision = self.runtime.get_llm().invoke(
            system_prompt=RETRIEVAL_ROUTER_SYSTEM_PROMPT,
            user_prompt="User question:\n" + state["question"],
            response_model=RetrievalDecision,
        )

        return {
            "retrieval_needed": decision.retrieve,
            "router_reason": decision.reason,
        }

    def direct_answer(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Answer requests that do not require research evidence."""

        result = self.runtime.get_llm().invoke(
            system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
            user_prompt="User request:\n" + state["question"],
            response_model=DirectAnswer,
        )

        return {
            "answer": result.answer,
            "final_status": "direct_answer",
        }

    def retrieve(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Run hybrid retrieval and accumulate targeted evidence."""

        agent_config = self.runtime.agent_config
        new_documents = retrieve_evidence(
            question=state["retrieval_query"],
            retriever=self.runtime.get_retriever(),
            top_k=int(agent_config.get("retrieval_top_k", 10)),
        )

        existing_documents = documents_from_state(state.get("documents", []))

        if existing_documents:
            documents = select_corrective_evidence(
                existing=existing_documents,
                incoming=new_documents,
                maximum_documents=int(
                    agent_config.get(
                        "maximum_accumulated_documents",
                        12,
                    )
                ),
            )
        else:
            documents = new_documents

        return {
            "documents": documents_to_state(documents),
        }

    def grade_context(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Grade retrieved chunks and decide the CRAG route."""

        agent_config = self.runtime.agent_config
        documents = documents_from_state(state.get("documents", []))

        assessment = run_crag_assessment(
            question=state["question"],
            documents=documents,
            llm=self.runtime.get_llm(),
            maximum_grade_characters_per_document=int(
                agent_config.get(
                    "maximum_grader_characters_per_document",
                    1600,
                )
            ),
            maximum_context_characters_per_document=int(
                agent_config.get(
                    "maximum_context_characters_per_document",
                    2000,
                )
            ),
        )

        return {
            "documents": documents_to_state(assessment.selected_documents),
            "graded_documents": documents_to_state(assessment.graded_documents),
            "crag_route": assessment.route,
            "context_status": assessment.context_status,
            "context_reason": assessment.context_reason,
            "missing_information": (assessment.missing_information),
        }

    def rewrite_query(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Create a focused query for the missing evidence."""

        agent_config = self.runtime.agent_config
        documents = documents_from_state(state.get("documents", []))

        rewritten = rewrite_query(
            question=state["question"],
            current_query=state["retrieval_query"],
            documents=documents,
            missing_information=state.get(
                "missing_information",
                "",
            ),
            llm=self.runtime.get_llm(),
            maximum_characters_per_document=int(
                agent_config.get(
                    "maximum_rewrite_characters_per_document",
                    1200,
                )
            ),
        )

        query_history = list(state.get("query_history", []))
        rewrite_reasons = list(state.get("rewrite_reasons", []))

        query_history.append(rewritten.rewritten_query)
        rewrite_reasons.append(rewritten.reason)

        return {
            "retrieval_query": rewritten.rewritten_query,
            "query_history": query_history,
            "rewrite_reasons": rewrite_reasons,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def web_search(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Search Tavily once and add its evidence to corpus evidence."""

        web_config = self.runtime.web_config
        web_query = self._build_web_query(state)
        search_result = self.runtime.get_web_search().search(web_query)

        update: dict[str, Any] = {
            "web_search_used": True,
            "web_search_status": search_result.status,
            "web_search_error": search_result.error,
        }

        if search_result.status != "complete":
            return update

        corpus_documents = documents_from_state(state.get("documents", []))
        combined_documents = combine_web_and_corpus_evidence(
            corpus_documents=corpus_documents,
            web_documents=search_result.documents,
            maximum_documents=int(
                web_config.get(
                    "maximum_combined_documents",
                    12,
                )
            ),
        )

        update["documents"] = documents_to_state(combined_documents)
        update["retrieval_query"] = web_query
        return update

    def generate_answer(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Generate a citation-constrained answer from selected evidence."""

        agent_config = self.runtime.agent_config
        documents = documents_from_state(state.get("documents", []))

        generated, source_ids = generate_answer(
            question=state["question"],
            documents=documents,
            llm=self.runtime.get_llm(),
            feedback=state.get("improvement_feedback", ""),
            previous_answer=state.get("answer", ""),
            maximum_characters_per_document=int(
                agent_config.get(
                    "maximum_context_characters_per_document",
                    2400,
                )
            ),
        )

        return {
            "answer": generated.answer,
            "source_ids": source_ids,
            "generation_count": (state.get("generation_count", 0) + 1),
            "grounded": False,
            "useful": False,
            "needs_more_context": False,
            "improvement_feedback": "",
        }

    def check_hallucination(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Verify every factual answer claim against the evidence."""

        agent_config = self.runtime.agent_config
        documents = documents_from_state(state.get("documents", []))

        result = check_hallucination(
            question=state["question"],
            answer=state["answer"],
            documents=documents,
            llm=self.runtime.get_llm(),
            maximum_characters_per_document=int(
                agent_config.get(
                    "maximum_verifier_characters_per_document",
                    1800,
                )
            ),
        )

        update: dict[str, Any] = {
            "grounded": result.grounded,
            "hallucination_reason": result.reason,
            "unsupported_claims": result.unsupported_claims,
        }

        if not result.grounded:
            update["improvement_feedback"] = hallucination_feedback(result)

        return update

    def critique_answer(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Check whether the grounded answer fully answers the question."""

        agent_config = self.runtime.agent_config
        documents = documents_from_state(state.get("documents", []))

        result = critique_answer(
            question=state["question"],
            answer=state["answer"],
            documents=documents,
            llm=self.runtime.get_llm(),
            maximum_characters_per_document=int(
                agent_config.get(
                    "maximum_verifier_characters_per_document",
                    1800,
                )
            ),
        )

        citations = build_citations(
            source_ids=state.get("source_ids", []),
            documents=documents,
        )
        citation_records: list[dict[str, Any]] = []

        for citation in citations:
            citation_records.append(citation.model_dump(mode="json"))

        update: dict[str, Any] = {
            "useful": result.useful,
            "needs_more_context": result.needs_more_context,
            "critic_reason": result.reason,
            "improvement_feedback": result.improvement_feedback,
            "citations": citation_records,
        }

        if result.useful:
            update["final_status"] = "accepted"

        if result.needs_more_context:
            missing_information = result.improvement_feedback

            if not missing_information:
                missing_information = result.reason

            update["missing_information"] = missing_information

        return update

    def fallback_answer(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Return a safe result after retrieval or retry limits are reached."""

        if state.get("answer") and state.get("grounded", False):
            return {
                "final_status": "needs_more_context",
                "needs_more_context": True,
            }

        return {
            "answer": ("There is not enough supported evidence to answer the question reliably."),
            "source_ids": [],
            "citations": [],
            "final_status": "insufficient_evidence",
        }

    def route_after_question(
        self,
        state: AgenticRAGState,
    ) -> Literal["retrieve", "direct_answer"]:
        if state.get("retrieval_needed", True):
            return "retrieve"

        return "direct_answer"

    def route_after_grading(
        self,
        state: AgenticRAGState,
    ) -> Literal[
        "generate_answer",
        "rewrite_query",
        "web_search",
        "fallback_answer",
    ]:
        route = state.get("crag_route", "incorrect")

        if route == "correct":
            return "generate_answer"

        maximum_rewrites = int(
            self.runtime.agent_config.get(
                "maximum_rewrite_attempts",
                2,
            )
        )

        if route == "ambiguous" and state.get("rewrite_count", 0) < maximum_rewrites:
            return "rewrite_query"

        if self._web_search_available(state):
            return "web_search"

        return "fallback_answer"

    def route_after_hallucination_check(
        self,
        state: AgenticRAGState,
    ) -> Literal[
        "critique_answer",
        "generate_answer",
        "fallback_answer",
    ]:
        if state.get("grounded", False):
            return "critique_answer"

        maximum_generations = int(
            self.runtime.agent_config.get(
                "maximum_generation_attempts",
                2,
            )
        )

        if state.get("generation_count", 0) < maximum_generations:
            return "generate_answer"

        return "fallback_answer"

    def route_after_critique(
        self,
        state: AgenticRAGState,
    ) -> Literal[
        "end",
        "rewrite_query",
        "web_search",
        "generate_answer",
        "fallback_answer",
    ]:
        if state.get("useful", False):
            return "end"

        if state.get("needs_more_context", False):
            maximum_rewrites = int(
                self.runtime.agent_config.get(
                    "maximum_rewrite_attempts",
                    2,
                )
            )

            if state.get("rewrite_count", 0) < maximum_rewrites:
                return "rewrite_query"

            if self._web_search_available(state):
                return "web_search"

            return "fallback_answer"

        maximum_generations = int(
            self.runtime.agent_config.get(
                "maximum_generation_attempts",
                2,
            )
        )

        if state.get("generation_count", 0) < maximum_generations:
            return "generate_answer"

        return "fallback_answer"

    def _web_search_available(
        self,
        state: AgenticRAGState,
    ) -> bool:
        enabled = bool(self.runtime.web_config.get("enabled", True))

        return enabled and not state.get(
            "web_search_used",
            False,
        )

    @staticmethod
    def _build_web_query(state: AgenticRAGState) -> str:
        query = state.get("retrieval_query", "").strip()
        missing_information = state.get(
            "missing_information",
            "",
        ).strip()

        if not query:
            query = state["question"]

        normalized_query = " ".join(query.lower().split())
        normalized_missing = " ".join(missing_information.lower().split())

        if normalized_missing and normalized_missing not in normalized_query:
            query = query + " " + missing_information

        return query.strip()


def build_agentic_rag_graph(
    runtime: AgenticRAGRuntime | None = None,
):
    """Build and compile the complete cyclic Agentic RAG graph."""

    if runtime is None:
        runtime = AgenticRAGRuntime()

    nodes = AgenticRAGNodes(runtime)
    builder = StateGraph(AgenticRAGState)

    builder.add_node("route_question", nodes.route_question)
    builder.add_node("direct_answer", nodes.direct_answer)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade_context", nodes.grade_context)
    builder.add_node("rewrite_query", nodes.rewrite_query)
    builder.add_node("web_search", nodes.web_search)
    builder.add_node("generate_answer", nodes.generate_answer)
    builder.add_node(
        "check_hallucination",
        nodes.check_hallucination,
    )
    builder.add_node("critique_answer", nodes.critique_answer)
    builder.add_node("fallback_answer", nodes.fallback_answer)

    builder.add_edge(START, "route_question")

    builder.add_conditional_edges(
        "route_question",
        nodes.route_after_question,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer",
        },
    )

    builder.add_edge("direct_answer", END)
    builder.add_edge("retrieve", "grade_context")

    builder.add_conditional_edges(
        "grade_context",
        nodes.route_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
            "fallback_answer": "fallback_answer",
        },
    )

    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("web_search", "grade_context")
    builder.add_edge("generate_answer", "check_hallucination")

    builder.add_conditional_edges(
        "check_hallucination",
        nodes.route_after_hallucination_check,
        {
            "critique_answer": "critique_answer",
            "generate_answer": "generate_answer",
            "fallback_answer": "fallback_answer",
        },
    )

    builder.add_conditional_edges(
        "critique_answer",
        nodes.route_after_critique,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
            "generate_answer": "generate_answer",
            "fallback_answer": "fallback_answer",
        },
    )

    builder.add_edge("fallback_answer", END)

    return builder.compile()


# LangGraph Studio loads this module-level compiled graph.
default_runtime = AgenticRAGRuntime()
graph = build_agentic_rag_graph(default_runtime)
