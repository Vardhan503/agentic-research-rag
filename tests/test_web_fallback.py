from types import SimpleNamespace
from typing import Any

from agentic_rag.graph.crag import CRAGAssessment
from agentic_rag.graph.documents import EvidenceDocument
from agentic_rag.graph.query_rewriting import (
    CorrectiveRetrievalResult,
)
from agentic_rag.graph.web_fallback import run_web_fallback
from agentic_rag.graph.web_search import (
    TavilyWebSearch,
    convert_tavily_results,
)
from agentic_rag.llm.ollama_client import OllamaStructuredClient


class FakeTavilyClient:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(str(kwargs["query"]))
        return {"results": self.results}


class SequenceOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def chat(self, **_kwargs: Any):
        response = self.responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(message=SimpleNamespace(content=response))


def create_corrective_result(
    route: str,
    documents: list[EvidenceDocument] | None = None,
) -> CorrectiveRetrievalResult:
    if documents is None:
        documents = []

    context_status = "irrelevant"

    if route == "correct":
        context_status = "sufficient"

    if route == "ambiguous":
        context_status = "incomplete"

    assessment = CRAGAssessment(
        route=route,
        selected_documents=documents,
        context_status=context_status,
        context_reason="Test assessment.",
        missing_information="Current external evidence.",
    )

    return CorrectiveRetrievalResult(
        original_question="What is the latest RAG method?",
        final_query="latest corrective RAG method",
        query_history=["latest corrective RAG method"],
        documents=documents,
        assessment=assessment,
    )


def create_llm(responses: list[str]):
    client = SequenceOllamaClient(responses)
    llm = OllamaStructuredClient(
        model="qwen3:14b",
        max_retries=1,
        client=client,
    )
    return llm, client


def test_tavily_results_become_stable_web_evidence() -> None:
    raw_results = [
        {
            "title": "Corrective RAG update",
            "url": "https://example.org/crag",
            "content": "A new corrective retrieval method.",
        },
        {
            "title": "Duplicate",
            "url": "https://example.org/crag",
            "content": "Duplicate content.",
        },
    ]

    documents = convert_tavily_results(raw_results)

    assert len(documents) == 1
    assert documents[0].source == "web"
    assert documents[0].source_id.startswith("web-")
    assert documents[0].url == "https://example.org/crag"


def test_tavily_published_date_reaches_the_grader_prompt() -> None:
    documents = convert_tavily_results(
        [
            {
                "title": "New agentic RAG benchmark",
                "url": "https://example.org/agentic-rag",
                "content": "A benchmark for agentic retrieval.",
                "published_date": "2026-10-06",
            },
            {
                "title": "Undated page",
                "url": "https://example.org/undated",
                "content": "No date is available.",
            },
        ]
    )

    assert documents[0].published_date == "2026-10-06"
    assert "Published: 2026-10-06" in documents[0].prompt_text()
    assert documents[1].published_date is None
    assert "Published:" not in documents[1].prompt_text()


def test_missing_api_key_is_nonfatal() -> None:
    web_search = TavilyWebSearch(api_key=None, client=None)
    web_search.api_key = None

    result = web_search.search("corrective RAG")

    assert result.status == "unavailable"
    assert result.documents == []


def test_correct_route_skips_web_search() -> None:
    corrective_result = create_corrective_result("correct")
    fake_tavily = FakeTavilyClient([])
    web_search = TavilyWebSearch(client=fake_tavily)
    llm, client = create_llm([])

    result = run_web_fallback(
        question="What is corrective RAG?",
        corrective_result=corrective_result,
        web_search=web_search,
        llm=llm,
    )

    assert result.status == "not_needed"
    assert result.used is False
    assert fake_tavily.queries == []
    assert client.call_count == 0


def test_incorrect_route_searches_and_regrades_web_evidence() -> None:
    corrective_result = create_corrective_result("incorrect")
    raw_results = [
        {
            "title": "Corrective RAG research",
            "url": "https://example.org/research",
            "content": (
                "The system detects retrieval failure and rewrites "
                "the query before retrieving again."
            ),
        }
    ]
    fake_tavily = FakeTavilyClient(raw_results)
    web_search = TavilyWebSearch(client=fake_tavily)

    web_source_id = convert_tavily_results(raw_results)[0].source_id
    grading_response = (
        '{"grades":[{"source_id":"'
        + web_source_id
        + '","grade":"correct",'
        + '"reason":"Explains corrective retrieval."}]}'
    )
    context_response = """
    {
      "status": "sufficient",
      "reason": "The web evidence answers the question.",
      "missing_information": ""
    }
    """
    llm, client = create_llm([grading_response, context_response])

    result = run_web_fallback(
        question="How does corrective RAG recover?",
        corrective_result=corrective_result,
        web_search=web_search,
        llm=llm,
    )

    assert result.used is True
    assert result.status == "complete"
    assert result.assessment.route == "correct"
    assert len(result.web_documents) == 1
    assert client.call_count == 2
