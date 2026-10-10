from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from agentic_rag.llm.factory import create_llm_client
from agentic_rag.llm.ollama_client import OllamaStructuredClient
from agentic_rag.llm.openai_client import OpenAIStructuredClient


class Verdict(BaseModel):
    grade: str
    reason: str


class FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        content = self.contents[len(self.calls) - 1]
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    def __init__(self, contents: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(contents))


def test_factory_defaults_to_ollama() -> None:
    llm = create_llm_client({"model": "qwen3:14b"}, client=object())

    assert isinstance(llm, OllamaStructuredClient)


def test_factory_creates_openai_client() -> None:
    llm = create_llm_client(
        {"provider": "openai", "openai_model": "gpt-4.1-mini"},
        client=FakeOpenAI([]),
    )

    assert isinstance(llm, OpenAIStructuredClient)
    assert llm.model == "gpt-4.1-mini"


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client({"provider": "anthropic"})


def test_openai_client_sends_schema_and_validates() -> None:
    fake = FakeOpenAI(['{"grade": "correct", "reason": "Direct evidence."}'])
    llm = OpenAIStructuredClient(model="gpt-4.1-mini", client=fake)

    result = llm.invoke(
        system_prompt="Grade it.",
        user_prompt="Evidence.",
        response_model=Verdict,
        num_predict=1664,
    )

    assert result.grade == "correct"

    call = fake.chat.completions.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["max_completion_tokens"] == 1664
    assert call["response_format"]["json_schema"]["name"] == "Verdict"
    assert "grade" in call["response_format"]["json_schema"]["schema"]["properties"]


def test_openai_client_retries_invalid_json() -> None:
    fake = FakeOpenAI(
        [
            '{"grade": "correct"',
            '{"grade": "ambiguous", "reason": "Indirect evidence."}',
        ]
    )
    llm = OpenAIStructuredClient(
        model="gpt-4.1-mini",
        retry_delay_seconds=0,
        client=fake,
    )

    result = llm.invoke(
        system_prompt="Grade it.",
        user_prompt="Evidence.",
        response_model=Verdict,
    )

    assert result.grade == "ambiguous"
    assert len(fake.chat.completions.calls) == 2


class ShortReason(BaseModel):
    status: str
    reason: str = Field(max_length=10)


def test_openai_client_trims_overlong_strings_without_retry() -> None:
    fake = FakeOpenAI(['{"status": "sufficient", "reason": "' + "x" * 50 + '"}'])
    llm = OpenAIStructuredClient(model="gpt-4.1-mini", client=fake)

    result = llm.invoke(
        system_prompt="Assess it.",
        user_prompt="Evidence.",
        response_model=ShortReason,
    )

    assert result.reason == "x" * 10
    assert len(fake.chat.completions.calls) == 1


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIStructuredClient(model="gpt-4.1-mini")
