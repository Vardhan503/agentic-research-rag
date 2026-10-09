from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from agentic_rag.evaluation.models import (
    EvaluationExample,
    PipelineOutput,
    RagasScores,
)


def build_sentence_transformer_embedding(model_name: str) -> Any:
    """Create a RAGAS embedding adapter around the project's local BGE model.

    The class is defined here so importing this module, and the evaluation
    runner that depends on it, does not require ragas to be installed.
    """

    from ragas.embeddings.base import BaseRagasEmbedding

    class SentenceTransformerRagasEmbedding(BaseRagasEmbedding):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.model_name = name
            self._model: Any | None = None

        def _get_model(self) -> Any:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            return self._model

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            vector = self._get_model().encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector.tolist()

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text, **kwargs)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            vectors = self._get_model().encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

        async def aembed_texts(
            self,
            texts: list[str],
            **kwargs: Any,
        ) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

    return SentenceTransformerRagasEmbedding(model_name)


def _metric_value(result: Any) -> float | None:
    """Convert a RAGAS MetricResult or numeric result into a safe float."""

    value = getattr(result, "value", result)
    if value is None:
        return None
    numeric_value = float(value)
    if math.isnan(numeric_value):
        return None
    return numeric_value


def build_judge_client(config: dict[str, Any]) -> tuple[Any, str]:
    """Return an AsyncOpenAI client and judge model for the configured provider.

    provider "openai" calls the OpenAI API with OPENAI_API_KEY; provider
    "ollama" calls Ollama's OpenAI-compatible endpoint at host + /v1.
    """

    from openai import AsyncOpenAI

    provider = str(config.get("provider", "ollama")).strip().lower()
    timeout = float(config.get("timeout_seconds", 300))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set; it is required for the OpenAI RAGAS judge."
            )
        client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        return client, str(config.get("openai_model", "gpt-4.1-mini"))

    if provider == "ollama":
        host = str(config.get("host", "http://localhost:11434")).rstrip("/")
        client = AsyncOpenAI(
            base_url=host + "/v1",
            api_key=str(config.get("api_key", "ollama")),
            timeout=timeout,
        )
        return client, str(config.get("model", "qwen3:14b"))

    raise ValueError("Unsupported RAGAS judge provider: " + provider)


# RAGAS defaults the judge to 1024 output tokens. Faithfulness lists every
# claim in the answer and then a verdict per claim, so long cited answers
# overflow that and the metric is dropped with a max_tokens error.
DEFAULT_JUDGE_MAX_TOKENS = 4096


def judge_model_args(config: dict[str, Any]) -> dict[str, Any]:
    """Return the generation settings passed to the RAGAS judge model."""

    max_tokens = int(config.get("judge_max_tokens", DEFAULT_JUDGE_MAX_TOKENS))

    if max_tokens <= 0:
        raise ValueError("judge_max_tokens must be positive.")

    return {
        "max_tokens": max_tokens,
        "temperature": float(config.get("judge_temperature", 0.0)),
    }


class RagasEvaluator:
    """Run selected RAGAS metrics with an OpenAI or local Ollama judge."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.metric_names = list(
            config.get(
                "metrics",
                [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "factual_correctness",
                ],
            )
        )
        self.maximum_contexts = int(config.get("maximum_contexts", 5))
        self.maximum_context_characters = int(
            config.get("maximum_context_characters", 2000)
        )
        self._metrics: dict[str, Any] | None = None

    def _build_metrics(self) -> dict[str, Any]:
        """Create metric objects lazily so normal tests do not load models."""

        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        client, judge_model = build_judge_client(self.config)
        llm = llm_factory(
            judge_model,
            provider="openai",
            client=client,
            **judge_model_args(self.config),
        )
        embeddings = build_sentence_transformer_embedding(
            str(
                self.config.get(
                    "embedding_model",
                    "BAAI/bge-small-en-v1.5",
                )
            )
        )

        return {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,
                strictness=int(self.config.get("answer_relevancy_strictness", 1)),
            ),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "factual_correctness": FactualCorrectness(llm=llm, mode="f1"),
        }

    def _get_metrics(self) -> dict[str, Any]:
        if self._metrics is None:
            self._metrics = self._build_metrics()
        return self._metrics

    def _limited_contexts(self, output: PipelineOutput) -> list[str]:
        contexts: list[str] = []
        for context in output.contexts[: self.maximum_contexts]:
            contexts.append(context[: self.maximum_context_characters])
        return contexts

    async def evaluate(
        self,
        example: EvaluationExample,
        output: PipelineOutput,
        metric_names: list[str] | None = None,
    ) -> RagasScores:
        """Evaluate one answer while isolating failures to individual metrics.

        metric_names restricts the run to a subset, used when rescoring only
        the metrics a previous judgement lost to errors.
        """

        scores = RagasScores()
        if output.error or not output.answer.strip():
            scores.errors["evaluation"] = "Pipeline did not produce an answer."
            return scores

        contexts = self._limited_contexts(output)
        metrics = self._get_metrics()

        for metric_name in metric_names if metric_names is not None else self.metric_names:
            metric = metrics.get(metric_name)
            if metric is None:
                scores.errors[metric_name] = "Unknown RAGAS metric."
                continue

            try:
                value = await self._score_metric(
                    metric_name=metric_name,
                    metric=metric,
                    example=example,
                    output=output,
                    contexts=contexts,
                )
                setattr(scores, metric_name, value)
            except Exception as error:  # noqa: BLE001 - isolate metric failures
                scores.errors[metric_name] = str(error)

        return scores

    async def _score_metric(
        self,
        metric_name: str,
        metric: Any,
        example: EvaluationExample,
        output: PipelineOutput,
        contexts: list[str],
    ) -> float | None:
        if metric_name == "answer_relevancy":
            result = await metric.ascore(
                user_input=example.question,
                response=output.answer,
            )
            return _metric_value(result)

        if metric_name == "factual_correctness":
            if example.reference_mode == "dynamic":
                return None
            result = await metric.ascore(
                response=output.answer,
                reference=example.reference_answer,
            )
            return _metric_value(result)

        if not contexts:
            return None

        if metric_name == "faithfulness":
            result = await metric.ascore(
                user_input=example.question,
                response=output.answer,
                retrieved_contexts=contexts,
            )
            return _metric_value(result)

        if metric_name == "context_precision":
            if example.reference_mode == "dynamic":
                return None
            result = await metric.ascore(
                user_input=example.question,
                reference=example.reference_answer,
                retrieved_contexts=contexts,
            )
            return _metric_value(result)

        if metric_name == "context_recall":
            if example.reference_mode == "dynamic":
                return None
            result = await metric.ascore(
                user_input=example.question,
                retrieved_contexts=contexts,
                reference=example.reference_answer,
            )
            return _metric_value(result)

        return None
