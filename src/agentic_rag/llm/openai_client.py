from __future__ import annotations

import json
import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def truncate_overlong_strings[ResponseModel: BaseModel](
    content: str,
    response_model: type[ResponseModel],
) -> ResponseModel:
    """Validate content, trimming strings that exceed a schema maxLength.

    OpenAI treats maxLength as a hint rather than a decoding constraint, so
    a model can return a valid answer with a reason a few words too long.
    Only string_too_long errors are repaired; any other error is raised.
    """

    try:
        return response_model.model_validate_json(content)
    except ValidationError as error:
        errors = error.errors()

        if not errors or any(
            item["type"] != "string_too_long" for item in errors
        ):
            raise

        data = json.loads(content)

        for item in errors:
            container = data
            *parents, field = item["loc"]

            for key in parents:
                container = container[key]

            max_length = int(item["ctx"]["max_length"])
            container[field] = container[field][:max_length].rstrip()

        return response_model.model_validate(data)


class OpenAIStructuredClient:
    """Call one OpenAI chat model and validate its JSON response with Pydantic.

    Exposes the same invoke() interface as OllamaStructuredClient, so graph
    nodes do not depend on which provider is configured.
    """

    def __init__(
        self,
        model: str,
        num_predict: int = 500,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("OpenAI model cannot be empty.")

        if num_predict <= 0:
            raise ValueError("num_predict must be positive.")

        if max_retries <= 0:
            raise ValueError("max_retries must be positive.")

        self.model = model
        self.num_predict = num_predict
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        if client is None:
            from openai import OpenAI

            resolved_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()

            if not resolved_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to .env or set "
                    "agentic_rag.provider to 'ollama'."
                )

            # Retries are handled in invoke() so validation failures and
            # API errors share one retry budget.
            client = OpenAI(
                api_key=resolved_key,
                timeout=timeout_seconds,
                max_retries=0,
            )

        self.client = client

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        client: Any | None = None,
    ) -> OpenAIStructuredClient:
        """Create the client from the agentic_rag YAML section."""

        return cls(
            model=str(config.get("openai_model", "gpt-4.1-mini")),
            num_predict=int(config.get("num_predict", 500)),
            temperature=float(config.get("temperature", 0.0)),
            seed=int(config.get("seed", 42)),
            max_retries=int(config.get("max_retries", 3)),
            retry_delay_seconds=float(config.get("retry_delay_seconds", 2.0)),
            timeout_seconds=float(config.get("openai_timeout_seconds", 60.0)),
            client=client,
        )

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
        num_predict: int | None = None,
    ) -> ResponseModel:
        """Return a response that conforms to response_model."""

        clean_system_prompt = system_prompt.strip()
        clean_user_prompt = user_prompt.strip()

        if not clean_system_prompt:
            raise ValueError("system_prompt cannot be empty.")

        if not clean_user_prompt:
            raise ValueError("user_prompt cannot be empty.")

        output_tokens = self.num_predict

        if num_predict is not None:
            if num_predict <= 0:
                raise ValueError("num_predict must be positive.")

            output_tokens = num_predict

        # Non-strict json_schema accepts the full Pydantic schema, including
        # minLength/maxLength, which strict mode rejects. Pydantic still
        # validates every response below.
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": False,
            },
        }

        last_error: Exception | None = None

        for attempt_number in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": clean_system_prompt},
                        {"role": "user", "content": clean_user_prompt},
                    ],
                    response_format=response_format,
                    temperature=self.temperature,
                    seed=self.seed,
                    max_completion_tokens=output_tokens,
                )

                content = response.choices[0].message.content or ""
                return truncate_overlong_strings(content, response_model)

            except Exception as error:
                last_error = error

            if attempt_number < self.max_retries:
                wait_seconds = self.retry_delay_seconds * attempt_number
                time.sleep(wait_seconds)

        raise RuntimeError(
            "OpenAI structured request failed after "
            + str(self.max_retries)
            + " attempts: "
            + str(last_error)
        )
