from __future__ import annotations

import time
from typing import Any, TypeVar

from ollama import Client
from pydantic import BaseModel


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OllamaStructuredClient:
    """Call one Ollama model and validate its JSON response with Pydantic."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        context_window: int = 8192,
        num_predict: int = 500,
        temperature: float = 0.0,
        seed: int = 42,
        keep_alive: str = "30m",
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        think: bool = False,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model cannot be empty.")

        if context_window <= 0:
            raise ValueError("context_window must be positive.")

        if num_predict <= 0:
            raise ValueError("num_predict must be positive.")

        if max_retries <= 0:
            raise ValueError("max_retries must be positive.")

        self.model = model
        self.context_window = context_window
        self.num_predict = num_predict
        self.temperature = temperature
        self.seed = seed
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.think = think

        if client is None:
            client = Client(host=host)

        self.client = client

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        client: Any | None = None,
    ) -> "OllamaStructuredClient":
        """Create the client from the agentic_rag YAML section."""

        return cls(
            model=str(config["model"]),
            host=str(config.get("host", "http://localhost:11434")),
            context_window=int(config.get("context_window", 8192)),
            num_predict=int(config.get("num_predict", 500)),
            temperature=float(config.get("temperature", 0.0)),
            seed=int(config.get("seed", 42)),
            keep_alive=str(config.get("keep_alive", "30m")),
            max_retries=int(config.get("max_retries", 3)),
            retry_delay_seconds=float(
                config.get("retry_delay_seconds", 2.0)
            ),
            think=bool(config.get("think", False)),
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

        last_error: Exception | None = None

        for attempt_number in range(1, self.max_retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": clean_system_prompt,
                        },
                        {
                            "role": "user",
                            "content": clean_user_prompt,
                        },
                    ],
                    format=response_model.model_json_schema(),
                    options={
                        "temperature": self.temperature,
                        "seed": self.seed,
                        "num_ctx": self.context_window,
                        "num_predict": output_tokens,
                    },
                    think=self.think,
                    stream=False,
                    keep_alive=self.keep_alive,
                )

                content = response.message.content
                return response_model.model_validate_json(content)

            except Exception as error:
                last_error = error

            if attempt_number < self.max_retries:
                wait_seconds = self.retry_delay_seconds * attempt_number
                time.sleep(wait_seconds)

        raise RuntimeError(
            "Ollama structured request failed after "
            + str(self.max_retries)
            + " attempts: "
            + str(last_error)
        )
