from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from base_client import BaseLLMClient
from clients import AnthropicClient, GeminiClient, GroqClient, OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class AsyncLLMManager:
    """Punto de entrada único para OpenAI, Anthropic, Groq y Gemini."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.client = self._build_client(config)

    @staticmethod
    def _required_key(provider: Provider) -> str:
        env_names = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.GROQ: "GROQ_API_KEY",
            Provider.GEMINI: "GEMINI_API_KEY",
        }
        env_name = env_names[provider]
        value = os.getenv(env_name)
        if not value:
            raise RuntimeError(
                f"Falta {env_name}. Copia .env.example a .env y agrega tu API key."
            )
        return value

    def _build_client(self, config: ModelConfig) -> BaseLLMClient:
        api_key = self._required_key(config.provider)

        if config.provider == Provider.OPENAI:
            return OpenAIClient(config, api_key)
        if config.provider == Provider.ANTHROPIC:
            return AnthropicClient(config, api_key)
        if config.provider == Provider.GROQ:
            return GroqClient(config, api_key)
        if config.provider == Provider.GEMINI:
            return GeminiClient(config, api_key)

        raise ValueError(f"Proveedor no soportado: {config.provider}")

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async for chunk in self.client.stream(messages):
            yield chunk

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if result is not None:
                await result
