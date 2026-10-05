from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from anthropic import AsyncAnthropic
from groq import AsyncGroq
from openai import AsyncOpenAI
from google import genai
from google.genai import types

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider
from utils import safe_error


def _openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            content = response.choices[0].message.content or ""
            return ModelResponse(
                ok=True,
                provider=Provider.OPENAI,
                model=self.config.model,
                content=content,
            )
        except Exception as exc:
            error_type, error = safe_error(exc)
            return ModelResponse(
                ok=False,
                provider=Provider.OPENAI,
                model=self.config.model,
                error=error,
                error_type=error_type,
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield text
        except Exception as exc:
            error_type, error = safe_error(exc)
            yield f"\n[ERROR:{error_type}] {error}\n"


class GroqClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncGroq(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            content = response.choices[0].message.content or ""
            return ModelResponse(
                ok=True,
                provider=Provider.GROQ,
                model=self.config.model,
                content=content,
            )
        except Exception as exc:
            error_type, error = safe_error(exc)
            return ModelResponse(
                ok=False,
                provider=Provider.GROQ,
                model=self.config.model,
                error=error,
                error_type=error_type,
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield text
        except Exception as exc:
            error_type, error = safe_error(exc)
            yield f"\n[ERROR:{error_type}] {error}\n"


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @staticmethod
    def _split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
        system_parts = [m.content for m in messages if m.role.value == "system"]
        chat_messages = [
            {"role": "user" if m.role.value == "user" else "assistant", "content": m.content}
            for m in messages
            if m.role.value != "system"
        ]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, chat_messages

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            system, chat_messages = self._split_system(messages)
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": chat_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            if system:
                kwargs["system"] = system

            response = await self.client.messages.create(**kwargs)
            content = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            return ModelResponse(
                ok=True,
                provider=Provider.ANTHROPIC,
                model=self.config.model,
                content=content,
            )
        except Exception as exc:
            error_type, error = safe_error(exc)
            return ModelResponse(
                ok=False,
                provider=Provider.ANTHROPIC,
                model=self.config.model,
                error=error,
                error_type=error_type,
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            system, chat_messages = self._split_system(messages)
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": chat_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            if system:
                kwargs["system"] = system

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:
            error_type, error = safe_error(exc)
            yield f"\n[ERROR:{error_type}] {error}\n"


class GeminiClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _contents(messages: Sequence[ChatMessage]) -> tuple[str | None, list[types.Content]]:
        system_parts = [m.content for m in messages if m.role.value == "system"]
        contents: list[types.Content] = []

        for message in messages:
            if message.role.value == "system":
                continue
            role = "model" if message.role.value == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.content)],
                )
            )

        system = "\n\n".join(system_parts) if system_parts else None
        return system, contents

    def _config(self, system: str | None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
        ),
        
    )

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            system, contents = self._contents(messages)
            response = await self.client.aio.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=self._config(system),
            )
            return ModelResponse(
                ok=True,
                provider=Provider.GEMINI,
                model=self.config.model,
                content=response.text or "",
            )
        except Exception as exc:
            error_type, error = safe_error(exc)
            return ModelResponse(
                ok=False,
                provider=Provider.GEMINI,
                model=self.config.model,
                error=error,
                error_type=error_type,
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            system, contents = self._contents(messages)
            stream = await self.client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=self._config(system),
            )
            async for chunk in stream:
                text = chunk.text or ""
                if text:
                    yield text
        except Exception as exc:
            error_type, error = safe_error(exc)
            yield f"\n[ERROR:{error_type}] {error}\n"

    async def close(self) -> None:
        await self.client.aio.aclose()
