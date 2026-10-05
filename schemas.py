from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    GEMINI = "gemini"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    timeout: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model no puede estar vacío")
        return value


class ModelResponse(BaseModel):
    ok: bool
    provider: Provider
    model: str
    content: str = ""
    error: str | None = None
    error_type: str | None = None
