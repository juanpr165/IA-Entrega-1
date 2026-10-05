from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Interfaz común para todos los proveedores."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa sin bloquear el event loop."""

    @abstractmethod
    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Produce fragmentos de texto conforme llegan del proveedor."""
        yield ""
