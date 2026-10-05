from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Provider


QUESTION = "¿Qué es la entropía? Explícalo en pocas palabras."


def config_from_env() -> ModelConfig:
    provider = Provider(os.getenv("LLM_PROVIDER", "openai"))
    model = os.getenv("LLM_MODEL", "gpt-5.5")

    return ModelConfig(
        provider=provider,
        model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
        top_p=float(os.getenv("LLM_TOP_P", "1.0")),
    )


async def main() -> None:
    load_dotenv()

    config = config_from_env()
    manager = AsyncLLMManager(config)

    messages = [
        ChatMessage(role="system", content="Responde en castellano argentino de forma clara."),
        ChatMessage(role="user", content=QUESTION),
    ]

    print("=" * 70)
    print(f"PROVEEDOR: {config.provider.value} | MODELO: {config.model}")
    print("=" * 70)

    print("\n1) GENERACIÓN NORMAL\n")
    response = await manager.generate(messages)

    if response.ok:
        print(response.content)
    else:
        print(f"[ERROR:{response.error_type}] {response.error}")

    print("\n2) STREAMING\n")
    async for chunk in manager.stream(messages):
        print(chunk, end="", flush=True)
    print("\n")

    await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
